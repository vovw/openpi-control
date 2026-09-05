"""USB transport for the native Quest controller streamer (no motor access)."""

from __future__ import annotations

import json
import math
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .exceptions import ConfigurationError

PACKAGE = "org.openpi.queststreamer"
ACTIVITY = f"{PACKAGE}/android.app.NativeActivity"
MAX_AGE = 0.25


def _run(command: list[str], timeout: float = 15) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as err:
        raise ConfigurationError(f"adb failed: {err}") from err
    if result.returncode:
        raise ConfigurationError(f"adb failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def _device(serial: str | None) -> list[str]:
    adb = shutil.which("adb")
    if not adb:
        raise ConfigurationError("adb is not installed. On Ubuntu: sudo apt install adb")
    devices = {}
    for line in _run([adb, "devices", "-l"]).splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[1] in {"device", "offline", "unauthorized"}:
            devices[fields[0]] = fields[1]
    if serial is None:
        if len(devices) != 1:
            raise ConfigurationError("Connect one Quest by USB, or select its --serial.")
        serial = next(iter(devices))
    if devices.get(serial) != "device":
        raise ConfigurationError(
            f"Quest {serial}: {devices.get(serial, 'not connected')}. "
            "Reconnect USB and accept Allow USB debugging in the headset."
        )
    return [adb, "-s", serial]


def install_apk(apk: Path, serial: str | None = None) -> int:
    apk = Path(apk).expanduser().resolve()
    if not apk.is_file() or apk.suffix.lower() != ".apk":
        raise ConfigurationError(f"APK file not found: {apk}")
    print(_run([*_device(serial), "install", "-r", str(apk)], timeout=120).strip())
    return 0


def stop_native(serial: str | None = None) -> int:
    _run([*_device(serial), "shell", "am", "force-stop", PACKAGE])
    return 0


def _clock_offset(adb: list[str]) -> float:
    """Map device realtime seconds onto host monotonic, with bounded RTT."""
    for _ in range(3):
        before = time.monotonic()
        raw = _run([*adb, "shell", "date", "+%s%N"])
        after = time.monotonic()
        try:
            stamp = int(raw.strip()) / 1e9
        except ValueError as err:
            raise ConfigurationError("Quest date does not provide nanosecond timestamps") from err
        if after - before <= 0.2 and stamp > 1e9:
            return (before + after) / 2 - stamp
    raise ConfigurationError("Quest USB timestamp calibration is too slow; reconnect USB.")


def _finite_vector(value, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(type(x) in (int, float) and math.isfinite(x) for x in value)
    )


class FrameFilter:
    """Drop old/duplicate frames and remove controllers without tracked poses."""

    def __init__(self, offset: float, started: float):
        self.offset = offset
        self.started = started
        self.last_stamp = 0
        self.session = None
        self.release_required = {"left": True, "right": True}

    def parse(self, raw: str, now: float) -> str | None:
        try:
            frame = json.loads(raw)
            if not isinstance(frame, dict) or frame.get("type") != "xr_frame":
                return None
            stamp = frame.get("t_unix_ns")
            if type(stamp) is not int or stamp <= self.last_stamp:
                return None
            sample_time = stamp / 1e9 + self.offset
            if sample_time < self.started or not -0.1 <= now - sample_time <= MAX_AGE:
                return None
            controllers = frame.get("controllers")
            if not isinstance(controllers, dict):
                return None
            viewer = frame.get("viewer")
            if frame.get("focused") is True:
                if (
                    not isinstance(viewer, dict)
                    or not _finite_vector(viewer.get("position"), 3)
                    or not _finite_vector(viewer.get("orientation"), 4)
                    or not 0.9 <= sum(x * x for x in viewer["orientation"]) <= 1.1
                ):
                    self.release_required = {"left": True, "right": True}
                    return None
            session = frame.get("session")
            if session != self.session:
                self.release_required = {"left": True, "right": True}
                self.session = session
            if self.last_stamp and (stamp - self.last_stamp) / 1e9 > MAX_AGE:
                self.release_required = {"left": True, "right": True}
            clean = {}
            for hand in ("left", "right"):
                controller = controllers.get(hand)
                if not isinstance(controller, dict):
                    continue
                if (
                    frame.get("focused") is not True
                    or controller.get("active") is not True
                    or controller.get("valid") is not True
                    or controller.get("tracked") is not True
                ):
                    continue
                if not _finite_vector(controller.get("position"), 3):
                    continue
                orientation = controller.get("orientation")
                if not _finite_vector(orientation, 4):
                    continue
                if not 0.9 <= sum(x * x for x in orientation) <= 1.1:
                    continue
                buttons = controller.get("buttons")
                if not isinstance(buttons, list) or len(buttons) != 6:
                    continue
                if any(
                    not isinstance(b, dict)
                    or type(b.get("p")) is not bool
                    or type(b.get("t")) is not bool
                    or type(b.get("v")) not in (int, float)
                    or not math.isfinite(b["v"])
                    or not 0 <= b["v"] <= 1
                    for b in buttons
                ):
                    continue
                if not _finite_vector(controller.get("axes"), 4):
                    continue
                if self.release_required[hand]:
                    if not buttons[1]["p"] and buttons[1]["v"] < 0.1:
                        self.release_required[hand] = False
                    else:
                        buttons[1] = {"p": False, "t": False, "v": 0.0}
                clean[hand] = controller
            for hand in self.release_required:
                if hand not in clean:
                    self.release_required[hand] = True
            frame["controllers"] = clean
            self.last_stamp = stamp
            return json.dumps(frame, allow_nan=False, separators=(",", ":"))
        except (ValueError, TypeError, OverflowError):
            return None


def stream_native(serial: str | None = None, url: str = "ws://127.0.0.1:8443/ws") -> int:
    from websockets.sync.client import connect

    adb = _device(serial)
    offset = _clock_offset(adb)
    frames: queue.Queue[str] = queue.Queue(maxsize=1)
    stopped = threading.Event()
    reader_done = threading.Event()
    relay_done = threading.Event()
    process = None
    reader = None
    receiver = None
    try:
        with connect(url, open_timeout=3, close_timeout=1, max_queue=1) as ws:
            started = time.monotonic()
            check = FrameFilter(offset, started)
            process = subprocess.Popen(
                [*adb, "logcat", "-T", "1", "-v", "raw", "OpenpiXR:I", "*:S"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )

            def read_frames():
                try:
                    for line in process.stdout:
                        if stopped.is_set():
                            break
                        try:
                            frames.put_nowait(line)
                        except queue.Full:
                            try:
                                frames.get_nowait()
                            except queue.Empty:
                                pass
                            frames.put_nowait(line)
                finally:
                    reader_done.set()

            def drain_relay():
                try:
                    while not stopped.is_set():
                        try:
                            ws.recv(timeout=0.2)
                        except TimeoutError:
                            pass
                except Exception:
                    pass
                finally:
                    relay_done.set()

            reader = threading.Thread(target=read_frames, daemon=True)
            receiver = threading.Thread(target=drain_relay, daemon=True)
            reader.start()
            receiver.start()
            result = _run([*adb, "shell", "am", "start", "-W", "-n", ACTIVITY])
            if "Error:" in result or "Error type" in result:
                raise ConfigurationError(f"Quest app could not start: {result.strip()}")
            print(f"Native Quest → USB → {url}. Ctrl-C stops the streamer.")
            count = 0
            report = time.monotonic()
            while not reader_done.is_set() and not relay_done.is_set():
                try:
                    raw = frames.get(timeout=0.1)
                except queue.Empty:
                    raw = None
                payload = check.parse(raw, time.monotonic()) if raw is not None else None
                if payload is not None:
                    ws.send(payload)
                    count += 1
                if time.monotonic() - report >= 5:
                    print(f"Native Quest: {count} fresh frames forwarded in 5s")
                    count = 0
                    report = time.monotonic()
            raise ConfigurationError("Quest USB log stream or VR relay disconnected.")
    except KeyboardInterrupt:
        return 0
    except ConfigurationError:
        raise
    except Exception as err:
        raise ConfigurationError(f"Native Quest stream failed: {err}") from err
    finally:
        stopped.set()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            if process.stdout:
                process.stdout.close()
        for thread in (reader, receiver):
            if thread is not None:
                thread.join(timeout=1)
        try:
            _run([*adb, "shell", "am", "force-stop", PACKAGE])
        except ConfigurationError as err:
            print(f"warning: could not stop Quest app: {err}")
