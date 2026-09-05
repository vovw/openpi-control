"""Small wrapper around adb for USB Quest relay forwarding."""

from __future__ import annotations

import shutil
import subprocess

from .exceptions import ConfigurationError


def connect_quest(serial: str | None = None, port: int = 8443, open_page: bool = False) -> int:
    if not 1 <= port <= 65535:
        raise ConfigurationError("--port must be between 1 and 65535")
    adb = shutil.which("adb")
    if not adb:
        raise ConfigurationError("adb is not installed. On Ubuntu: sudo apt install adb")

    def run(*arguments):
        try:
            result = subprocess.run([adb, *arguments], capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as err:
            raise ConfigurationError(f"adb failed: {err}") from err
        if result.returncode:
            raise ConfigurationError(
                f"adb failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout

    devices = {}
    for line in run("devices", "-l").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] in ("device", "unauthorized", "offline"):
            devices[fields[0]] = fields[1]
    if serial is None:
        if not devices:
            raise ConfigurationError(
                "No Android device found. Connect Quest by USB and enable USB debugging."
            )
        if len(devices) > 1:
            raise ConfigurationError(
                "Multiple devices found; select --serial " + " / ".join(devices)
            )
        serial = next(iter(devices))
    state = devices.get(serial)
    if state == "unauthorized":
        raise ConfigurationError(
            "Quest is unauthorized. Put on the headset and accept Allow USB debugging."
        )
    if state != "device":
        raise ConfigurationError(f"Device {serial} is {state or 'not connected'}; reconnect USB.")
    run("-s", serial, "reverse", f"tcp:{port}", f"tcp:{port}")
    url = f"http://localhost:{port}/"
    if open_page:
        run("-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url)
    print(f"Quest USB connected: {serial}\nRelay forwarded: {url}")
    print("In Quest, open that address and select Start Teleop. The relay must be running.")
    print("Check controller input: uv run openpi-control health vr")
    return 0
