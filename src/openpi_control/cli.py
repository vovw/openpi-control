"""Operator CLI: preflight an arm, and zero its servos.

Two commands, both aimed at the jobs you do before an arm is usable:

``doctor``
    Read-only checks: do the packaged assets resolve, is every servo model
    known to the driver registry, does the bus interface exist and is it up at
    the baud rate the model wants. Static by default; ``--probe`` additionally
    opens the bus and listens.

``zero``
    Write the arm's current physical pose into each servo's firmware as its
    zero reference, via the per-family drivers in :mod:`openpi_control.servos`.
    Destructive and pose-dependent, so it confirms before touching anything.

``live``
    Bring a whole rig up, mirror it in the browser -- arms and cameras on the
    one page -- then park it and put it back down. This is the one command here
    that energizes an arm, and it owns that lifecycle start to finish:
    ``pi_control_node`` dies with its parent process, so an arm cannot outlive
    the command that powered it on.

``cameras``
    Resolve the rig's cameras to device paths and say which ones are actually
    on the bus. Read-only; ``--probe`` additionally opens each stream and
    grabs a frame, and ``--snapshot`` writes those frames out so you can check
    where a wrist camera is pointing without a headset.

All of them attach :func:`openpi_control.runlog.setup_run_logging`, so every run
leaves a trace under ``~/openpi-data/logs/runtime/``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from . import cameras as cameras_mod
from . import meshes, runlog
from .config import (
    SUPPORTED_EFFECTORS,
    SUPPORTED_MODELS,
    ArmConfig,
    connection_for_interface,
    resolve_model_assets,
)
from .exceptions import ConfigurationError
from .rigs import Rig, RigArm, resolve_rig, rig_names
from .servos import SERVO_ZERO_DRIVERS, buses

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from .arms import FollowerArm, LeaderArm
    from .backend import ArmBackend
    from .session import ArmSession

_OK = "ok"
_WARN = "warn"
_FAIL = "fail"

_MARKS = {_OK: "OK  ", _WARN: "WARN", _FAIL: "FAIL"}

# How long --probe listens for traffic before giving up. Silence is not proof of
# a fault (DM servos answer requests rather than broadcasting), so this only
# ever reports what it saw.
_PROBE_WINDOW_S = 1.5

# How often ``live`` pushes each arm's newest pose into the browser. The native
# node publishes faster than this; the browser has no use for more.
_MIRROR_RATE_HZ = 30.0

# How long a camera gets to deliver its first frame before ``--probe`` calls it
# dead; how long the stream is then left alone to settle; and how long the
# probe counts frames to measure the real rate. A D405 needs a few hundred ms
# after the first frame before it is streaming at rate, and measuring across
# that ramp reports ~10 fps for a camera that is in fact doing 28.
_CAMERA_WARMUP_S = 5.0
_CAMERA_SETTLE_S = 1.0
_CAMERA_COUNT_S = 2.0

# Fraction of the requested rate a camera has to hit before the probe stops
# complaining. Loose on purpose: v4l2 delivers 27-28 of a requested 30 on a
# healthy bus, while a camera sharing a USB 2 hub with two others lands far
# below this.
_CAMERA_FPS_TOLERANCE = 0.8


@dataclass(frozen=True, slots=True)
class ServoEntry:
    """One servo declared by an arm or effector model catalog."""

    source: str  # "arm" or the effector model name
    joint_id: int
    servo_model: str
    servo_id: int
    driver: ModuleType | None

    @property
    def read_only(self) -> bool:
        """True for encoders whose zero is fixed in hardware — nothing to write."""
        return self.driver is None

    def label(self) -> str:
        return f"joint {self.joint_id} ({self.source})"


@dataclass(frozen=True, slots=True)
class CheckResult:
    status: str
    name: str
    detail: str

    def render(self) -> str:
        return f"  [{_MARKS[self.status]}] {self.name:<26} {self.detail}"


def servo_entries(config_path: Path, source: str) -> list[ServoEntry]:
    """Every servo a device catalog declares, in joint order."""
    data = json.loads(config_path.read_text())
    entries: list[ServoEntry] = []
    for index, joint in enumerate(data.get("joints", []), start=1):
        joint_id = int(joint.get("joint_id", index))
        for servo in joint.get("servos", []):
            servo_model = str(servo.get("servo_model", ""))
            if not servo_model:
                raise ConfigurationError(
                    f"{config_path.name} joint {joint_id} has a servo with no servo_model"
                )
            if servo_model not in SERVO_ZERO_DRIVERS:
                raise ConfigurationError(
                    f"servo model {servo_model!r} ({config_path.name} joint {joint_id}) is not in "
                    "the servo registry (openpi_control/servos): add it to SERVO_ZERO_DRIVERS"
                )
            entries.append(
                ServoEntry(
                    source=source,
                    joint_id=joint_id,
                    servo_model=servo_model,
                    servo_id=int(servo.get("servo_id", joint_id)),
                    driver=SERVO_ZERO_DRIVERS[servo_model],
                )
            )
    return entries


def build_plan(model: str, effector_model: str | None = None) -> tuple[ServoEntry, ...]:
    """The servos of an arm plus its effector, ordered by joint id."""
    assets = resolve_model_assets(model, effector_model=effector_model)
    entries = servo_entries(assets.model_config, "arm")
    if assets.effector_model_config is not None and effector_model is not None:
        entries += servo_entries(assets.effector_model_config, effector_model)
    if not entries:
        raise ConfigurationError(f"model {model!r} declares no servos")
    return tuple(sorted(entries, key=lambda entry: entry.joint_id))


def plan_port_type(plan: tuple[ServoEntry, ...]) -> str | None:
    """The single bus type the writable servos need, or None when all are read-only."""
    port_types = {entry.driver.PORT_TYPE for entry in plan if entry.driver is not None}
    if not port_types:
        return None
    if len(port_types) > 1:
        raise ConfigurationError(
            "this arm's servos span more than one bus type "
            f"({', '.join(sorted(port_types))}); zero them one bus at a time"
        )
    return port_types.pop()


def _can_sysfs(interface: str, leaf: str) -> str | None:
    path = pathlib.Path(f"/sys/class/net/{interface}/{leaf}")
    try:
        return path.read_text().strip()
    except OSError:
        return None


def can_bitrate(interface: str) -> int | None:
    """The configured bitrate of a SocketCAN interface, or None if unknown.

    Two sources, because neither is universal. ``/sys/class/net/<if>/
    can_bittiming/bitrate`` is the cheap one, but it does not exist on every
    kernel -- it is absent on 6.8 here, which made this check warn "interface
    does not report one" on every single run of a correctly configured 1 Mbit
    bus. A preflight that always warns is one nobody reads, so fall back to
    ``ip``, which reads the same value over netlink and is where ``ip -d link
    show`` gets it from.
    """
    from_sysfs = _can_sysfs(interface, "can_bittiming/bitrate")
    if from_sysfs:
        try:
            return int(from_sysfs)
        except ValueError:
            pass
    try:
        completed = subprocess.run(
            ["ip", "-d", "-json", "link", "show", interface],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        links = json.loads(completed.stdout)
        bitrate = links[0]["linkinfo"]["info_data"]["bittiming"]["bitrate"]
    except (json.JSONDecodeError, LookupError, TypeError):
        return None
    return int(bitrate) if bitrate else None


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def run_doctor(
    model: str,
    interface: str,
    *,
    effector_model: str | None = None,
    probe: bool = False,
) -> list[CheckResult]:
    """Collect the preflight checks. Opens no bus unless ``probe`` is set."""
    results: list[CheckResult] = []

    try:
        assets = resolve_model_assets(model, effector_model=effector_model)
    except ConfigurationError as err:
        return [CheckResult(_FAIL, "packaged assets", str(err))]
    results.append(
        CheckResult(
            _OK,
            "packaged assets",
            f"{assets.model_config.name}, {assets.instance_config.name}"
            + (f", {assets.effector_model_config.name}" if assets.effector_model_config else ""),
        )
    )
    results.append(
        CheckResult(
            _OK if assets.urdf is not None else _WARN,
            "urdf",
            assets.urdf.name
            if assets.urdf is not None
            else "none packaged — gravity compensation needs one (FR3 uses its controller)",
        )
    )

    try:
        plan = build_plan(model, effector_model)
    except ConfigurationError as err:
        results.append(CheckResult(_FAIL, "servo registry", str(err)))
        return results

    writable = [entry for entry in plan if not entry.read_only]
    read_only = [entry for entry in plan if entry.read_only]
    detail = f"{len(plan)} servos, all models known"
    if read_only:
        detail += f" ({len(read_only)} read-only: " + ", ".join(
            f"joint {entry.joint_id}" for entry in read_only
        ) + ")"
    results.append(CheckResult(_OK, "servo registry", detail))

    try:
        port_type = plan_port_type(plan)
    except ConfigurationError as err:
        results.append(CheckResult(_FAIL, "bus type", str(err)))
        return results
    if port_type is None:
        results.append(
            CheckResult(_WARN, "bus type", "every servo is read-only; nothing to zero")
        )
        return results
    results.append(CheckResult(_OK, "bus type", port_type))

    # The connection type the interface string implies must match what the
    # servos need, or the arm is configured against the wrong transport.
    connection = connection_for_interface(interface)
    results.append(CheckResult(_OK, "connection", type(connection).__name__))

    problem = buses.check_interface(port_type, interface)
    results.append(
        CheckResult(
            _FAIL if problem else _OK,
            f"interface {interface}",
            problem or "present",
        )
    )

    if port_type == buses.PORT_TYPE_CAN and problem is None:
        state = _can_sysfs(interface, "operstate")
        results.append(
            CheckResult(
                _OK if state == "up" else _FAIL,
                "link state",
                state or "unknown",
            )
        )
        want = _catalog_baudrate(assets.model_config)
        have = can_bitrate(interface)
        if want is None:
            results.append(CheckResult(_WARN, "bitrate", "model declares no catalog baudrate"))
        elif have is None:
            results.append(
                CheckResult(_WARN, "bitrate", f"model wants {want}; interface does not report one")
            )
        else:
            results.append(
                CheckResult(
                    _OK if have == want else _FAIL,
                    "bitrate",
                    f"{have} (model wants {want})",
                )
            )

    if port_type == buses.PORT_TYPE_SERIAL:
        want = _catalog_baudrate(assets.model_config)
        results.append(
            CheckResult(
                _OK if want else _FAIL,
                "baudrate",
                str(want) if want else "serial buses need a catalog baudrate",
            )
        )

    # An ArmConfig is what a session would be handed; building one here catches
    # model/connection mismatches before any hardware is involved.
    try:
        ArmConfig("preflight", model, connection, effector_model=effector_model)
        results.append(CheckResult(_OK, "arm config", "valid"))
    except ConfigurationError as err:
        results.append(CheckResult(_WARN, "arm config", str(err)))

    cached = meshes.cached_mesh_dir(model)
    results.append(
        CheckResult(
            _OK if cached else _WARN,
            "visual meshes",
            str(cached)
            if cached
            else f"not cached — openpi-control-viz --fetch-meshes --model {model}",
        )
    )

    if probe:
        results.append(_probe_bus(port_type, interface, writable, assets.model_config))
    return results


def _catalog_baudrate(model_config: Path) -> int | None:
    data = json.loads(model_config.read_text())
    value = data.get("catalog", {}).get("baudrate")
    return int(value) if isinstance(value, int) and value > 0 else None


def _probe_bus(
    port_type: str, interface: str, writable: list[ServoEntry], model_config: Path
) -> CheckResult:
    """Open the bus and listen. Reads only — never sends."""
    expected = tuple(entry.servo_id for entry in writable)
    try:
        with buses.open_bus(
            port_type, interface, baudrate=_catalog_baudrate(model_config)
        ) as bus:
            if port_type != buses.PORT_TYPE_CAN:
                return CheckResult(_OK, "bus probe", f"{port_type} session opened")
            answered = buses.recv_from(bus, expected, _PROBE_WINDOW_S)
    except Exception as err:  # noqa: BLE001 - surface any driver error as a check
        return CheckResult(_FAIL, "bus probe", f"{type(err).__name__}: {err}")
    if answered:
        return CheckResult(_OK, "bus probe", "traffic seen from a configured servo id")
    # DM servos answer requests rather than broadcasting, so quiet is normal on
    # an idle bus. Reporting this as a failure would cry wolf.
    return CheckResult(
        _WARN,
        "bus probe",
        f"opened, no traffic in {_PROBE_WINDOW_S}s — normal on an idle bus",
    )


# --------------------------------------------------------------------------- #
# cameras
# --------------------------------------------------------------------------- #


def run_camera_checks(
    rig: Rig,
    *,
    overrides: dict[str, str] | None = None,
    required: bool = False,
) -> list[CheckResult]:
    """One check per rig camera: did its serial resolve to a device path?

    A camera that is not on the bus is a warning, not a failure, because none
    of them are needed to drive an arm -- ``doctor`` should not refuse to
    green-light a bimanual cell because someone unplugged a wrist camera. The
    dataset recorder passes ``required=True``, where a missing camera really is
    fatal: it would otherwise write an episode with a view silently absent.
    """
    mark = _FAIL if required else _WARN
    results: list[CheckResult] = []
    if not rig.cameras:
        return [CheckResult(_OK, "cameras", "this rig declares none")]

    result = cameras_mod.discover(rig.cameras, overrides=overrides)
    for camera in rig.cameras:
        found = result.matched.get(camera.name)
        if found is None:
            results.append(
                CheckResult(
                    mark,
                    f"camera {camera.name}",
                    f"serial {camera.serial} not on the bus"
                    if camera.name not in (overrides or {})
                    else f"pinned device {(overrides or {})[camera.name]} does not exist",
                )
            )
            continue
        detail = f"{camera.width}x{camera.height}@{camera.fps}"
        if camera.rotate:
            detail += f" rot{camera.rotate}"
        detail += f" — {found.device}"
        if found.overridden:
            detail += " (pinned)"
        results.append(CheckResult(_OK, f"camera {camera.name}", detail))

    if result.unclaimed:
        # A camera on the bus that no rig claims is usually the interesting
        # half of "the top view is missing": it was swapped, and the new serial
        # needs to go into the rig.
        results.append(
            CheckResult(
                _WARN,
                "unclaimed cameras",
                f"serial(s) {', '.join(result.unclaimed)} present but in no rig camera — "
                "add them to YAM_BIMANUAL_CAMERA_SERIALS in rigs.py",
            )
        )
    return results


def check_camera_modes(rig: Rig) -> list[CheckResult]:
    """Can each camera actually deliver the mode the run is asking for?

    Kept out of :func:`run_camera_checks` because answering it needs the
    RealSense SDK, and ``doctor`` must keep working on a box without it. This
    only enumerates profiles -- it opens no stream -- so it is cheap enough to
    sit in a preflight, and it turns "you asked for 45 fps" into the list of
    rates the device has rather than a bare ioctl failure once the arms are
    already energized.
    """
    results: list[CheckResult] = []
    for camera in rig.cameras:
        label, size = f"mode {camera.name}", f"{camera.width}x{camera.height}"
        try:
            modes = cameras_mod.supported_color_modes(camera.serial)
        except ConfigurationError as err:
            # Not connected, or no SDK. Either way the presence checks already
            # said so; repeating it as a failure here would double-count.
            results.append(CheckResult(_WARN, label, str(err)))
            continue
        rates = modes.get((camera.width, camera.height))
        if rates is None:
            offered = ", ".join(f"{w}x{h}" for w, h in modes) or "nothing"
            detail = f"{size} is not offered; this camera has {offered}"
        elif camera.fps not in rates:
            detail = (
                f"{camera.fps} fps at {size} is not offered; this camera does "
                f"{', '.join(str(rate) for rate in rates)}"
            )
        else:
            results.append(
                CheckResult(
                    _OK,
                    label,
                    f"{size}@{camera.fps} {camera.pixel_format} "
                    f"(offers {', '.join(str(rate) for rate in rates)})",
                )
            )
            continue
        results.append(CheckResult(_FAIL, label, detail))
    return results


def probe_cameras(
    rig: Rig,
    *,
    overrides: dict[str, str] | None = None,
    snapshot_dir: Path | None = None,
    warmup_s: float = _CAMERA_WARMUP_S,
    settle_s: float = _CAMERA_SETTLE_S,
    count_s: float = _CAMERA_COUNT_S,
) -> list[CheckResult]:
    """Open each resolvable camera, grab frames, and report what came out.

    Opened one at a time on purpose. A v4l2 device opens once, so the useful
    error here is "this camera is held by something else", and holding all three
    at once would let the first camera's success hide the second's failure
    behind a shared USB bandwidth complaint.
    """
    results: list[CheckResult] = []
    discovery = cameras_mod.discover(rig.cameras, overrides=overrides)
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    for camera in rig.cameras:
        found = discovery.matched.get(camera.name)
        if found is None:
            results.append(
                CheckResult(_WARN, f"probe {camera.name}", "no device; nothing to open")
            )
            continue
        try:
            with cameras_mod.CameraReader(found.spec()) as reader:
                frame = reader.wait_for_frame(warmup_s)
                if frame is None:
                    results.append(
                        CheckResult(
                            _FAIL,
                            f"probe {camera.name}",
                            f"opened but delivered no frame in {warmup_s:g}s",
                        )
                    )
                    continue
                time.sleep(settle_s)
                measured = reader.measure_fps(count_s)
                fourcc, _, _, _ = reader.negotiated
                # Snapshot the freshest frame, not the warm-up one. `or` is not
                # an option here: a numpy array has no truth value.
                newest = reader.latest()
                if newest is not None:
                    frame = newest
                height, width = frame.shape[:2]
                detail = (
                    f"{width}x{height} {fourcc}, {measured:.0f} fps (asked {camera.fps})"
                )
                status = (
                    _OK if measured >= camera.fps * _CAMERA_FPS_TOLERANCE else _WARN
                )
                if snapshot_dir is not None:
                    path = snapshot_dir / f"{camera.name}.png"
                    cameras_mod.write_snapshot(path, frame)
                    detail += f" -> {path}"
                results.append(CheckResult(status, f"probe {camera.name}", detail))
        except Exception as err:  # noqa: BLE001 - surface any driver error as a check
            results.append(
                CheckResult(_FAIL, f"probe {camera.name}", f"{type(err).__name__}: {err}")
            )
    return results


# --------------------------------------------------------------------------- #
# zero
# --------------------------------------------------------------------------- #


def zero_arm(
    plan: tuple[ServoEntry, ...],
    port_type: str,
    interface: str,
    *,
    baudrate: int | None = None,
) -> list[tuple[ServoEntry, str | None]]:
    """Zero every writable servo in ``plan``; returns (entry, error or None) pairs.

    Read-only encoders are skipped rather than reported as failures, per the
    servo registry's contract. Whole-arm controllers zero every joint in one
    transaction, so they are called once.
    """
    outcomes: list[tuple[ServoEntry, str | None]] = []
    writable = [entry for entry in plan if not entry.read_only]
    if not writable:
        return outcomes

    whole_arm = [entry for entry in writable if getattr(entry.driver, "WHOLE_ARM_ZERO", False)]
    with buses.open_bus(port_type, interface, baudrate=baudrate) as bus:
        if whole_arm:
            driver = whole_arm[0].driver
            assert driver is not None
            error = driver.set_zero_whole_arm(bus)
            return [(entry, error) for entry in whole_arm]
        for entry in writable:
            driver = entry.driver
            assert driver is not None
            outcomes.append((entry, driver.set_zero(bus, entry.servo_id)))
    return outcomes


def _confirm(plan: tuple[ServoEntry, ...], model: str, interface: str) -> bool:
    writable = [entry for entry in plan if not entry.read_only]
    print(f"About to write a new firmware zero on {model} via {interface}:")
    for entry in writable:
        print(f"  {entry.label():<24} {entry.servo_model:<24} servo id {entry.servo_id}")
    skipped = [entry for entry in plan if entry.read_only]
    for entry in skipped:
        print(f"  {entry.label():<24} {entry.servo_model:<24} read-only, skipped")
    print()
    print("The arm's CURRENT physical pose becomes zero for every servo listed.")
    print("Move it to the intended zero pose and support it before continuing.")
    reply = input("Type 'zero' to proceed: ").strip()
    return reply == "zero"


# --------------------------------------------------------------------------- #
# live
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LiveArm:
    """One energized arm, paired with the rig entry that placed it."""

    rig_arm: RigArm
    arm: FollowerArm | LeaderArm

    @property
    def name(self) -> str:
        return self.rig_arm.name


def preflight_rig(rig: Rig) -> tuple[int, list[tuple[str, list[CheckResult]]]]:
    """Run the doctor checks for every arm in a rig. Returns (failures, per-arm).

    Nothing is energized until every arm passes, so a rig with one unplugged
    adapter does not come half up: bringing the good arm to a hold and then
    aborting would leave it stiff with no session left to put it down.
    """
    failures = 0
    reports: list[tuple[str, list[CheckResult]]] = []
    for rig_arm in rig.arms:
        results = run_doctor(
            rig_arm.model, rig_arm.interface, effector_model=rig_arm.effector_model
        )
        failures += sum(1 for result in results if result.status == _FAIL)
        reports.append((rig_arm.name, results))
    return failures, reports


def power_up(
    rig: Rig,
    *,
    float_mode: bool = False,
    backend_factory: Callable[[RigArm], ArmBackend] | None = None,
) -> tuple[ArmSession, list[LiveArm]]:
    """Start one session and energize every arm in the rig.

    Connecting *is* the power-on: the native node comes up holding whatever pose
    it found, so a follower goes stiff here. ``float_mode`` swaps that for the
    gravity feed-forward, which leaves the arm backdrivable -- an explicit
    choice, never the default, because a compliant arm sags if its torq_rescale
    is untuned. Leaders float either way; that is what ``ArmSession.connect``
    already does for them.

    ``backend_factory`` exists so this can be driven against a stand-in backend
    in tests without a bus or a native node.
    """
    from .session import ArmSession

    session = ArmSession()
    entries: list[LiveArm] = []
    for rig_arm in rig.arms:
        config = rig_arm.arm_config()
        backend = backend_factory(rig_arm) if backend_factory is not None else None
        arm: FollowerArm | LeaderArm = (
            session.add_follower(config, backend=backend)
            if rig_arm.is_follower
            else session.add_leader(config, backend=backend)
        )
        entries.append(LiveArm(rig_arm, arm))

    try:
        session.connect()
        if float_mode:
            for entry in entries:
                if not entry.rig_arm.is_follower:
                    continue
                if not entry.arm.capabilities.supports_gravity_compensation:
                    print(f"  {entry.name}: node has no gravity float; left holding")
                    continue
                entry.arm.enter_gravity_compensation()
    except BaseException:
        # A partial bring-up must not survive: whatever did connect is already
        # energized, and this session is the only thing that can put it down.
        session.close()
        raise
    return session, entries


def power_down(session: ArmSession, live_arms: list[LiveArm], *, park: bool = True) -> int:
    """Park each arm at its home pose, then de-energize. Returns the failure count.

    Parking rides the native MOVE_TO_READY_AND_SHUTDOWN path, so the arm drives
    to the ``home_pos`` in its instance JSON and the node exits at the end of
    that move -- it is never dropped from wherever it happened to be standing.
    A node that does not advertise CAP_MOVE_TO_READY is closed in place, said
    out loud rather than silently.
    """
    failures = 0
    if park:
        for entry in live_arms:
            try:
                supported = entry.arm.capabilities.supports_move_to_ready
            except Exception as err:  # noqa: BLE001 - keep putting the rest down
                failures += 1
                print(f"  {entry.name}: cannot check the ready move: {err}")
                continue
            if not supported:
                print(f"  {entry.name}: node has no ready move; closing in place")
                continue
            print(f"  parking {entry.name} -> home_pos ...", end="", flush=True)
            try:
                entry.arm.close(move_to_ready=True)
            except Exception as err:  # noqa: BLE001 - keep putting the rest down
                failures += 1
                print(f" FAILED: {type(err).__name__}: {err}")
            else:
                print(" done")

    # Always reached, parked or not: session.close() is what retires the nodes
    # and the sockets, and it is idempotent for an arm already closed above.
    try:
        session.close()
    except Exception as err:  # noqa: BLE001 - report, do not mask the shutdown
        failures += 1
        print(f"  power-down error: {type(err).__name__}: {err}")
    print("de-energized, with errors above" if failures else "de-energized")
    return failures


def open_preview_cameras(
    rig: Rig, *, overrides: dict[str, str] | None = None
) -> dict[str, cameras_mod.CameraReader]:
    """Open every camera of ``rig`` that can be opened, and say what was not.

    Deliberately not the recorder's all-or-none open: a dataset with a view
    silently missing is a corrupt dataset, but ``live`` exists to drive arms,
    and refusing to energize them because someone unplugged a wrist camera --
    or because another process is already streaming it -- would be the wrong
    trade. Every camera that opens is previewed; every one that does not is
    named.

    Opened one at a time for the same reason ``--probe`` does it: the useful
    error is "this camera is held by something else", and it has to name the
    camera it is actually about. Identical failures are collapsed into one
    line, because the common one -- no SDK installed -- is the same sentence
    for all three.
    """
    discovery = cameras_mod.discover(rig.cameras, overrides=overrides)
    for name, serial in discovery.missing.items():
        pinned = (overrides or {}).get(name)
        why = (
            f"pinned device {pinned} does not exist"
            if pinned
            else f"serial {serial} not on the bus"
        )
        print(f"  camera   {name:<12} not previewing — {why}")

    readers: dict[str, cameras_mod.CameraReader] = {}
    problems: dict[str, list[str]] = {}
    for name, found in discovery.matched.items():
        try:
            readers[name] = cameras_mod.CameraReader(found.spec())
        except (ConfigurationError, OSError) as err:
            problems.setdefault(str(err), []).append(name)
    for message, names in problems.items():
        print(f"  camera   {', '.join(names)} not previewing — {message}")
    return readers


def mirror(
    scene: object | None,
    live_arms: list[LiveArm],
    *,
    stop: threading.Event,
    rate_hz: float = _MIRROR_RATE_HZ,
    control: object | None = None,
    cameras: object | None = None,
) -> None:
    """Pump each arm's newest pose into the scene until ``stop`` is set.

    Reads ``latest_state`` rather than ``read_state`` on purpose: it does not
    block and does not raise on a briefly silent arm. A stale pose left on
    screen is the right failure mode here -- raising would tear down a session
    that is holding two energized arms.

    A ``control`` panel is stepped on this same clock rather than on a thread
    of its own, so the pose that is drawn and the pose that is commanded are
    always one tick apart at most, and the two can never interleave. A
    ``cameras`` panel rides the same clock for the same reason -- it throttles
    itself down to a preview rate internally, so the images and the poses share
    one websocket instead of racing for it.
    """
    period = 1.0 / rate_hz
    while not stop.is_set():
        if scene is not None:
            for entry in live_arms:
                state = entry.arm.latest_state
                if state is not None:
                    scene.update(entry.name, state.joints.position_rad)  # type: ignore[attr-defined]
        if control is not None:
            control.step(period)  # type: ignore[attr-defined]
        if cameras is not None:
            cameras.step(period)  # type: ignore[attr-defined]
        stop.wait(period)


def run_live(
    rig: Rig,
    *,
    float_mode: bool = False,
    park: bool = True,
    visualize: bool = True,
    control: bool = False,
    camera_preview: bool = True,
    camera_overrides: dict[str, str] | None = None,
    port: int = 8080,
    mesh_dir: Path | None = None,
    rate_hz: float = _MIRROR_RATE_HZ,
    backend_factory: Callable[[RigArm], ArmBackend] | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Power the rig up, mirror it, then park it and power it down.

    The whole lifecycle lives inside this call because it has to: the native
    node holds a liveness pipe to this process, so returning from here is what
    de-energizes the arms. There is no way to leave them up for a later command.
    """
    stop = stop if stop is not None else threading.Event()
    if control and not visualize:
        raise ConfigurationError("browser control needs the browser view; drop --no-viz")
    scene = None
    if visualize:
        # Imported here, not at module scope: doctor and zero must keep working
        # without the optional viz extra installed.
        from .viz import ArmSceneVisualizer

        scene = ArmSceneVisualizer.from_rig(rig, mesh_dir=mesh_dir, port=port)
        # Deliberately no add_gui(): viz's own sliders drive the render, and the
        # render here belongs to the hardware. Browser control is a different
        # surface -- see viser_control, where the sliders are targets instead.

    # Cameras before the motors, as in ``record``: opening them is what finds
    # out a camera is held by another process, and finding that out with two
    # arms already energized is worse. Unlike ``record`` it is not fatal here.
    camera_readers: dict[str, cameras_mod.CameraReader] = {}
    if scene is not None and camera_preview and rig.cameras:
        camera_readers = open_preview_cameras(rig, overrides=camera_overrides)

    mode = "gravity float (backdrivable)" if float_mode else "holding"
    try:
        session, live_arms = power_up(rig, float_mode=float_mode, backend_factory=backend_factory)
    except BaseException:
        # The capture threads are already running; nothing below will reach the
        # finally that would have stopped them.
        cameras_mod.close_readers(camera_readers)
        raise
    panel = None
    camera_panel = None
    failures = 0
    try:
        for entry in live_arms:
            print(
                f"  {entry.name:<8} {entry.rig_arm.model} on {entry.rig_arm.interface}"
                f" — energized, {mode}"
            )
        if control and scene is not None:
            from .viser_control import RigControlPanel

            followers = {
                entry.name: entry.arm for entry in live_arms if entry.rig_arm.is_follower
            }
            panel = RigControlPanel(scene, followers, float_mode=float_mode)  # type: ignore[arg-type]
            print(f"  control  {', '.join(followers)} — disarmed; arm each one in the browser")
        if camera_readers and scene is not None:
            from .viz import CameraPanel

            camera_panel = CameraPanel(scene.server, camera_readers)
            print(f"  cameras  {', '.join(camera_readers)} — live in the browser")
        if scene is not None:
            print(f"  viser    {scene.url}")
        print(f"ctrl-c to {'park at home_pos and ' if park else ''}power down")
        mirror(
            scene,
            live_arms,
            stop=stop,
            rate_hz=rate_hz,
            control=panel,
            cameras=camera_panel,
        )
    except KeyboardInterrupt:
        print()
    finally:
        # Disarm before anything else: the park below moves the arms, and it
        # must not race a panel that is still pushing targets at them.
        if panel is not None:
            panel.disarm_all("session ending")
        # Released before the park below, not after: parking takes seconds, and
        # a camera left held for them is a camera the next command cannot open.
        cameras_mod.close_readers(camera_readers)
        if scene is not None:
            scene.server.stop()
        failures = power_down(session, live_arms, park=park)
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #


def run_record(
    rig: Rig,
    *,
    task: str,
    repo_id: str | None,
    teleop: str = "vr",
    fps: int = 30,
    num_episodes: int = 0,
    hold_duration_s: float = 10.0,
    root: Path | None = None,
    dry_run: bool = False,
    park: bool = True,
    push_to_hub: bool = False,
    private: bool = False,
    camera_overrides: dict[str, str] | None = None,
    vr_url: str | None = None,
    vr_kit: Path | None = None,
    yam_xml: str | None = None,
    backend_factory: Callable[[RigArm], ArmBackend] | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Record teleoperated episodes from a rig into a LeRobot dataset.

    The whole session lives inside this call for the same reason ``live`` does:
    the native node dies with its parent, so returning from here is what
    de-energizes the arms. The order of the teardown matters and is deliberate
    -- the dataset is finalized on disk *before* the arms are parked, so an
    interrupted shutdown still leaves a complete, loadable dataset; and an
    upload happens *after* the arms are down, because a few hundred megabytes of
    video takes minutes and there is no reason to hold motors energized for it.
    """
    from . import record as record_mod

    stop = stop if stop is not None else threading.Event()
    cameras: dict[str, object] = {}

    # Cameras first: they need no motors, and finding out now that a camera is
    # held by another process beats finding out with two arms energized.
    if rig.cameras:
        discovery = cameras_mod.discover(rig.cameras, overrides=camera_overrides)
        if not discovery.complete:
            missing = ", ".join(
                f"{name} (serial {serial})" for name, serial in discovery.missing.items()
            )
            print(f"error: camera(s) not on the bus: {missing}", file=sys.stderr)
            return 1
        cameras = dict(cameras_mod.open_readers(discovery.specs()))
        print(f"  cameras  {', '.join(cameras)}")

    session = None
    live_arms: list[LiveArm] = []
    source = None
    status = 0
    try:
        shapes = record_mod.camera_shapes(cameras)

        session, live_arms = power_up(rig, float_mode=False, backend_factory=backend_factory)
        # Followers only. A leader has no `command`, so handing one to the record
        # loop would fail mid-session on a rig that has one -- and its pose is
        # not what the dataset is about anyway.
        arms = {entry.name: entry.arm for entry in live_arms if entry.rig_arm.is_follower}
        if not arms:
            raise ConfigurationError(
                f"rig {rig.name!r} has no follower arms to record from"
            )
        for entry in live_arms:
            print(
                f"  {entry.name:<8} {entry.rig_arm.model} on {entry.rig_arm.interface}"
                " — energized, holding"
            )

        dofs = {name: arm.capabilities.dof for name, arm in arms.items()}
        state_names = record_mod.arm_feature_names(list(arms), dofs)
        features = record_mod.build_features(state_names, shapes)

        source = _build_teleop_source(
            teleop,
            dofs=dofs,
            hold_duration_s=hold_duration_s,
            vr_url=vr_url,
            vr_kit=vr_kit,
            yam_xml=yam_xml,
        )
        print(f"  teleop   {source.describe()}")

        if dry_run:
            sink: object = record_mod.MemorySink()
            print("  dataset  --dry-run: nothing is written to disk")
        else:
            assert repo_id is not None  # guaranteed by the caller
            sink = record_mod.LeRobotSink(
                repo_id=repo_id,
                fps=fps,
                features=features,
                robot_type=record_mod.rig_robot_type(rig),
                root=root,
                image_writer_threads=4 * len(cameras),
            )
            print(f"  dataset  {repo_id} at {sink.root}")
        print(f"  schema   {len(state_names)}-dim state/action, {len(shapes)} camera(s)")
        print(f"  task     {task!r}")
        print()
        if teleop == "vr":
            print("  right B: start (or redo) an episode    left Y: save it")
        print(f"  ctrl-c to {'park and ' if park else ''}power down")
        print()

        result = record_mod.record_session(
            arms=arms,
            source=source,
            sink=sink,  # type: ignore[arg-type]
            cameras=cameras,
            task=task,
            fps=fps,
            num_episodes=num_episodes,
            stop=stop,
        )
        print(f"\n{result.summary()}")
        if result.episodes == 0:
            status = 1
            print("no episode was saved", file=sys.stderr)
    finally:
        if source is not None:
            source.close()
        cameras_mod.close_readers(cameras)  # type: ignore[arg-type]
        if session is not None:
            failures = power_down(session, live_arms, park=park)
            status = status or (1 if failures else 0)

    if push_to_hub and not dry_run and status == 0:
        print(f"pushing {repo_id} to the Hub ({'private' if private else 'public'}) ...")
        try:
            sink.push_to_hub(private=private)  # type: ignore[union-attr]
        except Exception as err:  # noqa: BLE001 - the dataset is already safe on disk
            print(f"push failed: {type(err).__name__}: {err}", file=sys.stderr)
            print("the local dataset is complete; retry the upload by hand", file=sys.stderr)
            status = 1
        else:
            print(f"pushed: https://huggingface.co/datasets/{repo_id}")
    return status


def _build_teleop_source(
    teleop: str,
    *,
    dofs: dict[str, int],
    hold_duration_s: float,
    vr_url: str | None,
    vr_kit: Path | None,
    yam_xml: str | None,
):  # noqa: ANN202 - a TeleopSource
    from . import record as record_mod

    if teleop == "hold":
        return record_mod.HoldSource(dofs, duration_s=hold_duration_s)
    if teleop == "vr":
        from .teleop_vr import DEFAULT_WS_URL, QuestTeleopSource

        return QuestTeleopSource(
            list(dofs),
            ws_url=vr_url or DEFAULT_WS_URL,
            kit_path=vr_kit,
            model_path=yam_xml,
        )
    raise ConfigurationError(f"unknown teleop source {teleop!r}; use 'vr' or 'hold'")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _add_common(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--model", required=required, help=f"one of: {', '.join(SUPPORTED_MODELS)}"
    )
    parser.add_argument(
        "--interface",
        required=required,
        help="CAN interface name, /dev path, or controller IPv4 address",
    )
    parser.add_argument(
        "--effector", default=None, help=f"one of: {', '.join(SUPPORTED_EFFECTORS)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openpi-control", description="Preflight and maintenance for a connected arm."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="read-only preflight checks for an arm or a rig")
    _add_common(doctor, required=False)
    doctor.add_argument(
        "--rig",
        default=None,
        help=f"check every arm of a packaged rig instead of one arm: {', '.join(rig_names())}",
    )
    doctor.add_argument(
        "--interface-override",
        action="append",
        metavar="ARM=IFACE",
        help="with --rig: check one arm on a different bus, e.g. left=can2 (repeatable)",
    )
    doctor.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="with --rig: pin one camera to an explicit device (repeatable)",
    )
    doctor.add_argument(
        "--probe",
        action="store_true",
        help="also open the bus and listen for traffic (reads only, never sends)",
    )

    zero = sub.add_parser("zero", help="write the current pose as each servo's firmware zero")
    _add_common(zero)
    zero.add_argument(
        "--joint",
        type=int,
        default=None,
        help="zero only this joint id instead of every joint",
    )
    zero.add_argument(
        "--dry-run", action="store_true", help="print the plan and exit without opening the bus"
    )
    zero.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    live = sub.add_parser(
        "live", help="energize a rig, mirror it in the browser, then park it and power down"
    )
    live.add_argument(
        "--rig", default="yam_bimanual", help=f"one of: {', '.join(rig_names())}"
    )
    live.add_argument(
        "--only",
        action="append",
        metavar="ARM",
        help="bring up only this arm of the rig, e.g. --only left (repeatable)",
    )
    live.add_argument(
        "--interface",
        action="append",
        metavar="ARM=IFACE",
        help="move one arm to a different bus, e.g. --interface left=can2 (repeatable)",
    )
    live.add_argument(
        "--float",
        dest="float_mode",
        action="store_true",
        help="gravity float instead of holding: the arms become backdrivable",
    )
    live.add_argument(
        "--no-park",
        dest="park",
        action="store_false",
        help="de-energize where the arm stands instead of parking it at home_pos first",
    )
    live.add_argument(
        "--no-viz",
        dest="visualize",
        action="store_false",
        help="energize and hold without serving the browser view",
    )
    live.add_argument(
        "--control",
        action="store_true",
        help="add per-arm browser control: confirm the pose, Arm, then drive with sliders",
    )
    live.add_argument(
        "--no-cameras",
        dest="camera_preview",
        action="store_false",
        help="skip the camera tiles, leaving the rig's cameras free for another process",
    )
    live.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="pin one camera to an explicit device, e.g. top=/dev/video4 (repeatable)",
    )
    live.add_argument("--port", type=int, default=8080, help="viser HTTP port")
    live.add_argument(
        "--mesh-dir", type=Path, default=None, help="directory holding the URDF's meshes"
    )
    live.add_argument(
        "--skip-preflight",
        action="store_true",
        help="energize without running the doctor checks first",
    )
    live.add_argument("--list", action="store_true", help="describe the rig and exit")

    cams = sub.add_parser(
        "cameras", help="resolve a rig's cameras to device paths and say which are present"
    )
    cams.add_argument("--rig", default="yam_bimanual", help=f"one of: {', '.join(rig_names())}")
    cams.add_argument(
        "--only",
        action="append",
        metavar="ARM",
        help="narrow to one arm of the rig, which drops the other arm's wrist camera",
    )
    cams.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="pin one camera to an explicit device, e.g. top=/dev/video4 (repeatable)",
    )
    cams.add_argument(
        "--probe",
        action="store_true",
        help="also open each stream and grab a frame (needs the cameras extra)",
    )
    cams.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        metavar="DIR",
        help="with --probe: write each grabbed frame to DIR/<name>.png",
    )

    rec = sub.add_parser(
        "record", help="teleoperate a rig and record the episodes as a LeRobot dataset"
    )
    rec.add_argument("--rig", default="yam_bimanual", help=f"one of: {', '.join(rig_names())}")
    rec.add_argument(
        "--only",
        action="append",
        metavar="ARM",
        help="record one arm of the rig, which also drops the other wrist camera",
    )
    rec.add_argument(
        "--interface",
        action="append",
        metavar="ARM=IFACE",
        help="move one arm to a different bus, e.g. --interface left=can2 (repeatable)",
    )
    rec.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="pin one camera to an explicit device (repeatable)",
    )
    rec.add_argument(
        "--repo-id",
        default=None,
        help="dataset id, e.g. you/yam-fold-towel. Required unless --dry-run",
    )
    rec.add_argument(
        "--task",
        default=None,
        help="the language instruction stored on every frame; relabelling later "
        "means rewriting the dataset, so set it",
    )
    rec.add_argument(
        "--fps",
        type=int,
        default=30,
        help="dataset, control-loop, and camera rate. The D405s here sustain 90 "
        "at 848x480 with all three running; the arms publish at 200",
    )
    rec.add_argument(
        "--camera-fps",
        type=int,
        default=None,
        help="run the cameras at a different rate from the loop (default: --fps)",
    )
    rec.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="end the session after this many saved episodes (0 = until ctrl-c)",
    )
    rec.add_argument("--root", type=Path, default=None, help="local dataset root")
    rec.add_argument(
        "--teleop",
        choices=("vr", "hold"),
        default="vr",
        help="vr: drive from a Quest via vr-teleop-kit. hold: arms stay still for "
        "--hold-seconds, to check the pipeline without a headset",
    )
    rec.add_argument(
        "--hold-seconds",
        type=float,
        default=10.0,
        help="episode length for --teleop hold",
    )
    rec.add_argument("--vr-url", default=None, help="relay WebSocket URL")
    rec.add_argument(
        "--vr-kit", type=Path, default=None, help="path to a vr-teleop-kit checkout"
    )
    rec.add_argument(
        "--yam-xml", default=None, help="YAM MJCF the VR inverse kinematics loads"
    )
    rec.add_argument(
        "--no-cameras",
        dest="cameras_enabled",
        action="store_false",
        help="record state and action only",
    )
    rec.add_argument(
        "--dry-run",
        action="store_true",
        help="run the whole session, including the arms, but write nothing to disk",
    )
    rec.add_argument(
        "--no-park",
        dest="park",
        action="store_false",
        help="de-energize where the arms stand instead of parking them first",
    )
    rec.add_argument(
        "--skip-preflight", action="store_true", help="record without the doctor checks"
    )
    rec.add_argument(
        "--push-to-hub", action="store_true", help="upload once the arms are down"
    )
    rec.add_argument("--private", action="store_true", help="with --push-to-hub, keep it private")

    args = parser.parse_args(argv)
    log_path = runlog.setup_run_logging(args.command)

    commands = {
        "doctor": _command_doctor,
        "zero": _command_zero,
        "live": _command_live,
        "cameras": _command_cameras,
        "record": _command_record,
    }
    try:
        return commands[args.command](args, log_path)
    except ConfigurationError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


def _command_doctor(args: argparse.Namespace, log_path: Path) -> int:
    if args.rig is not None:
        if args.model is not None or args.interface is not None:
            raise ConfigurationError(
                "--rig already names every arm and its bus; drop --model/--interface "
                "(use --interface-override ARM=IFACE to move one)"
            )
        return _doctor_rig(args, log_path)
    if args.model is None or args.interface is None:
        raise ConfigurationError("doctor needs either --rig, or both --model and --interface")

    print(f"doctor: {args.model} on {args.interface}")
    results = run_doctor(
        args.model, args.interface, effector_model=args.effector, probe=args.probe
    )
    for result in results:
        print(result.render())
    failures = sum(1 for result in results if result.status == _FAIL)
    warnings = sum(1 for result in results if result.status == _WARN)
    print(f"\n{len(results)} checks, {failures} failed, {warnings} warned")
    print(f"log: {log_path}")
    return 1 if failures else 0


def _doctor_rig(args: argparse.Namespace, log_path: Path) -> int:
    """Every arm of a rig, checked with the same code ``live`` preflights with."""
    rig = resolve_rig(args.rig).with_interfaces(
        _parse_interface_overrides(args.interface_override)
    )
    print(f"doctor: rig {rig.name} — {rig.description}")
    failures = warnings = 0
    for rig_arm in rig.arms:
        results = run_doctor(
            rig_arm.model,
            rig_arm.interface,
            effector_model=rig_arm.effector_model,
            probe=args.probe,
        )
        failures += sum(1 for result in results if result.status == _FAIL)
        warnings += sum(1 for result in results if result.status == _WARN)
        print(f"\n{rig_arm.name} ({rig_arm.model} on {rig_arm.interface}):")
        for result in results:
            print(result.render())

    # Cameras belong to the rig, not to an arm, so they are checked once here
    # rather than repeated under every arm.
    camera_results = run_camera_checks(
        rig, overrides=cameras_mod.parse_camera_overrides(args.camera)
    )
    failures += sum(1 for result in camera_results if result.status == _FAIL)
    warnings += sum(1 for result in camera_results if result.status == _WARN)
    print(f"\ncameras ({len(rig.cameras)} declared):")
    for result in camera_results:
        print(result.render())

    print(
        f"\n{len(rig.arms)} arms, {len(rig.cameras)} cameras, "
        f"{failures} failed, {warnings} warned"
    )
    print(f"log: {log_path}")
    return 1 if failures else 0


def _command_zero(args: argparse.Namespace, log_path: Path) -> int:
    full_plan = build_plan(args.model, args.effector)
    plan = full_plan
    if args.joint is not None:
        plan = tuple(entry for entry in full_plan if entry.joint_id == args.joint)
        if not plan:
            known = ", ".join(str(entry.joint_id) for entry in full_plan)
            raise ConfigurationError(
                f"{args.model} has no joint {args.joint}; joints are {known}"
            )

    port_type = plan_port_type(plan)
    if port_type is None:
        print("every servo in this plan is read-only; nothing to zero")
        return 0

    if args.dry_run:
        print(f"dry run: would zero via {port_type} on {args.interface}")
        for entry in plan:
            state = "read-only, skipped" if entry.read_only else "zero"
            print(f"  {entry.label():<24} {entry.servo_model:<24} id {entry.servo_id:<6} {state}")
        return 0

    problem = buses.check_interface(port_type, args.interface)
    if problem:
        raise ConfigurationError(problem)

    if not args.yes and not _confirm(plan, args.model, args.interface):
        print("aborted; nothing was written")
        return 1

    assets = resolve_model_assets(args.model, effector_model=args.effector)
    outcomes = zero_arm(
        plan,
        port_type,
        args.interface,
        baudrate=_catalog_baudrate(assets.model_config),
    )
    failed = 0
    for entry, error in outcomes:
        if error is None:
            print(f"  {entry.label():<24} zeroed")
        else:
            failed += 1
            print(f"  {entry.label():<24} FAILED: {error}")
    for entry in plan:
        if entry.read_only:
            print(f"  {entry.label():<24} skipped (read-only)")
    print(f"\n{len(outcomes) - failed} zeroed, {failed} failed")
    print(f"log: {log_path}")
    return 1 if failed else 0


def _parse_interface_overrides(values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values or []:
        arm, _, interface = value.partition("=")
        if not arm or not interface:
            raise ConfigurationError(
                f"--interface takes ARM=IFACE (e.g. left=can2), got {value!r}"
            )
        overrides[arm] = interface
    return overrides


def _command_live(args: argparse.Namespace, log_path: Path) -> int:
    rig = resolve_rig(args.rig).with_interfaces(_parse_interface_overrides(args.interface))
    if args.only:
        rig = rig.subset(args.only)

    print(f"live: rig {rig.name} — {rig.description}")
    for rig_arm in rig.arms:
        effector = rig_arm.effector_model or "no effector"
        print(
            f"  {rig_arm.name:<8} {rig_arm.model:<6} {rig_arm.interface:<8} "
            f"{effector:<8} {rig_arm.role}"
        )
    if args.list:
        return 0

    if not args.skip_preflight:
        failures, reports = preflight_rig(rig)
        for name, results in reports:
            problems = [r for r in results if r.status != _OK]
            summary = "all checks pass" if not problems else f"{len(problems)} to look at"
            print(f"\npreflight {name}: {summary}")
            for result in problems:
                print(result.render())
        if failures:
            print(
                f"\n{failures} failed check(s); nothing was energized. "
                "Fix them, or re-run with --skip-preflight.",
                file=sys.stderr,
            )
            return 1

    print()
    status = run_live(
        rig,
        float_mode=args.float_mode,
        park=args.park,
        visualize=args.visualize,
        control=args.control,
        camera_preview=args.camera_preview,
        camera_overrides=cameras_mod.parse_camera_overrides(args.camera),
        port=args.port,
        mesh_dir=args.mesh_dir,
    )
    print(f"log: {log_path}")
    return status


def _command_cameras(args: argparse.Namespace, log_path: Path) -> int:
    if args.snapshot is not None and not args.probe:
        raise ConfigurationError(
            "--snapshot needs --probe: a frame has to be grabbed before it can be written"
        )
    rig = resolve_rig(args.rig)
    if args.only:
        rig = rig.subset(args.only)
    overrides = cameras_mod.parse_camera_overrides(args.camera)

    print(f"cameras: rig {rig.name}")
    results = run_camera_checks(rig, overrides=overrides)
    for result in results:
        print(result.render())

    if args.probe:
        print()
        probed = probe_cameras(rig, overrides=overrides, snapshot_dir=args.snapshot)
        for result in probed:
            print(result.render())
        results += probed

    failures = sum(1 for result in results if result.status == _FAIL)
    warnings = sum(1 for result in results if result.status == _WARN)
    print(f"\n{len(results)} checks, {failures} failed, {warnings} warned")
    print(f"log: {log_path}")
    return 1 if failures else 0


def _command_record(args: argparse.Namespace, log_path: Path) -> int:
    if not args.dry_run and not args.repo_id:
        raise ConfigurationError(
            "record needs --repo-id (e.g. you/yam-fold-towel), or --dry-run to "
            "rehearse the session without writing a dataset"
        )
    if args.fps <= 0:
        raise ConfigurationError(f"--fps must be positive, got {args.fps}")

    overrides = cameras_mod.parse_camera_overrides(args.camera)
    rig = resolve_rig(args.rig).with_interfaces(_parse_interface_overrides(args.interface))
    if args.only:
        rig = rig.subset(args.only)
    if not args.cameras_enabled:
        rig = rig.without_cameras()
    # The run's capture settings, applied once here so the preflight validates
    # exactly the mode the recorder will open. Cameras run at the record rate:
    # a loop at 90 Hz reading 30 fps cameras would write each frame three times
    # and call it data. RGB straight from the SDK, because converting in numpy
    # costs 4.1 ms of an 11.1 ms tick with three cameras (see record.to_rgb).
    rig = rig.with_camera_capture(
        fps=args.camera_fps or args.fps, pixel_format="rgb8"
    )

    # A dataset carries its task string on every frame, and relabelling means
    # rewriting the dataset, so an unset --task is worth a word rather than a
    # silent default that ends up in a thousand episodes.
    task = args.task or "teleop"
    if not args.task:
        print("warning: no --task given; every frame will say 'teleop'", file=sys.stderr)

    print(f"record: rig {rig.name} — {rig.description}")
    for rig_arm in rig.arms:
        print(f"  {rig_arm.name:<8} {rig_arm.model:<6} {rig_arm.interface}")

    if not args.skip_preflight:
        failures, reports = preflight_rig(rig)
        camera_results = run_camera_checks(rig, overrides=overrides, required=True)
        # Asking for a rate the camera does not have should fail here, not after
        # two arms are energized and the first pipeline refuses to start.
        camera_results += check_camera_modes(rig)
        failures += sum(1 for result in camera_results if result.status == _FAIL)
        for name, results in reports:
            problems = [r for r in results if r.status != _OK]
            if problems:
                print(f"\npreflight {name}:")
                for result in problems:
                    print(result.render())
        camera_problems = [r for r in camera_results if r.status != _OK]
        if camera_problems:
            print("\npreflight cameras:")
            for result in camera_problems:
                print(result.render())
        if failures:
            print(
                f"\n{failures} failed check(s); nothing was energized. "
                "Fix them, or re-run with --skip-preflight.",
                file=sys.stderr,
            )
            return 1

    print()
    status = run_record(
        rig,
        task=task,
        repo_id=args.repo_id,
        teleop=args.teleop,
        fps=args.fps,
        num_episodes=args.num_episodes,
        hold_duration_s=args.hold_seconds,
        root=args.root,
        dry_run=args.dry_run,
        park=args.park,
        push_to_hub=args.push_to_hub,
        private=args.private,
        camera_overrides=overrides,
        vr_url=args.vr_url,
        vr_kit=args.vr_kit,
        yam_xml=args.yam_xml,
    )
    print(f"log: {log_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
