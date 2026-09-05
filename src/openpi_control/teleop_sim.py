"""VR-driven kinematic twin. Never constructs a hardware session."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np

from .exceptions import ConfigurationError
from .rigs import resolve_rig
from .teleop_vr import _import_vr_kit


def decode_action(action, names):
    """Validate IK targets before updating the browser; grippers use native polarity."""
    joints, grippers = {}, {}
    for name in names:
        try:
            q = np.array([action[f"{name}_joint_{i}.pos"] for i in range(1, 7)], dtype=float)
            grip = float(action[f"{name}_gripper.pos"])
        except (KeyError, TypeError, ValueError) as err:
            raise ConfigurationError(f"invalid VR action for {name}: {err}") from err
        if not np.all(np.isfinite(q)) or not math.isfinite(grip) or not 0 <= grip <= 1:
            raise ConfigurationError(f"invalid VR joint/gripper values for {name}")
        joints[name], grippers[name] = q, 1.0 - grip
    return joints, grippers


def run_sim(args, stop: threading.Event) -> int:
    from .viz import ArmSceneVisualizer, GripperPanel

    if not math.isfinite(args.rate) or args.rate <= 0:
        raise ConfigurationError("--rate must be positive and finite")
    kit = args.vr_kit
    if kit is None:
        candidates = [
            Path(__file__).resolve().parents[2] / "external/vr-teleop-kit",
            Path.home() / "vr-teleop-kit",
        ]
        kit = next((path for path in candidates if (path / "src").is_dir()), None)
    module = _import_vr_kit(kit)
    model = args.yam_xml
    if not model and kit:
        candidate = kit / "i2rt/i2rt/robot_models/arm/yam/yam.xml"
        if candidate.is_file():
            model = str(candidate)
    rig = resolve_rig("yam_bimanual")
    teleop = module.BiQuestTeleoperator(
        module.BiQuestTeleoperatorConfig(
            id="openpi-viser-sim",
            ws_url=args.vr_url,
            model_path=model or "",
        )
    )
    scene = None
    try:
        scene = ArmSceneVisualizer.from_rig(rig, port=args.port)
        scene.server.gui.add_markdown(
            "## VR kinematic twin\n**SIMULATION — no hardware connected**\n\n"
            "Hold a controller **grip** to move its arm. Release to clutch. "
            "**Trigger** closes the gripper.\n\n"
            "IK preview only: no contact physics. Gripper values are shown below; "
            "the URDF fingers are fixed."
        )
        status = scene.server.gui.add_text("VR", initial_value="Connecting to relay", disabled=True)
        quit_button = scene.server.gui.add_button("Stop simulation", color="red")

        @quit_button.on_click
        def _quit(_event):
            stop.set()

        gripper = GripperPanel(scene.server, rig.names)
        print(f"SIMULATION ONLY · Viser: {scene.url}")
        print(f"Quest relay: {args.vr_url}")
        teleop.connect()
        print("Relay connected. Open the Quest page, start teleop, then hold a grip.")
        render_at = 0.0
        period = 1.0 / args.rate
        while not stop.is_set():
            started = time.monotonic()
            action = teleop.get_action()
            joints, grips = decode_action(action, rig.names)
            if started >= render_at:
                for name, q in joints.items():
                    scene.update(name, q)
                gripper.update(grips, grips)
                status.value = (
                    "Grip engaged" if teleop.is_engaged() else "Waiting for grip / XR frames"
                )
                render_at = started + 1.0 / 30
            stop.wait(max(0.0, period - (time.monotonic() - started)))
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            teleop.disconnect()
        finally:
            if scene is not None:
                scene.server.stop()
