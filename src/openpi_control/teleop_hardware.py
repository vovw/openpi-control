"""Hardware VR entry point with optional recording and measured-pose Viser output."""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime
from uuid import uuid4

import numpy as np

from .exceptions import ConfigurationError
from .rigs import resolve_rig
from .teleop_vr import QuestTeleopSource, resolve_vr_paths
from .types import PositionCommand


def new_recording_id() -> str:
    return f"local/vr-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"


def run_hardware(args, stop: threading.Event) -> int:
    from . import cli

    if not math.isfinite(args.rate) or args.rate <= 0:
        raise ConfigurationError("--rate must be positive and finite")
    rig = resolve_rig(args.rig).with_interfaces(cli._parse_interface_overrides(args.interface))
    if args.only:
        rig = rig.subset(args.only)
    if not set(rig.names) <= {"left", "right"}:
        raise ConfigurationError("VR currently supports left/right YAM arms")
    kit, model = resolve_vr_paths(args.vr_kit, args.yam_xml)
    if args.record:
        # Reuse the established recorder, including camera preflight and parking.
        record_args = vars(args).copy()
        record_args.update(
            _teleop_stop=stop,
            dry_run=False,
            repo_id=args.repo_id or new_recording_id(),
            teleop="vr",
            fps=30,
            camera_fps=None,
            camera=None,
            cameras_enabled=not args.no_cameras,
            skip_preflight=False,
            num_episodes=0,
            hold_seconds=10,
            root=None,
            park=True,
            push_to_hub=False,
            private=False,
            vr_kit=kit,
            yam_xml=model,
        )
        from argparse import Namespace

        from .log_paths import runtime_log_dir

        return cli._command_record(Namespace(**record_args), runtime_log_dir() / "teleop.log")

    failures, reports = cli.preflight_rig(rig)
    for name, results in reports:
        print(
            f"preflight {name}: "
            + (
                "all checks pass"
                if all(r.status == cli._OK for r in results)
                else ", ".join(r.render() for r in results)
            )
        )
    if failures:
        raise ConfigurationError("Hardware preflight failed; no arms were energized")

    # Establish input before any motor power. A WebSocket alone is not a headset.
    source = QuestTeleopSource(rig.names, kit_path=kit, model_path=model, ws_url=args.vr_url)
    scene = None
    session = None
    arms = []
    try:
        print("Relay connected. Waiting for fresh Quest controller poses; motors remain off.")
        last_notice = 0.0
        while not stop.is_set():
            teleop = source._teleop
            with teleop._lock:
                frame = teleop._latest_xr_frame or {}
                age = time.time() - teleop._last_xr_frame_time
                controllers = frame.get("controllers") or {}
                ready = age < 0.2 and all(name in controllers for name in rig.names)
            if ready:
                break
            if time.monotonic() - last_notice >= 5:
                print(
                    "Waiting for VR: open http://localhost:8443 in Quest and select Start Teleop."
                )
                last_notice = time.monotonic()
            stop.wait(0.1)
        if stop.is_set():
            return 0
        if not args.no_viz:
            from .viz import ArmSceneVisualizer

            scene = ArmSceneVisualizer.from_rig(rig, port=args.port)
            button = scene.server.gui.add_button("Stop & park", color="red")

            @button.on_click
            def _stop(_event):
                stop.set()

            scene.server.gui.add_markdown("**REAL HARDWARE** — displaying measured joint positions")
            print(f"Viser: {scene.url}")
        session, arms = cli.power_up(rig)
        print("Hardware energized. Hold grip to move. q / Ctrl-C stops and parks.")
        period = 1.0 / args.rate
        render_at = 0.0
        commanded: dict[str, np.ndarray] = {}
        while not stop.is_set():
            started = time.monotonic()
            states = cli._inference_states(arms, max_age_s=0.25)
            step = source.poll(states)
            for entry in arms:
                target = step.targets.get(entry.name)
                if target is not None:
                    measured = states[entry.name].joints.position_rad
                    # Slew the command at 1 rad/s, seeded from feedback once.
                    # Re-anchoring every tick to measured position caps the
                    # servo's position error at 0.01 rad at 100 Hz: too little
                    # to overcome load/friction, so motion stalls at a nudge.
                    previous = commanded.get(entry.name, measured)
                    q = previous + np.clip(
                        np.asarray(target.position_rad) - previous, -period, period
                    )
                    entry.arm.command(PositionCommand(q, target.effector))
                    commanded[entry.name] = q.copy()
            if scene is not None and started >= render_at:
                for name, state in states.items():
                    scene.update(name, state.joints.position_rad)
                render_at = started + 1 / 30
            stop.wait(max(0, period - (time.monotonic() - started)))
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        source.close()
        try:
            if session is not None:
                failures = cli.power_down(session, arms, park=True)
                if failures:
                    raise ConfigurationError("Teleop stopped with shutdown errors; see logs")
        finally:
            if scene is not None:
                scene.server.stop()
