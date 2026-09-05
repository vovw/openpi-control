"""Read-only Quest relay diagnostics. Does not construct a teleoperator or arm."""

from __future__ import annotations

import json
import math
import time

from .exceptions import ConfigurationError


def check_vr(url: str, timeout: float = 5) -> int:
    from websockets.sync.client import connect

    if not math.isfinite(timeout) or timeout <= 0:
        raise ConfigurationError("--timeout must be positive and finite")
    print(f"VR health · {url} · observing for {timeout:g}s")
    print("Put on the headset and squeeze each controller grip during this check.")
    frames = 0
    hands = set()
    gripped = set()
    last_frame = None
    malformed = 0
    try:
        with connect(url, open_timeout=3, close_timeout=1) as ws:
            print("Relay: connected")
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    raw = ws.recv(timeout=max(0.001, deadline - time.monotonic()))
                except TimeoutError:
                    break
                try:
                    message = json.loads(raw)
                    if not isinstance(message, dict) or message.get("type") != "xr_frame":
                        continue
                    controllers = message.get("controllers") or {}
                    if not isinstance(controllers, dict):
                        raise ValueError("controllers must be an object")
                    valid = set()
                    for hand in ("left", "right"):
                        controller = controllers.get(hand)
                        if not controller:
                            continue
                        position, orientation = controller["position"], controller["orientation"]
                        if len(position) != 3 or len(orientation) != 4:
                            raise ValueError("invalid pose shape")
                        if not all(math.isfinite(float(x)) for x in (*position, *orientation)):
                            raise ValueError("invalid pose values")
                        valid.add(hand)
                        buttons = controller.get("buttons") or []
                        if len(buttons) > 1 and buttons[1].get("p"):
                            gripped.add(hand)
                    hands.update(valid)
                    if valid:
                        frames += 1
                        last_frame = time.monotonic()
                except (ValueError, TypeError, KeyError, AttributeError):
                    malformed += 1
    except (OSError, TimeoutError) as err:
        print(f"error: relay unavailable: {err}")
        print("Start vr-teleop-relay in the VR kit environment, then rerun health vr.")
        return 1
    except Exception as err:
        print(f"error: VR stream failed: {type(err).__name__}: {err}")
        return 1

    print(f"Controller frames: {frames} (~{frames / timeout:.1f} Hz)")
    for hand in ("left", "right"):
        print(
            f"{hand}: {'pose received' if hand in hands else 'MISSING pose'}; "
            f"grip {'detected' if hand in gripped else 'not observed'}"
        )
    if malformed:
        print(f"warning: {malformed} malformed frame(s)")
    if not frames:
        print(
            "error: Relay works, but no valid headset controller poses arrived.\n"
            "Run: adb reverse tcp:8443 tcp:8443\n"
            "In Quest open http://localhost:8443/, select Start Teleop, and enter VR."
        )
        return 1
    if hands != {"left", "right"} or last_frame is None or time.monotonic() - last_frame > 0.5:
        print("warning: missing controller or stale stream; check headset tracking and connection")
        return 1
    print(
        "VR poses: all checks pass"
        if len(gripped) == 2
        else "VR poses: receiving. Squeeze both side grips to confirm clutch input."
    )
    return 0
