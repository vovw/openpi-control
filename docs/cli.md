# CLI commands

Run commands from the repository after `uv sync`. Append `--help` to any
command for all options.

## Preflight

```bash
# Check both arms and camera identities; no motor power.
uv run --no-sync openpi-control doctor --rig yam_bimanual

# Check a single arm; --probe opens the bus and listens without sending.
uv run --no-sync openpi-control doctor \
    --model Yam --interface can_right --effector E_Yam --probe

# Check a rig with a different CAN interface.
uv run --no-sync openpi-control doctor \
    --rig yam_bimanual --interface-override left=can0

# Print the rig configuration and exit.
uv run --no-sync openpi-control live --list
```

## Set servo zeros

Place the arm at its intended mechanical zero before writing calibration.

```bash
# Print the zeroing plan without opening the bus.
uv run --no-sync openpi-control zero \
    --model Yam --interface can_right --effector E_Yam --dry-run

# Write the current pose as firmware zero after confirmation.
uv run --no-sync openpi-control zero \
    --model Yam --interface can_right --effector E_Yam
```

`--joint ID` selects one joint. `--yes` skips the confirmation.

## Live control

These commands power real arms. Ctrl-C parks at `home_pos` and powers down.

```bash
# Power both arms, hold position, and show measured poses and cameras.
uv run --no-sync openpi-control live --rig yam_bimanual

# Add browser sliders. Confirm the measured pose and click Arm to enable them.
uv run --no-sync openpi-control live --control

# Enable gravity-compensated, backdrivable motion.
uv run --no-sync openpi-control live --float

# Power only the left arm, on can0, without a browser server.
uv run --no-sync openpi-control live --only left --interface left=can0 --no-viz

# Keep cameras free for another process.
uv run --no-sync openpi-control live --no-cameras
```

| Option | Effect |
| --- | --- |
| `--port 8081` | Change the Viser port from 8080. |
| `--camera top=/dev/video4` | Override one camera device. |
| `--mesh-dir PATH` | Use local rendering meshes. |
| `--no-park` | Power down in place instead of moving home. |
| `--skip-preflight` | Skip configuration checks before powering arms. |

## Other commands

| Command | What it does |
| --- | --- |
| [`cameras`](cameras.md) | Find cameras, probe streams, save snapshots. |
| [`adb connect`, `health vr`](vr.md) | Connect and diagnose Quest input. |
| [`teleop`](vr.md) | Drive real arms; `--backend sim` selects virtual arms. |
| [`record`](recording.md) | Record teleoperation as a LeRobot dataset. |
| [`infer`](inference.md) | Execute policy actions. |
| [`rollout`](inference.md) | Record timed policy attempts and outcome labels. |

Runtime logs are under `~/openpi-data/logs/runtime/`; `OPENPI_LOG_DIR`
overrides the log root.
