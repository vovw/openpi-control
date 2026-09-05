# Policy inference and rollouts

Start a compatible MolmoAct server separately. These commands control both
real YAM arms and require the top, left wrist, and right wrist cameras.
Replace `POLICY_HOST:8202` with the server address.

## Execute a policy

```bash
# Check configuration and camera streams first.
uv run --no-sync openpi-control doctor --rig yam_bimanual
uv run --no-sync openpi-control cameras --probe

# Run policy actions until stopped.
uv run --no-sync openpi-control infer \
    --server http://POLICY_HOST:8202 --instruction "fold the towel"

# Play the same action path at half speed.
uv run --no-sync openpi-control infer \
    --server http://POLICY_HOST:8202 --instruction "fold the towel" --speed 0.5

# Run without Viser or the terminal dashboard.
uv run --no-sync openpi-control infer \
    --server http://POLICY_HOST:8202 --instruction "fold the towel" --no-viz --plain
```

`infer` retries startup connection failures before powering motors. The
client checks the server's reported bimanual dimensions and negotiates its
JPEG container. Ctrl-C or a runtime error stops execution, parks, and powers down.

| Option | Effect |
| --- | --- |
| `--speed FLOAT` | Playback speed; `infer` defaults to 1.0. |
| `--control-rate HZ` | Nominal action rate; default 30. |
| `--max-step-rad FLOAT` | Limit joint movement per command tick. |
| `--max-effector-step FLOAT` | Limit normalized gripper movement per tick; default 0.30. |
| `--carry-targets` | Carry joint targets across chunk boundaries. |
| `--reach-actions` | Walk to each action using measured-pose substeps. |
| `--no-prefetch` | Wait for inference between chunks instead of overlapping it. |
| `--prefetch-margin-s FLOAT` | Add lead time when scheduling inference. |
| `--reset-start-pose` | Move to the training start pose before capturing the first observation. |
| `--request-timeout SECONDS` | HTTP timeout; default 60. |
| `--num-steps N` | Request this many denoising steps. |
| `--no-cuda-graph` | Disable the CUDA graph request. |
| `--jpeg-quality N` | Set JPEG quality; default 95, 0 sends raw arrays. |
| `--raw-frames` | Send raw RGB arrays. |
| `--interface ARM=IFACE` | Override a CAN interface. |
| `--camera NAME=DEVICE` | Override a camera device. |
| `--no-park` | Power down in place instead of moving home. |

Viser shows measured arms, predicted paths, and live camera previews. The
previews update independently from the images sent to the policy.

## Record policy attempts

```bash
# Add LeRobot dataset support (Python 3.12 or 3.13).
uv sync --extra lerobot

# Record three attempts, up to two minutes each, at half playback speed.
uv run --no-sync openpi-control rollout \
    --server http://POLICY_HOST:8202 \
    --repo-id local/towel-rollouts --root ~/openpi-data/rollouts/towel \
    --episodes 3 --episode-seconds 120 --speed 0.5
```

Enter the instruction before each attempt and a success label after it.
The arms park between attempts. Ctrl-C during an attempt saves its partial
episode; Ctrl-C at the prompt exits. Labels and attempt metadata are stored
in `openpi_control_rollouts.json` beside the LeRobot dataset.

| Rollout option | Effect |
| --- | --- |
| `--episodes N` | Number of attempts; default 3. |
| `--episode-seconds N` | Duration limit; default 120 seconds. |
| `--fps N` | Recording and command rate; default 30. |
| `--speed FLOAT` | Playback speed; default 0.5. |
| `--chunk-size N` | Execute only the first N actions of each chunk, from 1 to 30. |
| `--no-reset-pause` | Skip the manual scene-reset pause. |
| `--no-viz` | Omit Viser. |

Run `uv run --no-sync openpi-control rollout --help` for the complete options.
