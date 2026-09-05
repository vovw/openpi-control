# Record teleoperation

Complete [VR setup](vr.md), start the relay, and connect the Quest first.
Recording commands power real arms. Python 3.12 or 3.13 is required for LeRobot.

```bash
# Install dataset support.
uv sync --extra lerobot

# Record with an automatically generated dataset ID.
uv run --no-sync openpi-control teleop --record --task "fold the towel"

# Record 20 saved episodes into an explicit directory.
uv run --no-sync openpi-control record \
    --repo-id local/yam-fold-towel --task "fold the towel" \
    --root ~/openpi-data/recordings/yam-fold-towel --num-episodes 20

# Record joints and actions without cameras.
uv run --no-sync openpi-control record \
    --repo-id local/state-only --task "move the arm" --no-cameras

# Exercise the recording loop without writing a dataset; still powers real arms.
uv run --no-sync openpi-control record --teleop hold --hold-seconds 5 --dry-run
```

## Episode controls

| Control | Effect |
| --- | --- |
| Hold controller grip | Move the corresponding arm. |
| Trigger | Close the gripper. |
| Right B | Start recording; if already recording, discard and restart the take. |
| Left Y | Save the current episode. |
| Ctrl-C | Discard an open episode, finish the dataset, park, and power down. |

Arms remain controllable between episodes. Press Y before exiting to keep the
current take. `rollout` saves partial episodes differently; see [inference](inference.md).

## Recording options

| Option | Effect |
| --- | --- |
| `--fps N` | Dataset and command rate; default 30 Hz. |
| `--camera-fps N` | Separate camera rate; otherwise follows `--fps`. |
| `--num-episodes N` | Stop after N saved episodes; 0 runs until interrupted. |
| `--only left` | Use one arm and its wrist camera, plus the top camera. |
| `--interface left=can0` | Override an arm's CAN interface. |
| `--camera top=/dev/video4` | Override a camera device. |
| `--vr-kit PATH` | Select another VR kit checkout. |
| `--vr-url ws://HOST:8443/ws` | Select another relay. |
| `--yam-xml PATH` | Select the YAM IK model. |
| `--push-to-hub` | Upload after recording and hardware shutdown. |
| `--private` | Make the uploaded dataset private. |

Use a new dataset destination for each session. Set `--repo-id` to your
Hugging Face namespace when uploading. A failed upload leaves the local data.

Datasets use LeRobot v3 with measured joint state, commanded actions, task
text, and camera video. Native grippers use 1 = open; dataset grippers use
0 = open. The recorder converts between them.
