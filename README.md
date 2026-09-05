# openpi-control

Control YAM, ARX, FR3, SO101, and Trossen arms from Python or the CLI.

## Install

```bash
# Clone the project and its VR submodule.
git clone --recurse-submodules https://github.com/vovw/openpi-control.git
cd openpi-control

# Install native build dependencies, then the Python environment.
sudo ./scripts/install_build_deps_ubuntu.sh
./scripts/build_deps.sh
uv sync

# Add dataset recording (Python 3.12 or 3.13).
uv sync --extra lerobot
```

Already cloned? Run `git submodule update --init --recursive`.

## Check the cell

```bash
# Check arm configuration and interfaces without powering motors.
uv run --no-sync openpi-control doctor --rig yam_bimanual

# Open cameras and save one image per view.
uv run --no-sync openpi-control cameras --probe --snapshot /tmp/cameras
```

The packaged YAM rig uses `can_left` and `can_right`. Use
`--interface left=YOUR_CAN --interface right=YOUR_CAN` on motion commands to
change them; `doctor` uses `--interface-override` instead.

## Move the arms

```bash
# Power both arms and show measured poses in Viser.
uv run --no-sync openpi-control live

# Enable browser controls; confirm the pose and click Arm before moving sliders.
uv run --no-sync openpi-control live --control

# Make the arms backdrivable with gravity compensation.
uv run --no-sync openpi-control live --float
```

Open the printed Viser URL. Ctrl-C parks the arms and powers them down.

## VR teleoperation

Complete [VR setup](docs/vr.md) and keep the relay running first.

```bash
# Forward the USB-connected Quest to the relay and open its browser page.
uv run --no-sync openpi-control adb connect --open

# Check incoming controller frames.
uv run --no-sync openpi-control health vr

# Move virtual arms only.
uv run --no-sync openpi-control teleop --backend sim

# Move real arms. Hold a controller grip to move; release to clutch.
uv run --no-sync openpi-control teleop

# Move real arms and record episodes. B starts/restarts; Y saves.
uv run --no-sync openpi-control teleop --record --task "fold the towel"
```

## Run a policy

Start a compatible MolmoAct server separately. Replace the address below.

```bash
# Execute policy actions on both arms.
uv run --no-sync openpi-control infer \
    --server http://POLICY_HOST:8202 --instruction "fold the towel"

# Record three timed attempts; enter an instruction and success label per attempt.
uv run --no-sync openpi-control rollout \
    --server http://POLICY_HOST:8202 --repo-id local/towel-rollouts \
    --episodes 3 --episode-seconds 120
```

## Commands and development

- [CLI](docs/cli.md): preflight, calibration, and live control.
- [VR](docs/vr.md): relay, Quest USB, simulation, and hardware.
- [Cameras](docs/cameras.md): discovery, snapshots, and overrides.
- [Recording](docs/recording.md): teleop datasets and episode controls.
- [Inference](docs/inference.md): policy execution and rollout options.
- [Viser](docs/viser.md): model viewer and live visualization.
- [FR3](docs/fr3.md): Franka and Robotiq configuration.
- [YAM teaching handle](docs/yam_teaching_handle.md): trigger and buttons.
- [Development](docs/development.md): builds and tests.
