# VR teleoperation

## Setup

```bash
# Initialize the pinned VR kit checkout.
git submodule update --init --recursive

# Install the main project and recording dependencies.
uv sync --extra lerobot

# Install the relay in its own environment.
uv sync --project external/vr-teleop-kit --extra relay

# Fetch the YAM model files used by inverse kinematics.
git clone https://github.com/i2rt-robotics/i2rt.git ~/i2rt
export YAM_XML="$HOME/i2rt/i2rt/robot_models/arm/yam/yam.xml"
```

Set `YAM_XML` in each terminal that runs teleoperation, or pass
`--yam-xml /path/to/yam.xml`. The `i2rt` checkout supplies models; OpenPI owns
hardware control. `external/vr-teleop-kit` is discovered automatically;
`--vr-kit PATH` selects another checkout.

## Start the relay

Keep this running in a separate terminal:

```bash
uv run --project external/vr-teleop-kit --no-sync vr-teleop-relay
```

The relay listens on `127.0.0.1:8443`. `--vr-url ws://HOST:8443/ws` selects a
different relay for OpenPI commands.

## Connect the Quest browser over USB

Enable Quest developer mode and USB debugging, install `adb` on the host,
then accept the headset's USB debugging prompt.

```bash
# Forward port 8443 and open the relay page in the Quest browser.
uv run --no-sync openpi-control adb connect --open

# Observe incoming frames for five seconds.
uv run --no-sync openpi-control health vr --timeout 5
```

Start teleop on the headset page. Repeat `adb connect` after reconnecting USB.
Use `--serial DEVICE_SERIAL` when multiple Android devices are connected.

## Move virtual or real arms

```bash
# Virtual YAM arms in Viser; no hardware connection or contact physics.
uv run --no-sync openpi-control teleop --backend sim

# Real YAM arms; wait for fresh controller poses, then power up.
uv run --no-sync openpi-control teleop

# Drive only the left arm on can0.
uv run --no-sync openpi-control teleop --only left --interface left=can0

# Record real-arm episodes with an automatically generated dataset ID.
uv run --no-sync openpi-control teleop --record --task "fold the towel"
```

| Control / option | Effect |
| --- | --- |
| Controller grip | Hold to move its arm; release to clutch. |
| Trigger | Close the gripper. |
| Right B / left Y | Start or restart / save a recording episode. |
| `q`, Ctrl-C, Viser Stop | Stop standalone teleop; hardware parks and powers down. |
| `--no-viz` | Omit the standalone hardware Viser display. |
| `--port 8081` | Change the Viser port. |
| `--rate 100` | Set standalone IK update rate in Hz. |
| `--record --no-cameras` | Record state and actions only. |
| `--record --repo-id local/my-session` | Set the dataset ID explicitly. |

Recording uses a 30 FPS loop and has no Viser mirror. See
[recording](recording.md) for save/discard behavior.
