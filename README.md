# openpi-basic-control

> [!WARNING]
> `openpi-basic-control` is meant to be a minimal but hackable working example. It is not thoroughly documented or tested, and should not be considered a production-quality solution. We (Physical Intelligence) may not have the bandwidth to maintain this repository, or review any contributions. Feel free to fork it instead!

`openpi-basic-control` is a minimal C++ robot-control library. Each arm
runs one `pi_control_node` process that communicates with the main
Python process via ZeroMQ.

| Arm | Joints | Bus | Follower effector | Leader effector |
| --- | --- | --- | --- | --- |
| `Yam` | 6 | SocketCAN | `E_Yam` | `E_Yam_Handle` |
| `ARX_X5` | 6 | SocketCAN | `E_ARX` | `E_ARX_ENC` |
| `ARX_L5` | 6 | SocketCAN | `E_ARX` | `E_ARX_ENC` |
| `ARX_ENC` | 6 | SocketCAN | — (leader only, read-only) | `E_ARX_ENC` |
| `FR3` | 7 | Franka controller | `Robotiq` | — |
| `SO101` | 5 | USB serial | `E_SO101` | — |
| `Trossen_wai_ctrl` | 6 | Ethernet controller | `E_Trossen_ctrl` | — |

```python
from openpi_control import ArmConfig, ArmSession, PositionCommand, SocketCanConnection

with ArmSession() as session:
    follower = session.add_follower(
        ArmConfig(
            "right_follower",
            "Yam",
            SocketCanConnection("can_follower_r"),
            effector_model="E_Yam",
            follower_gravity_compensation=True,
        )
    )
    session.connect()
    follower.command(PositionCommand([0, 0, 0, 0, 0, 0], 1.0))
```

FR3 uses the same `ArmSession`, `FollowerArm`, and `PositionCommand` API. Its
connection selects one of the two real Robotiq Modbus transports:

```python
from openpi_control import FR3Connection, RobotiqConnection

config = ArmConfig(
    "follower",
    "FR3",
    FR3Connection("192.168.1.10"),
    effector_model="Robotiq",
    effector_connection=RobotiqConnection.rtu("/dev/serial/by-id/usb-robotiq"),
    # Or: RobotiqConnection.tcp("192.168.1.11", port=502),
)
```

See [docs/fr3.md](docs/fr3.md) for firmware, networking, controller, and
hardware-validation details.

## Operator CLI

`openpi-control` preflights an arm, sets its servo zeros, and brings a rig up
and back down. Every run logs to `~/openpi-data/logs/runtime/`.

```bash
uv run openpi-control doctor --model Yam --interface can_follower_r --effector E_Yam
uv run openpi-control zero   --model Yam --interface can_follower_r --dry-run
uv run openpi-control doctor --rig yam_bimanual        # both arms, read-only
uv run openpi-control live   --rig yam_bimanual        # energize, mirror, park
```

`doctor` is read-only and opens no bus unless given `--probe`; it checks the
packaged assets, that every `servo_model` has a driver, and that the interface
is present, up, and at the bit rate the model wants. `zero` writes the arm's
current pose as each servo's firmware zero, so it confirms first. See
[docs/cli.md](docs/cli.md).

## Rigs, and turning them on and off

A rig names a whole cell — which arms it has, the bus each sits on, and where
their bases sit relative to each other — so the CLI and the visualizer mean the
same thing by `left`. `yam_bimanual` is two YAM followers with `E_Yam` grippers,
left on `can0` and right on `can1`.

`openpi-control live` is the one command here that energizes an arm, and it owns
the whole arc:

```bash
uv run openpi-control live --rig yam_bimanual            # both arms up, holding
uv run openpi-control live --rig yam_bimanual --float    # backdrivable instead
uv run openpi-control live --rig yam_bimanual --only left --no-viz
```

Preflight runs first and nothing is energized unless every arm passes. Both
followers then come up *holding* — `--float` is the opt-in for a backdrivable
arm. ctrl-c parks each arm at the `home_pos` in its instance JSON before cutting
torque, so an arm is never dropped from wherever it stood.

There is no separate `up` and `down`: `pi_control_node` holds a liveness pipe to
the process that spawned it, so an arm cannot stay energized after the command
returns. One foreground process owns the lifecycle, and ctrl-c is the way out.

## Cameras

Three RealSense D405s watch the bimanual cell: one overhead, one per wrist. They
are part of the rig, pinned by serial number — a `/dev/videoN` is not a camera,
it changes with boot order and which port you used.

```bash
uv sync --extra cameras
uv run openpi-control cameras                # what is plugged in, and where
uv run openpi-control cameras --probe        # open each one and measure it
```

A wrist camera names the arm it rides on, so `--only right` narrows to `top` and
`right_wrist` without anyone special-casing it. Discovery itself reads udev and
nothing else, so `doctor --rig` can report an unplugged camera on a box with no
SDK installed.

Two defaults are measurements rather than taste: 848x480 (the D405's native
mode — 640x480 makes the firmware rescale and three cameras drop to 15-20 fps),
and capture through `pyrealsense2` rather than OpenCV (whose V4L2 path tops out
near 10-13 fps on the same node that `v4l2-ctl` streams at 30). See
[docs/cameras.md](docs/cameras.md).

## Collecting data

`openpi-control record` teleoperates a rig from a Meta Quest and writes the
episodes as a LeRobot dataset — parquet for state and action, one mp4 per camera.

```bash
uv sync --extra lerobot
vr-teleop-relay                                     # in the vr-teleop-kit checkout
uv run openpi-control record --repo-id you/yam-fold-towel \
    --task "fold the towel in half" --num-episodes 20 --vr-kit ~/vr-teleop-kit
```

VR teleoperation itself is not reimplemented here:
[`vr-teleop-kit`](https://github.com/Dream-Machines-Robotics/vr-teleop-kit) owns
the WebXR relay, the pose mapping, and the YAM inverse kinematics, and
`openpi_control.teleop_vr` adapts it so the headset drives arms through this
package's native stack. Right B starts an episode, left Y saves it.

`--teleop hold --dry-run` rehearses a whole session — arms up, cameras open,
nothing written — so the pipeline can be checked without a headset.

Note that the gripper convention is **inverted** between this package (`1.0` =
open) and LeRobot (`0.0` = open); recorded datasets use LeRobot's, and
`record.to_native_gripper` is the one place that converts back. That and the
other three ways a dataset comes out quietly wrong are in
[docs/recording.md](docs/recording.md).

## Visualizing an arm

`openpi_control.viz` serves any packaged model in the browser with
[viser](https://viser.studio). It opens no bus and starts no native node, so it
runs with the hardware down or absent.

```bash
uv sync --extra viz
uv run openpi-control-viz --fetch-meshes --model Yam     # once, needs network
uv run openpi-control-viz --model Yam --effector E_Yam   # http://localhost:8080
```

The wheel ships each URDF but not its meshes — the URDFs are here for the
gravity-compensation model, which needs link inertias and joint origins and
never needs geometry. `--fetch-meshes` caches I2RT's YAM meshes (MIT) under
`~/openpi-data/meshes/`, after which every run renders the real arm offline with
no flags. Without them you get a kinematic skeleton built from the joint
origins, which needs no assets at all.

GUI sliders drive the joints. To follow a live arm instead, hand it poses from
your own session — the visualizer only draws, so hardware stays your call:

```python
from openpi_control.viz import ArmVisualizer

viz = ArmVisualizer("Yam", effector_model="E_Yam")
viz.update(follower.read_state().joints.position_rad)
```

`ArmSceneVisualizer` puts several arms in one scene, each with its own base
pose. Pass it a packaged rig to draw a whole cell:

```bash
uv run openpi-control-viz --rig yam_bimanual   # both arms, sliders, no bus
```

To mirror two *live* arms instead of sliders, use `openpi-control live` — it
owns the power-on and power-off that a live view implies. See
[docs/viser.md](docs/viser.md).

## Documentation

| Doc | Covers |
| --- | --- |
| [docs/cli.md](docs/cli.md) | `doctor` checks, `zero` safeguards, rigs, `live` power on/off |
| [docs/cameras.md](docs/cameras.md) | camera identity, discovery, the two D405 serials, capture rates |
| [docs/recording.md](docs/recording.md) | LeRobot datasets, VR teleop, gripper polarity, episode boundaries |
| [docs/viser.md](docs/viser.md) | render modes, mesh sourcing, rigs, joint ordering |
| [docs/fr3.md](docs/fr3.md) | FR3 firmware, networking, controller, validation |
| [docs/yam_teaching_handle.md](docs/yam_teaching_handle.md) | YAM handle CAN protocol and trigger calibration |

## Building from source

Building requires CMake and a C++17 compiler. The dependency builder pins and
builds Pinocchio, ZeroMQ, cppzmq, Trossen, libfranka 0.21.3, and libmodbus;
the resulting libfranka and libmodbus archives are linked into
`pi_control_node`. It has been tested on Ubuntu 22.04 and 24.04.

```bash
sudo ./scripts/install_build_deps_ubuntu.sh
./scripts/build_deps.sh
uv build --wheel
```

The wheel is written to `dist/`.
