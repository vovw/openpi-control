# FR3 and Robotiq

Use an Ethernet-connected FR3 with Robot System 5.9.0 or newer. The native
build pins libfranka 0.21.3. The Robotiq gripper needs Modbus RTU or Modbus TCP.

## Build

```bash
# Build libfranka, libmodbus, and the OpenPI runtime.
sudo ./scripts/install_build_deps_ubuntu.sh
./scripts/build_deps.sh
uv sync
```

## Configure and move

Replace the controller address and serial path. This code connects real
hardware; `move_to_ready()` recovers errors, moves to the reset pose, and
activates and opens the gripper.

```python
from openpi_control import ArmConfig, ArmSession, FR3Connection, RobotiqConnection

config = ArmConfig(
    "follower",
    "FR3",
    FR3Connection("192.168.1.10"),
    effector_model="Robotiq",
    effector_connection=RobotiqConnection.rtu(
        "/dev/serial/by-id/usb-robotiq", baud_rate=115200, slave_id=9,
    ),
)

with ArmSession() as session:
    arm = session.add_follower(config)
    session.connect()
    arm.move_to_ready()
```

For a TCP gripper, replace `effector_connection` with:

```python
RobotiqConnection.tcp("192.168.1.11", port=502)
```

| API / setting | Effect |
| --- | --- |
| `session.connect()` | Connect and hold the measured arm pose; gripper remains inactive. |
| `arm.move_to_ready()` | Recover, move to reset pose, activate and open the gripper. |
| `arm.command(PositionCommand(joints, gripper))` | Set seven joint targets in radians and a gripper target. |
| Gripper `0.0` / `1.0` | Fully closed / fully open. |
| libfranka callback | Runs torque control at 1 kHz; policy commands can arrive more slowly. |

The driver uses `RealtimeConfig::kIgnore`, so a real-time kernel is optional.
RTU-over-TCP gateways are not serial RTU connections.
