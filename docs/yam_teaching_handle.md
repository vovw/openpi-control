# YAM teaching handle

The `E_Yam_Handle` leader effector supplies a trigger and top/bottom buttons.
It has no driven gripper or joystick.

## Check configuration

```bash
# Check the leader model and CAN interface without powering motors.
uv run --no-sync openpi-control doctor \
    --model Yam --interface can_leader --effector E_Yam_Handle

# Print the servo-zero plan without opening the bus.
uv run --no-sync openpi-control zero \
    --model Yam --interface can_leader --effector E_Yam_Handle --dry-run
```

## Read inputs

This connects a real leader arm. Use the cell's CAN interface name.

```python
from openpi_control import ArmConfig, ArmSession, SocketCanConnection

with ArmSession() as session:
    leader = session.add_leader(ArmConfig(
        "leader", "Yam", SocketCanConnection("can_leader"),
        effector_model="E_Yam_Handle",
    ))
    session.connect()
    print(leader.read_inputs())
```

| Input / setting | Meaning |
| --- | --- |
| Trigger `1.0` / `0.0` | Released (open) / fully squeezed (closed). |
| Buttons | `top` and `bottom`, from digital bits 0 and 1. |
| Firmware | Version 2.2.12 or newer is required at startup. |
| CAN request / response IDs | Default `0x50E` / `0x50F`. |
| Trigger calibration | `pos_max` in `E_Yam_Handle_01.json`; shipped value 0.63 rad. |
| Polling | At most 250 Hz; missing replies trigger retry backoff. |

Startup validates firmware and corrects mismatched ADC/report settings before
motor enable. Input loss holds the last trigger value, warns after 2 seconds,
attempts one firmware restart after 3 seconds, and raises safe mode after
20 seconds. Implementation: `native/pi_control/src/pi_servo_can_encoder.cpp`.

Run the handle's native coverage using the [SIL commands](development.md#software-in-the-loop-tests).
