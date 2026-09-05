import pytest

from openpi_control.exceptions import ConfigurationError
from openpi_control.teleop_sim import decode_action


def test_sim_action_mapping():
    action = {f"left_joint_{i}.pos": i / 10 for i in range(1, 7)}
    action["left_gripper.pos"] = 0.25
    joints, grips = decode_action(action, ("left",))
    assert joints["left"].tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert grips == {"left": 0.75}
    action["left_joint_1.pos"] = float("nan")
    with pytest.raises(ConfigurationError):
        decode_action(action, ("left",))


def test_sim_rejects_partial_actions():
    with pytest.raises(ConfigurationError):
        decode_action({}, ("left",))
