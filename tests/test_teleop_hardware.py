import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from openpi_control import cli, teleop_hardware


def test_hardware_command_advances_despite_tracking_lag(monkeypatch):
    """A lagging servo must not pin every requested pose to measured + 0.01."""
    stop = threading.Event()
    monkeypatch.setattr(stop, "wait", lambda _: None)
    rig = SimpleNamespace(names=("left", "right"))
    rig.with_interfaces = lambda _: rig
    monkeypatch.setattr(teleop_hardware, "resolve_rig", lambda _: rig)
    monkeypatch.setattr(teleop_hardware, "resolve_vr_paths", lambda *_: (None, None))
    monkeypatch.setattr(cli, "preflight_rig", lambda _: (0, []))
    initial = {"left": np.full(6, 0.2), "right": np.full(6, -0.3)}
    targets = {"left": np.full(6, 0.6), "right": np.full(6, -0.7)}
    source = Mock()
    source._teleop = SimpleNamespace(
        _lock=threading.Lock(),
        _latest_xr_frame={"controllers": {name: {} for name in rig.names}},
        _last_xr_frame_time=time.time(),
    )
    source.poll.return_value = SimpleNamespace(
        targets={
            name: SimpleNamespace(position_rad=targets[name], effector=0.5) for name in rig.names
        }
    )
    monkeypatch.setattr(teleop_hardware, "QuestTeleopSource", lambda *a, **kw: source)
    issued = {name: [] for name in rig.names}

    def command(name, cmd):
        issued[name].append(cmd.position_rad.copy())
        if name == "right" and len(issued[name]) == 60:
            stop.set()

    arms = [
        SimpleNamespace(name=name, arm=Mock(command=lambda cmd, n=name: command(n, cmd)))
        for name in rig.names
    ]
    monkeypatch.setattr(cli, "power_up", lambda _: (object(), arms))
    shutdown = Mock(return_value=0)
    monkeypatch.setattr(cli, "power_down", shutdown)
    # Fresh feedback with persistent lag, isolating the trajectory generator.
    monkeypatch.setattr(
        cli,
        "_inference_states",
        lambda *a, **kw: {
            name: SimpleNamespace(joints=SimpleNamespace(position_rad=initial[name]))
            for name in rig.names
        },
    )
    args = SimpleNamespace(
        rate=100,
        rig="yam_bimanual",
        interface=None,
        only=None,
        vr_kit=None,
        yam_xml=None,
        record=False,
        vr_url="ws://test",
        no_viz=True,
    )
    assert teleop_hardware.run_hardware(args, stop) == 0
    for name in rig.names:
        trajectory = np.vstack([initial[name], *issued[name]])
        assert np.max(np.abs(np.diff(trajectory, axis=0))) <= 0.010000001
        np.testing.assert_allclose(trajectory[-1], targets[name])
        assert np.all(trajectory >= np.minimum(initial[name], targets[name]) - 1e-9)
        assert np.all(trajectory <= np.maximum(initial[name], targets[name]) + 1e-9)
    source.close.assert_called_once()
    shutdown.assert_called_once()
