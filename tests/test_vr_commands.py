import json
from contextlib import nullcontext
from unittest.mock import Mock

from openpi_control import cli, teleop_hardware, vr_health


def test_health_reports_missing_frames(monkeypatch, capsys):
    import websockets.sync.client

    ws = Mock()
    ws.recv.side_effect = [json.dumps({"type": "camera_list"}), TimeoutError()]
    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **kw: nullcontext(ws))
    assert vr_health.check_vr("ws://test", 0.01) == 1
    assert "no valid headset controller poses" in capsys.readouterr().out


def test_teleop_defaults_to_hardware_without_recording(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.runlog, "setup_run_logging", lambda _: tmp_path / "test.log")
    called = Mock(return_value=0)
    monkeypatch.setattr(teleop_hardware, "run_hardware", called)
    assert cli.main(["teleop", "--plain"]) == 0
    args, stop = called.call_args.args
    assert args.backend == "hardware"
    assert not args.record
    assert not stop.is_set()


def test_recording_names_are_unique():
    assert teleop_hardware.new_recording_id() != teleop_hardware.new_recording_id()
