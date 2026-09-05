import copy
import json
from types import SimpleNamespace

import pytest

from openpi_control import quest_native as native
from openpi_control.exceptions import ConfigurationError


def frame(stamp=2_000_000_000_000_000_000):
    payload = {
        "type": "xr_frame",
        "t_unix_ns": stamp,
        "focused": True,
        "viewer": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
        "controllers": {
            "left": {
                "valid": True,
                "tracked": True,
                "active": True,
                "position": [0, 1, 2],
                "orientation": [0, 0, 0, 1],
                "buttons": [{"p": False, "t": False, "v": 0} for _ in range(6)],
                "axes": [0, 0, 0, 0],
            }
        },
    }
    payload["controllers"]["right"] = copy.deepcopy(payload["controllers"]["left"])
    return payload


def test_fresh_frame_once_and_no_cached_replay():
    check = native.FrameFilter(-1_999_999_900, 99.9)
    raw = json.dumps(frame())
    assert json.loads(check.parse(raw, 100.01))["controllers"]["left"]["valid"]
    assert check.parse(raw, 100.02) is None


@pytest.mark.parametrize("now", [99.0, 100.3])
def test_stale_or_future_frame_dropped(now):
    check = native.FrameFilter(-1_999_999_900, 90)
    assert check.parse(json.dumps(frame()), now) is None


def test_prelaunch_frame_dropped():
    assert native.FrameFilter(-1_999_999_900, 101).parse(json.dumps(frame()), 100) is None


@pytest.mark.parametrize(
    "change",
    [
        {"tracked": False},
        {"valid": False},
        {"active": False},
        {"position": [float("nan"), 0, 0]},
        {"orientation": [0, 0, 0, 0]},
        {"buttons": [{"p": "false", "t": False, "v": 0}] * 6},
        {"axes": [float("inf"), 0, 0, 0]},
    ],
)
def test_invalid_hand_removed_and_releases_clutch(change):
    payload = frame()
    payload["controllers"]["left"].update(change)
    result = native.FrameFilter(-1_999_999_900, 90).parse(json.dumps(payload), 100)
    assert "left" not in json.loads(result)["controllers"]


def test_unfocused_frame_clears_controllers():
    payload = frame()
    payload["focused"] = False
    result = native.FrameFilter(-1_999_999_900, 90).parse(json.dumps(payload), 100)
    assert json.loads(result)["controllers"] == {}


def test_invalid_viewer_drops_frame():
    payload = frame()
    payload["viewer"]["orientation"] = [0, 0, float("nan"), 1]
    assert native.FrameFilter(-1_999_999_900, 90).parse(json.dumps(payload), 100) is None


@pytest.mark.parametrize("gap,session", [(0.3, None), (0.01, 2)])
def test_gap_or_new_session_requires_release(gap, session):
    check = native.FrameFilter(-1_999_999_900, 90)
    assert check.parse(json.dumps(frame()), 100)
    payload = frame(2_000_000_000_000_000_000 + int(gap * 1e9))
    payload["session"] = session
    payload["controllers"]["left"]["buttons"][1] = {"p": True, "t": True, "v": 1}
    result = json.loads(check.parse(json.dumps(payload), 100 + gap))
    assert result["controllers"]["left"]["buttons"][1]["p"] is False


def test_lost_tracking_requires_grip_release_before_rearming():
    check = native.FrameFilter(-1_999_999_900, 90)
    first = frame()
    assert check.parse(json.dumps(first), 100)
    lost = frame(first["t_unix_ns"] + 10_000_000)
    lost["controllers"] = {}
    assert check.parse(json.dumps(lost), 100.01)
    restored = frame(first["t_unix_ns"] + 20_000_000)
    restored["controllers"]["left"]["buttons"][1] = {"p": True, "t": True, "v": 1}
    result = json.loads(check.parse(json.dumps(restored), 100.02))
    assert not result["controllers"]["left"]["buttons"][1]["p"]
    released = frame(first["t_unix_ns"] + 30_000_000)
    assert check.parse(json.dumps(released), 100.03)
    restored["t_unix_ns"] += 20_000_000
    result = json.loads(check.parse(json.dumps(restored), 100.04))
    assert result["controllers"]["left"]["buttons"][1]["p"]


@pytest.mark.parametrize("raw", ["noise", "[]", "{}", '{"type":"xr_frame","t_unix_ns":NaN}'])
def test_malformed_frame_dropped(raw):
    assert native.FrameFilter(0, 0).parse(raw, 1) is None


def test_ambiguous_devices_cannot_launch(monkeypatch):
    monkeypatch.setattr(native.shutil, "which", lambda _: "adb")
    monkeypatch.setattr(native, "_run", lambda _: "one device\ntwo device\n")
    with pytest.raises(ConfigurationError, match="select its --serial"):
        native._device(None)


def test_install_only_selected_device(monkeypatch, tmp_path):
    apk = tmp_path / "test.apk"
    apk.touch()
    calls = []
    monkeypatch.setattr(native.shutil, "which", lambda _: "adb")

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="quest device\n", stderr="")

    monkeypatch.setattr(native.subprocess, "run", run)
    assert native.install_apk(apk, "quest") == 0
    assert calls[-1] == ["adb", "-s", "quest", "install", "-r", str(apk)]


def test_slow_clock_calibration_fails_closed(monkeypatch):
    values = iter([0, 1, 2, 3, 4, 5])
    monkeypatch.setattr(native.time, "monotonic", lambda: next(values))
    monkeypatch.setattr(native, "_run", lambda _: "2000000000000000000")
    with pytest.raises(ConfigurationError, match="too slow"):
        native._clock_offset(["adb"])
