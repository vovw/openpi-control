from types import SimpleNamespace

import pytest

from openpi_control import quest_usb
from openpi_control.exceptions import ConfigurationError


def test_connect_forwards_only_selected_device(monkeypatch):
    calls = []
    monkeypatch.setattr(quest_usb.shutil, "which", lambda _: "/usr/bin/adb")

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0, stderr="", stdout="List of devices attached\nquest device\n"
        )

    monkeypatch.setattr(quest_usb.subprocess, "run", run)
    assert quest_usb.connect_quest() == 0
    assert calls[-1] == ["/usr/bin/adb", "-s", "quest", "reverse", "tcp:8443", "tcp:8443"]


def test_unauthorized_device_does_not_forward(monkeypatch):
    calls = []
    monkeypatch.setattr(quest_usb.shutil, "which", lambda _: "adb")

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="", stdout="quest unauthorized\n")

    monkeypatch.setattr(quest_usb.subprocess, "run", run)
    with pytest.raises(ConfigurationError, match="Allow USB debugging"):
        quest_usb.connect_quest()
    assert len(calls) == 1
