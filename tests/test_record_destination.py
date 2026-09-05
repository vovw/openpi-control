from unittest.mock import Mock

import pytest

from openpi_control import cli
from openpi_control.exceptions import ConfigurationError
from openpi_control.record import _create_dataset, check_recording_destination
from openpi_control.rigs import resolve_rig


def test_existing_destination_fails_before_power_up(tmp_path, monkeypatch):
    marker = tmp_path / "preserve.txt"
    marker.write_text("existing recording")
    power = Mock()
    monkeypatch.setattr(cli, "power_up", power)
    with pytest.raises(ConfigurationError, match="Dataset folder already exists"):
        cli.run_record(
            resolve_rig("yam_bimanual"), task="test", repo_id="local/test", root=tmp_path
        )
    power.assert_not_called()
    assert marker.read_text() == "existing recording"


def test_unused_destination_is_not_created(tmp_path):
    target = tmp_path / "new"
    assert check_recording_destination("local/test", target) == target
    assert not target.exists()


def test_creation_race_is_a_friendly_error(tmp_path):
    module = Mock()
    module.LeRobotDataset.create.side_effect = FileExistsError(17, "exists", str(tmp_path))
    with pytest.raises(ConfigurationError, match="Existing recordings were preserved"):
        _create_dataset(module, root=tmp_path)
