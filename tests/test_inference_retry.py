from unittest.mock import Mock

import pytest

from openpi_control.cli import _wait_for_inference_server
from openpi_control.inference import InferenceConnectionError, InferenceError


def test_startup_retries_every_five_seconds_until_connected():
    client = Mock()
    client.health.side_effect = [InferenceConnectionError("offline"), {}]
    stop = Mock()
    stop.is_set.return_value = False
    stop.wait.return_value = False
    assert _wait_for_inference_server(client, stop)
    stop.wait.assert_called_once_with(5.0)
    assert client.health.call_count == 2


def test_stop_cancels_retry_without_another_request():
    client = Mock()
    client.health.side_effect = InferenceConnectionError("offline")
    stop = Mock()
    stop.is_set.return_value = False
    stop.wait.return_value = True
    assert not _wait_for_inference_server(client, stop)
    assert client.health.call_count == 1


def test_bad_protocol_does_not_retry():
    client = Mock()
    client.health.side_effect = InferenceError("wrong state dimension")
    stop = Mock()
    stop.is_set.return_value = False
    with pytest.raises(InferenceError, match="wrong state dimension"):
        _wait_for_inference_server(client, stop)
    stop.wait.assert_not_called()
