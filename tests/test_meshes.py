"""Visual-mesh fetching and caching. Never touches a bus; never needs network."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from openpi_control import meshes
from openpi_control.exceptions import ConfigurationError


@pytest.fixture
def no_network(monkeypatch):
    """Fail any HTTP attempt, so a test that should stay local proves it."""
    calls = []

    def blocked(url, *args, **kwargs):
        calls.append(url)
        raise OSError("network disabled for test")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    return calls


@pytest.fixture
def fake_vendor(monkeypatch):
    """Serve STL bytes for every name except those in ``absent``."""
    served = []

    def make(absent: set[str] = frozenset()):
        # Fresh log per call, so a test can re-arm the vendor mid-way.
        served.clear()

        class _Response:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def fetch(url, *args, **kwargs):
            name = url.rsplit("/", 1)[-1]
            served.append(name)
            if name in absent:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            return _Response(b"solid fake\n" + name.encode())

        monkeypatch.setattr(urllib.request, "urlopen", fetch)
        return served

    return make


def test_yam_mesh_names_come_from_the_packaged_urdf() -> None:
    names = meshes.urdf_mesh_names("Yam")
    # Yam.urdf carries a visual and a collision mesh for the base and six links.
    assert len(names) == 14
    assert "base_link_visual.stl" in names
    assert "link_6_visual.stl" in names
    assert all(name.endswith(".stl") for name in names)
    # Basenames only — the URDF writes package://assets/<name>.
    assert all("/" not in name for name in names)


def test_mesh_names_for_a_model_without_a_urdf_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="ships no URDF"):
        meshes.urdf_mesh_names("FR3")


def test_unknown_model_has_no_mesh_source(no_network) -> None:
    with pytest.raises(ConfigurationError, match="no mesh source is known"):
        meshes.fetch_meshes("SO101")
    assert not no_network, "must fail before touching the network"


def test_fetch_writes_every_available_mesh(tmp_path, fake_vendor) -> None:
    served = fake_vendor(absent={"link_6_visual.stl", "link_6_collision.stl"})
    report = meshes.fetch_meshes("Yam", destination=tmp_path)

    assert len(report.fetched) == 12
    assert report.available == 12
    assert sorted(report.unavailable) == ["link_6_collision.stl", "link_6_visual.stl"]
    assert len(served) == 14
    for name in report.fetched:
        assert (tmp_path / name).read_bytes().startswith(b"solid fake")
    # No half-written files left behind.
    assert not list(tmp_path.glob("*.partial"))


def test_fetch_records_provenance(tmp_path, fake_vendor) -> None:
    fake_vendor()
    meshes.fetch_meshes("Yam", destination=tmp_path)
    note = (tmp_path / "SOURCE.txt").read_text()
    assert "MIT" in note
    assert "I2RT Robotics" in note
    # Pinned revision, so a cached directory cannot silently mean two things.
    assert meshes.MESH_SOURCES["Yam"].revision in note


def test_second_fetch_needs_no_network(tmp_path, fake_vendor, monkeypatch) -> None:
    """Everything cached, and the upstream gaps remembered."""
    fake_vendor(absent={"link_6_visual.stl", "link_6_collision.stl"})
    first = meshes.fetch_meshes("Yam", destination=tmp_path)
    assert first.fetched

    calls = []

    def blocked(url, *args, **kwargs):
        calls.append(url)
        raise OSError("network disabled for test")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    second = meshes.fetch_meshes("Yam", destination=tmp_path)

    assert not calls, f"re-fetch hit the network for {calls}"
    assert len(second.already_present) == 12
    assert sorted(second.unavailable) == ["link_6_collision.stl", "link_6_visual.stl"]


def test_force_refetches_everything(tmp_path, fake_vendor) -> None:
    fake_vendor()
    meshes.fetch_meshes("Yam", destination=tmp_path)
    served = fake_vendor()
    report = meshes.fetch_meshes("Yam", destination=tmp_path, force=True)
    assert len(report.fetched) == 14
    assert not report.already_present
    assert len(served) == 14


def test_a_server_error_is_not_mistaken_for_an_absent_mesh(tmp_path, monkeypatch) -> None:
    def failing(url, *args, **kwargs):
        raise urllib.error.HTTPError(url, 500, "Server Error", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", failing)
    with pytest.raises(ConfigurationError, match="HTTP 500"):
        meshes.fetch_meshes("Yam", destination=tmp_path)


def test_a_network_failure_says_the_cache_helps_later(tmp_path, no_network) -> None:
    with pytest.raises(ConfigurationError, match="needs network access"):
        meshes.fetch_meshes("Yam", destination=tmp_path)


def test_cached_mesh_dir_is_none_until_a_mesh_lands(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(meshes, "mesh_cache_root", lambda: tmp_path)
    monkeypatch.setattr(meshes, "mesh_cache_dir", lambda model: tmp_path / model)
    assert meshes.cached_mesh_dir("Yam") is None

    directory = tmp_path / "Yam"
    directory.mkdir()
    # A provenance note alone is not geometry.
    (directory / "SOURCE.txt").write_text("note\n")
    assert meshes.cached_mesh_dir("Yam") is None

    (directory / "base_link_visual.stl").write_bytes(b"solid\n")
    assert meshes.cached_mesh_dir("Yam") == directory


def test_cache_lives_under_the_openpi_data_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENPI_LOG_DIR", str(tmp_path / "logs"))
    assert meshes.mesh_cache_root() == tmp_path / "meshes"
    assert meshes.mesh_cache_dir("Yam") == tmp_path / "meshes" / "Yam"
