"""Viser visualization of the packaged models. Never touches a bus."""

from __future__ import annotations

import numpy as np
import pytest

from openpi_control.config import SUPPORTED_MODELS, resolve_model_assets
from openpi_control.exceptions import ConfigurationError
from openpi_control.rigs import resolve_rig, rig_names

viz = pytest.importorskip("openpi_control.viz", reason="needs the 'viz' extra")
pytest.importorskip("viser", reason="needs the 'viz' extra")
pytest.importorskip("yourdfpy", reason="needs the 'viz' extra")

# Models that ship a URDF; FR3 defers its model to the vendor controller.
URDF_MODELS = [m for m in SUPPORTED_MODELS if resolve_model_assets(m).urdf is not None]

# Ports are per-test so a leaked server cannot wedge the next case.
_PORT = iter(range(8500, 8700))


@pytest.fixture(autouse=True)
def isolate_mesh_cache(monkeypatch):
    """Hide any real ~/openpi-data/meshes cache.

    Without this, a developer who has run --fetch-meshes gets different render
    modes than CI. Tests that want a cache patch this again themselves.
    """
    from openpi_control import meshes

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: None)


@pytest.fixture
def make_viz():
    created = []

    def factory(model: str, **kwargs):
        instance = viz.ArmVisualizer(model, port=next(_PORT), **kwargs)
        created.append(instance)
        return instance

    yield factory
    for instance in created:
        instance.server.stop()


@pytest.fixture
def make_scene():
    created = []

    def factory(rig, **kwargs):
        scene = viz.ArmSceneVisualizer.from_rig(rig, port=next(_PORT), **kwargs)
        created.append(scene)
        return scene

    yield factory
    for scene in created:
        scene.server.stop()


def _load_urdf(model: str):
    yourdfpy = viz.import_viz_modules().yourdfpy
    return yourdfpy.URDF.load(
        str(resolve_model_assets(model).urdf),
        load_meshes=False,
        build_collision_scene_graph=False,
    )


@pytest.mark.parametrize("model", URDF_MODELS)
def test_every_packaged_urdf_renders_without_meshes(model: str, make_viz) -> None:
    """The wheel ships no assets/*.stl, so the mesh-free path must carry all models."""
    instance = make_viz(model)
    assert instance.render_mode == "skeleton"
    assert instance.joint_names
    assert len(instance._link_frames) == len(instance._urdf.link_map)


@pytest.mark.parametrize("model", URDF_MODELS)
def test_joint_order_follows_the_kinematic_chain(model: str) -> None:
    """Arm state is indexed along the chain, not by URDF document order.

    Yam.urdf and SO101.urdf both declare their joints tip-first, so trusting
    yourdfpy's document order would drive joint 1 from joint 6's command.
    """
    urdf = _load_urdf(model)
    chain = viz.chain_ordered_actuated_joints(urdf)
    assert sorted(chain) == sorted(urdf.actuated_joint_names)

    # Depth from the base link, counting fixed joints too (ARX_L5 reaches its
    # base_link from a "world" root through a fixed joint).
    depth = {urdf.base_link: 0}
    pending = list(urdf.robot.joints)
    while pending:
        remaining = [j for j in pending if j.parent not in depth]
        for joint in pending:
            if joint.parent in depth:
                depth[joint.child] = depth[joint.parent] + 1
        if len(remaining) == len(pending):
            break
        pending = remaining

    depths = [depth[urdf.joint_map[name].child] for name in chain]
    assert depths == sorted(depths), f"{model} chain is not ordered base-to-tip: {depths}"


def test_yam_chain_order_is_base_to_tip() -> None:
    urdf = _load_urdf("Yam")
    assert tuple(urdf.actuated_joint_names) == (
        "joint6",
        "joint5",
        "joint4",
        "joint3",
        "joint2",
        "joint1",
    )
    assert viz.chain_ordered_actuated_joints(urdf) == (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )


def test_update_matches_a_by_name_reference(make_viz) -> None:
    """A positional update must land on the joints its index names."""
    instance = make_viz("Yam")
    pose = [0.3, 0.9, 0.7, -0.2, 0.4, -0.6]

    reference = _load_urdf("Yam")
    reference.update_cfg(dict(zip(instance.joint_names, pose, strict=True)))
    expected = reference.get_transform("end_link", reference.base_link)

    instance.update(pose)
    actual = instance._urdf.get_transform("end_link", instance._urdf.base_link)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_first_joint_swings_the_tip_about_the_base_axis(make_viz) -> None:
    instance = make_viz("Yam")
    instance.update([0.0] * 6)
    rest = instance._urdf.get_transform("end_link", instance._urdf.base_link)[:3, 3].copy()
    instance.update([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    swung = instance._urdf.get_transform("end_link", instance._urdf.base_link)[:3, 3]
    assert abs(swung[2] - rest[2]) < 1e-6
    assert np.linalg.norm(swung[:2] - rest[:2]) > 1e-3


@pytest.mark.parametrize("model", URDF_MODELS)
def test_positions_are_clamped_to_the_urdf_limits(model: str, make_viz) -> None:
    instance = make_viz(model)
    count = len(instance.joint_specs)
    lower = np.array([spec.lower for spec in instance.joint_specs])
    upper = np.array([spec.upper for spec in instance.joint_specs])

    instance.update([1e3] * count)
    np.testing.assert_array_less(instance.positions, upper + 1e-9)
    instance.update([-1e3] * count)
    np.testing.assert_array_less(lower - 1e-9, instance.positions)


def test_rest_pose_is_inside_the_limits(make_viz) -> None:
    for model in URDF_MODELS:
        instance = make_viz(model)
        for spec in instance.joint_specs:
            assert spec.lower <= spec.rest <= spec.upper


def test_update_accepts_a_partial_mapping(make_viz) -> None:
    instance = make_viz("Yam")
    instance.update([0.1] * 6)
    instance.update({"joint3": 0.5})
    assert instance.positions[2] == pytest.approx(0.5)
    assert instance.positions[0] == pytest.approx(0.1)


def test_update_rejects_bad_input(make_viz) -> None:
    instance = make_viz("Yam")
    with pytest.raises(ConfigurationError, match="expects 6 joint positions"):
        instance.update([0.0, 0.0])
    with pytest.raises(ConfigurationError, match="finite"):
        instance.update([float("nan")] * 6)
    with pytest.raises(ConfigurationError, match="unknown joint names"):
        instance.update({"not_a_joint": 0.0})


def test_fr3_reports_that_it_ships_no_urdf() -> None:
    with pytest.raises(ConfigurationError, match="ships no URDF"):
        viz.ArmVisualizer("FR3", port=next(_PORT))


def test_unsupported_model_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unsupported model"):
        viz.ArmVisualizer("NotAnArm", port=next(_PORT))


def test_missing_mesh_dir_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="not a directory"):
        viz.ArmVisualizer("Yam", mesh_dir="/nonexistent/assets", port=next(_PORT))


def test_mesh_mode_permutes_into_viser_urdf_order(tmp_path, make_viz) -> None:
    """ViserUrdf indexes by its own actuated order; ours is chain order."""
    trimesh = viz.import_viz_modules().trimesh
    assets = tmp_path / "assets"
    assets.mkdir()
    for link in ("base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"):
        for kind in ("visual", "collision"):
            trimesh.creation.box(extents=(0.05, 0.05, 0.08)).export(
                assets / f"{link}_{kind}.stl"
            )

    instance = make_viz("Yam", mesh_dir=assets)
    assert instance.render_mode == "mesh"
    vis_order = list(instance._urdf_vis.get_actuated_joint_names())
    permuted = [vis_order[i] for i in instance._vis_permutation]
    assert tuple(permuted) == instance.joint_names

    instance.update([0.0] * 6)
    rest = instance._urdf.get_transform("end_link", instance._urdf.base_link)[:3, 3].copy()
    instance.update([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    swung = instance._urdf.get_transform("end_link", instance._urdf.base_link)[:3, 3]
    assert abs(swung[2] - rest[2]) < 1e-6, "joint1 must rotate about the base axis"


def test_gui_builds_one_slider_per_joint(make_viz) -> None:
    instance = make_viz("Yam", effector_model="E_Yam")
    instance.add_gui()
    assert len(instance._sliders) == len(instance.joint_specs)


def test_cli_list_reports_every_model(capsys) -> None:
    assert viz.main(["--list"]) == 0
    out = capsys.readouterr().out
    for model in SUPPORTED_MODELS:
        assert model in out
    assert "no URDF" in out  # FR3


# --------------------------------------------------------------------------- #
# Mesh discovery
# --------------------------------------------------------------------------- #


def test_mesh_dir_falls_back_to_the_local_cache(tmp_path, monkeypatch, make_viz) -> None:
    """With no mesh_dir given, a filled cache should render the real arm."""
    from openpi_control import meshes

    cache = tmp_path / "Yam"
    cache.mkdir()
    trimesh = viz.import_viz_modules().trimesh
    for name in meshes.urdf_mesh_names("Yam"):
        if "link_6" in name:
            continue  # absent upstream, so absent here too
        trimesh.creation.box(extents=(0.05, 0.05, 0.08)).export(cache / name)

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: cache if model == "Yam" else None)
    instance = make_viz("Yam")
    assert instance.render_mode == "mesh"
    assert instance.mesh_dir == cache


def test_an_empty_cache_leaves_the_skeleton(monkeypatch, make_viz) -> None:
    from openpi_control import meshes

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: None)
    instance = make_viz("Yam")
    assert instance.render_mode == "skeleton"
    assert instance.missing_meshes == ()
    assert "--fetch-meshes" in instance.mesh_status()


def test_missing_meshes_are_reported_not_hidden(tmp_path, monkeypatch, make_viz) -> None:
    """i2rt ships no link_6 geometry; the gap should be stated, not silent."""
    from openpi_control import meshes

    cache = tmp_path / "Yam"
    cache.mkdir()
    trimesh = viz.import_viz_modules().trimesh
    for name in meshes.urdf_mesh_names("Yam"):
        if "link_6" in name:
            continue
        trimesh.creation.box(extents=(0.05, 0.05, 0.08)).export(cache / name)

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: cache)
    instance = make_viz("Yam")
    assert set(instance.missing_meshes) == {"link_6_visual.stl", "link_6_collision.stl"}
    assert "missing upstream (2)" in instance.mesh_status()
    instance.add_gui()  # must not raise with a missing-mesh row present


def test_explicit_mesh_dir_beats_the_cache(tmp_path, monkeypatch, make_viz) -> None:
    from openpi_control import meshes

    monkeypatch.setattr(meshes, "cached_mesh_dir", lambda model: tmp_path / "cache")
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    trimesh = viz.import_viz_modules().trimesh
    trimesh.creation.box(extents=(0.05, 0.05, 0.08)).export(explicit / "base_link_visual.stl")

    instance = make_viz("Yam", mesh_dir=explicit)
    assert instance.mesh_dir == explicit


# --------------------------------------------------------------------------- #
# Rigs in one scene
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rig_name", rig_names())
def test_every_packaged_rig_renders_as_one_scene(rig_name: str, make_scene) -> None:
    """A rig has to reach the browser, not just resolve: this is what --rig runs."""
    rig = resolve_rig(rig_name)
    scene = make_scene(rig)

    assert scene.names == rig.names
    assert scene.label == rig.name
    # One shared server, so the arms land in a single browser scene.
    for name in scene.names:
        assert scene[name].server is scene.server


def test_from_rig_places_each_arm_at_its_rig_base_pose(make_scene) -> None:
    # Without this the two YAM arms render on top of each other and a bimanual
    # scene is unreadable.
    rig = resolve_rig("yam_bimanual")
    scene = make_scene(rig)

    for arm in rig.arms:
        rendered = np.asarray(scene[arm.name].base_frame.position, dtype=np.float64)
        assert rendered == pytest.approx(np.asarray(arm.base_position), abs=1e-6)


def test_from_rig_carries_the_effector_and_tints_the_arms_apart(make_scene) -> None:
    scene = make_scene(resolve_rig("yam_bimanual"))

    assert [scene[n].effector_model for n in scene.names] == ["E_Yam", "E_Yam"]
    colors = {scene[n]._bone_color for n in scene.names}
    assert len(colors) == len(scene.names)


def test_from_rig_arm_names_are_the_rig_s_own(make_scene) -> None:
    """The CLI, the session, and the scene must share one vocabulary for an arm."""
    scene = make_scene(resolve_rig("yam_bimanual"))

    scene.update("left", [0.1] * 6)
    assert scene["left"].positions[0] == pytest.approx(0.1)
    with pytest.raises(ConfigurationError, match="unknown arm"):
        scene.update("left_follower", [0.0] * 6)


def test_from_rig_takes_an_explicit_label_over_the_rig_name(make_scene) -> None:
    scene = make_scene(resolve_rig("yam_bimanual"), label="bench")
    assert scene.label == "bench"


def test_scene_update_all_drives_both_arms_independently(make_scene) -> None:
    scene = make_scene(resolve_rig("yam_bimanual"))

    scene.update_all({"left": [0.2] * 6, "right": [-0.2] * 6})

    assert scene["left"].positions[0] == pytest.approx(0.2)
    assert scene["right"].positions[0] == pytest.approx(-0.2)


def test_cli_list_rigs_describes_each_rig_and_its_arms(capsys) -> None:
    assert viz.main(["--list-rigs"]) == 0
    out = capsys.readouterr().out
    for name in rig_names():
        rig = resolve_rig(name)
        assert name in out
        for arm in rig.arms:
            assert arm.name in out
            assert arm.interface in out


def test_cli_refuses_rig_and_model_together() -> None:
    # --rig takes its models from the rig; silently ignoring --model would be
    # the kind of quiet surprise that costs an hour.
    with pytest.raises(SystemExit):
        viz.main(["--rig", "yam_bimanual", "--model", "Yam"])


def test_live_drives_the_rig_scene_from_the_arms(monkeypatch) -> None:
    """cli.run_live's viser path: the seam test_cli covers only with visualize=False.

    Uses the CLI's own fake backends, so no bus is opened and no node is
    started -- this checks the wiring between run_live and from_rig.
    """
    import threading

    from fake_arm_backend import FakeArmBackend

    from openpi_control import cli

    rig = resolve_rig("yam_bimanual")
    made: dict[str, object] = {}

    def backend_factory(rig_arm):
        backend = FakeArmBackend()
        backend.position = 0.3 if rig_arm.name == "left" else -0.3
        made[rig_arm.name] = backend
        return backend

    scenes = []
    real_from_rig = viz.ArmSceneVisualizer.from_rig

    def capturing_from_rig(cls_rig, **kwargs):
        kwargs["port"] = next(_PORT)
        scene = real_from_rig(cls_rig, **kwargs)
        scenes.append(scene)
        return scene

    monkeypatch.setattr(viz.ArmSceneVisualizer, "from_rig", capturing_from_rig)

    stop = threading.Event()
    stop.set()  # one pass through the mirror loop, then unwind

    assert cli.run_live(rig, visualize=True, backend_factory=backend_factory, stop=stop) == 0

    assert len(scenes) == 1
    assert scenes[0].names == rig.names
    # Parked on the way down, and the scene's server was stopped with it.
    for backend in made.values():
        assert backend.closes[0] is True
        assert not backend.connected
