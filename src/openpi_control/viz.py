"""Browser visualization of the packaged arm models, served by viser.

This module is deliberately hardware-free: it resolves the same packaged URDF
and model JSON that :mod:`openpi_control.config` hands to the native node, and
renders them over HTTP. It never opens a CAN, serial, Ethernet, or Modbus
connection and never starts a ``pi_control_node``, so it is safe to run with
the buses down or no arm attached.

Feed a live pose in with :meth:`ArmVisualizer.update`; the caller owns the
hardware session, this module only draws.

    from openpi_control.viz import ArmVisualizer

    viz = ArmVisualizer("Yam", effector_model="E_Yam")
    viz.update(follower.read_state().joints.position_rad)

:class:`ArmSceneVisualizer` puts several arms in one scene, each with its own
base pose -- a bimanual cell, or a leader/follower pair::

    from openpi_control.viz import ArmSceneVisualizer, ArmSpec

    scene = ArmSceneVisualizer({
        "left":  ArmSpec("Yam", effector_model="E_Yam"),
        "right": ArmSpec("Yam", effector_model="E_Yam",
                         base_position=(0.0, -0.61, 0.0)),
    })
    scene.update("left", left.read_state().joints.position_rad)

:class:`CameraPanel` puts the rig's cameras on the same page, one tile each. It
takes readers that are already open and never opens one itself, so this module
still holds no device and still needs no RealSense SDK to import::

    from openpi_control.viz import CameraPanel

    panel = CameraPanel(scene.server, readers)
    panel.step(dt)          # on the caller's clock; throttles itself internally

Run ``python -m openpi_control.viz --model Yam`` for a standalone viewer whose
GUI sliders drive the joints.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from . import meshes
from .config import SUPPORTED_EFFECTORS, SUPPORTED_MODELS, resolve_model_assets
from .exceptions import ConfigurationError
from .rigs import Rig, resolve_rig, rig_names

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

_EXTRA_HINT = (
    "viser visualization needs the optional 'viz' extra: "
    "uv sync --extra viz  (or: pip install 'openpi-control[viz]')"
)

# Revolute joints without an explicit limit still need a slider range.
_DEFAULT_JOINT_RANGE_RAD = np.pi

# Rendered skeleton proportions, in metres. Sized for a tabletop 6-DoF arm.
_BONE_RADIUS_M = 0.011
_AXIS_MARKER_RADIUS_M = 0.004
_AXIS_MARKER_LENGTH_M = 0.045

_BONE_COLOR = (110, 125, 145)
_AXIS_MARKER_COLOR = (250, 165, 40)

# Browser camera preview. A preview answers "where is that wrist pointing" and
# "is the light any good" -- it is not a recording, so it wants to be legible
# and cheap rather than faithful. Three 848x480 streams pushed whole at the
# 30 Hz mirror rate is ~35 MB/s of websocket for a question a 400px thumbnail
# at 10 Hz answers just as well.
_PREVIEW_MAX_WIDTH = 400
_PREVIEW_RATE_HZ = 10.0
_PREVIEW_JPEG_QUALITY = 70

# What a preview tile shows before its camera's first frame lands, so the panel
# has its final layout from the moment the page opens rather than growing a
# tile per camera over the first second.
_PREVIEW_PLACEHOLDER = (32, 34, 38)


# Per-arm tints, so the arms of a bimanual scene stay tellable apart.
_ARM_COLORS = (
    (92, 132, 186),
    (196, 128, 66),
    (118, 168, 118),
    (166, 118, 176),
)


@dataclass(frozen=True, slots=True)
class VizModules:
    """The lazily imported optional visualization dependencies."""

    viser: ModuleType
    extras: ModuleType
    transforms: ModuleType
    trimesh: ModuleType
    yourdfpy: ModuleType


def import_viz_modules() -> VizModules:
    """Import viser/yourdfpy, raising a ConfigurationError naming the extra."""
    try:
        import trimesh
        import viser
        import viser.extras
        import viser.transforms as vtf
        import yourdfpy
    except ImportError as err:
        raise ConfigurationError(f"{_EXTRA_HINT} (missing: {err.name})") from err
    return VizModules(viser, viser.extras, vtf, trimesh, yourdfpy)


@dataclass(frozen=True, slots=True)
class JointSpec:
    """One actuated URDF joint, in kinematic-chain order."""

    name: str
    lower: float
    upper: float

    @property
    def rest(self) -> float:
        """A pose inside the limits: zero when reachable, else the midpoint."""
        if self.lower <= 0.0 <= self.upper:
            return 0.0
        return 0.5 * (self.lower + self.upper)


def chain_ordered_actuated_joints(urdf: Any) -> tuple[str, ...]:
    """Actuated joint names ordered by depth from the base link.

    ``yourdfpy.URDF.actuated_joint_names`` follows URDF document order, which
    packaged models do not guarantee (``Yam.urdf`` declares joint6 first). Arm
    state and :class:`~openpi_control.types.PositionCommand` are indexed along
    the chain instead, so walk the tree to recover that order.
    """
    children: dict[str, list[Any]] = {}
    for joint in urdf.robot.joints:
        children.setdefault(joint.parent, []).append(joint)

    ordered: list[str] = []
    seen: set[str] = set()
    stack = [urdf.base_link]
    while stack:
        link = stack.pop()
        if link in seen:
            continue
        seen.add(link)
        # Reversed so the document order of siblings survives the LIFO pop.
        for joint in reversed(children.get(link, [])):
            if joint.type not in ("fixed", "floating"):
                ordered.append(joint.name)
            stack.append(joint.child)

    actuated = set(urdf.actuated_joint_names)
    chain = tuple(name for name in ordered if name in actuated)
    missing = actuated.difference(chain)
    if missing:
        raise ConfigurationError(
            "URDF actuated joints unreachable from the base link: " + ", ".join(sorted(missing))
        )
    return chain


class ArmVisualizer:
    """A viser scene for one packaged arm model.

    Renders the arm's visual meshes when they are available, and otherwise a
    kinematic skeleton built from the joint origins. The packaged URDFs ship
    without their ``assets/*.stl``, so ``skeleton`` is the usual mode; point
    ``mesh_dir`` at a directory holding the meshes to get the full model.
    """

    def __init__(
        self,
        model: str,
        *,
        name: str | None = None,
        effector_model: str | None = None,
        instance_config: Path | None = None,
        urdf: Path | None = None,
        mesh_dir: Path | None = None,
        server: Any | None = None,
        port: int = 8080,
        root_node_name: str | None = None,
        base_position: Sequence[float] = (0.0, 0.0, 0.0),
        base_rotation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        color: Sequence[int] | None = None,
        show_grid: bool = True,
        show_label: bool = False,
        up_axis: str | None = "+z",
    ) -> None:
        self._mods = import_viz_modules()
        self.model = model
        # The logical arm name, e.g. "left_follower" — matches ArmConfig.name.
        self.name = name or model
        self.effector_model = effector_model
        # An explicit color tints both the bones and the meshes, so a bimanual
        # scene stays readable. Left unset, meshes keep their URDF materials.
        self._color_override = tuple(int(c) for c in color) if color is not None else None
        self._bone_color = self._color_override or _BONE_COLOR

        self.assets = resolve_model_assets(
            model,
            effector_model=effector_model,
            instance_config=instance_config,
            urdf=urdf,
        )
        if self.assets.urdf is None:
            raise ConfigurationError(
                f"model {model!r} ships no URDF (FR3 uses the vendor controller's own model); "
                "pass urdf=<path> to visualize it"
            )
        self.urdf_path = self.assets.urdf
        self.mesh_dir = self._resolve_mesh_dir(mesh_dir)

        self._urdf, self.render_mode = self._load_urdf()
        self._joints = self._read_joint_specs()
        self._positions = np.array([spec.rest for spec in self._joints], dtype=np.float64)

        self.server = server if server is not None else self._mods.viser.ViserServer(port=port)
        self._root = root_node_name or f"/{self.name}"
        self._link_frames: dict[str, Any] = {}
        self._bones: list[Any] = []
        self._axis_markers: list[Any] = []
        self._urdf_vis: Any = None
        self._vis_permutation: np.ndarray | None = None
        self.missing_meshes: tuple[str, ...] = ()
        self._sliders: list[Any] = []

        # A shared server sets these once for the whole scene, so skip them when
        # this arm is one of several.
        if up_axis is not None:
            self.server.scene.set_up_direction(up_axis)
        self.base_frame = self.server.scene.add_frame(
            self._root,
            show_axes=False,
            position=np.asarray(base_position, dtype=np.float32),
            wxyz=np.asarray(base_rotation_wxyz, dtype=np.float32),
        )
        if show_grid:
            self.server.scene.add_grid(
                "/ground", width=1.5, height=1.5, cell_size=0.05, section_size=0.25
            )
        if show_label:
            self.server.scene.add_label(f"{self._root}/label", self.name)
        self._build_scene()
        self.update(self._positions)

    # ------------------------------------------------------------------ #
    # Asset resolution and loading
    # ------------------------------------------------------------------ #

    def _resolve_mesh_dir(self, mesh_dir: Path | None) -> Path | None:
        """Where this model's meshes live, or None to render a skeleton.

        Checked in order: an explicit directory, a sibling ``assets/`` beside the
        URDF, then the local mesh cache that ``--fetch-meshes`` fills. The wheel
        ships no meshes, so the cache is what makes the real arm the default.
        """
        if mesh_dir is not None:
            resolved = Path(mesh_dir).expanduser().resolve()
            if not resolved.is_dir():
                raise ConfigurationError(f"mesh_dir {resolved} is not a directory")
            return resolved
        sibling = self.urdf_path.parent / "assets"
        if sibling.is_dir():
            return sibling
        return meshes.cached_mesh_dir(self.model)

    def _load_urdf(self) -> tuple[Any, str]:
        """Load with meshes when a mesh dir resolved, else load the tree only."""
        yourdfpy = self._mods.yourdfpy
        if self.mesh_dir is not None:
            try:
                loaded = yourdfpy.URDF.load(str(self.urdf_path), mesh_dir=str(self.mesh_dir))
            except Exception as err:  # noqa: BLE001 - trimesh/lxml raise many types
                raise ConfigurationError(
                    f"failed to load {self.urdf_path} with meshes from {self.mesh_dir}: {err}"
                ) from err
            return loaded, "mesh"
        loaded = yourdfpy.URDF.load(
            str(self.urdf_path), load_meshes=False, build_collision_scene_graph=False
        )
        return loaded, "skeleton"

    def _read_joint_specs(self) -> tuple[JointSpec, ...]:
        specs = []
        for name in chain_ordered_actuated_joints(self._urdf):
            joint = self._urdf.joint_map[name]
            limit = getattr(joint, "limit", None)
            lower = getattr(limit, "lower", None)
            upper = getattr(limit, "upper", None)
            if lower is None or upper is None or upper <= lower:
                lower, upper = -_DEFAULT_JOINT_RANGE_RAD, _DEFAULT_JOINT_RANGE_RAD
            specs.append(JointSpec(name, float(lower), float(upper)))
        if not specs:
            raise ConfigurationError(f"{self.urdf_path} declares no actuated joints")
        return tuple(specs)

    # ------------------------------------------------------------------ #
    # Scene construction
    # ------------------------------------------------------------------ #

    def _build_scene(self) -> None:
        if self.render_mode == "mesh":
            override = (
                None
                if self._color_override is None
                else tuple(c / 255.0 for c in self._color_override)
            )
            self._urdf_vis = self._mods.extras.ViserUrdf(
                self.server, self._urdf, root_node_name=self._root, mesh_color_override=override
            )
            self.missing_meshes = self._find_missing_meshes()
            # ViserUrdf.update_cfg indexes by its own actuated order, not ours.
            vis_order = list(self._urdf_vis.get_actuated_joint_names())
            self._vis_permutation = np.array(
                [vis_order.index(spec.name) for spec in self._joints], dtype=np.intp
            )
            return
        self._build_skeleton()

    def _find_missing_meshes(self) -> tuple[str, ...]:
        """Mesh files the URDF references that the mesh dir does not hold.

        i2rt's YAM assets omit link_6_visual/collision.stl, so the wrist link
        renders bare. Worth surfacing rather than leaving as a silent gap.
        """
        if self.mesh_dir is None:
            return ()
        try:
            names = meshes.urdf_mesh_names(self.model, urdf=self.urdf_path)
        except ConfigurationError:
            return ()
        return tuple(name for name in names if not (self.mesh_dir / name).is_file())

    def _build_skeleton(self) -> None:
        """Link frames plus static bone meshes parented under them.

        Each bone spans a parent link's origin to its child's, which is fixed in
        the parent's frame, so :meth:`update` only has to move the frames.
        """
        scene = self.server.scene
        trimesh = self._mods.trimesh
        actuated = {spec.name for spec in self._joints}

        for link in self._urdf.link_map:
            self._link_frames[link] = scene.add_frame(
                self._link_node(link), show_axes=True, axes_length=0.04, axes_radius=0.0025
            )

        for joint in self._urdf.robot.joints:
            # yourdfpy leaves origin/axis as None when the URDF omits the tag;
            # the URDF spec defaults them to identity and +x respectively.
            origin = np.eye(4) if joint.origin is None else np.asarray(joint.origin, np.float64)
            offset = origin[:3, 3]
            if float(np.linalg.norm(offset)) > 1e-5:
                bone = trimesh.creation.cylinder(
                    radius=_BONE_RADIUS_M, segment=np.array([np.zeros(3), offset]), sections=14
                )
                bone.visual.face_colors = self._bone_color
                self._bones.append(
                    # A child of the parent link's frame, which supplies the pose.
                    scene.add_mesh_trimesh(
                        f"{self._link_node(joint.parent)}/bone_{joint.name}", bone
                    )
                )
            if joint.name not in actuated:
                continue
            axis = np.asarray(
                (1.0, 0.0, 0.0) if joint.axis is None else joint.axis, dtype=np.float64
            )
            norm = float(np.linalg.norm(axis))
            if norm < 1e-9:
                continue
            marker = trimesh.creation.cylinder(
                radius=_AXIS_MARKER_RADIUS_M,
                segment=np.array(
                    [
                        -0.5 * _AXIS_MARKER_LENGTH_M * axis / norm,
                        0.5 * _AXIS_MARKER_LENGTH_M * axis / norm,
                    ]
                ),
                sections=10,
            )
            marker.visual.face_colors = _AXIS_MARKER_COLOR
            self._axis_markers.append(
                scene.add_mesh_trimesh(
                    f"{self._link_node(joint.child)}/axis_{joint.name}", marker
                )
            )

    def _link_node(self, link: str) -> str:
        return f"{self._root}/links/{link}"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def joint_specs(self) -> tuple[JointSpec, ...]:
        """Actuated joints in kinematic-chain order — the arm's joint index order."""
        return self._joints

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._joints)

    @property
    def positions(self) -> np.ndarray:
        """The last rendered joint positions, in radians, chain order."""
        return self._positions.copy()

    @property
    def url(self) -> str:
        return f"http://localhost:{self.server.get_port()}"

    def update(self, positions: Sequence[float] | Mapping[str, float] | np.ndarray) -> None:
        """Render a joint pose given in radians.

        Accepts a sequence in chain order (matching
        :class:`~openpi_control.types.PositionCommand`) or a name-keyed mapping.
        Values are clamped to the URDF limits so a stale or out-of-range sample
        cannot fold the render through a joint stop.
        """
        if hasattr(positions, "keys"):
            unknown = set(positions.keys()).difference(self.joint_names)  # type: ignore[union-attr]
            if unknown:
                raise ConfigurationError(
                    f"unknown joint names for {self.model}: {', '.join(sorted(unknown))}"
                )
            values = np.array(
                [
                    float(positions.get(spec.name, current))  # type: ignore[union-attr]
                    for spec, current in zip(self._joints, self._positions, strict=True)
                ],
                dtype=np.float64,
            )
        else:
            values = np.asarray(positions, dtype=np.float64).reshape(-1)
            if values.size != len(self._joints):
                raise ConfigurationError(
                    f"{self.model} expects {len(self._joints)} joint positions, "
                    f"got {values.size}"
                )
        if not np.all(np.isfinite(values)):
            raise ConfigurationError("joint positions must all be finite")

        lower = np.array([spec.lower for spec in self._joints])
        upper = np.array([spec.upper for spec in self._joints])
        self._positions = np.clip(values, lower, upper)

        if self.render_mode == "mesh":
            self._urdf_vis.update_cfg(self._positions[self._vis_permutation])
            return
        self._urdf.update_cfg(
            dict(zip(self.joint_names, self._positions.tolist(), strict=True))
        )
        self._refresh_link_frames()

    def _refresh_link_frames(self) -> None:
        so3 = self._mods.transforms.SO3
        base = self._urdf.base_link
        for link, frame in self._link_frames.items():
            transform = self._urdf.get_transform(link, base, collision_geometry=False)
            frame.wxyz = so3.from_matrix(np.asarray(transform)[:3, :3]).wxyz
            frame.position = np.asarray(transform)[:3, 3]

    def mesh_status(self) -> str:
        """One line on where the geometry came from, and what is missing."""
        if self.render_mode == "skeleton":
            return (
                f"  meshes: none — drawing a kinematic skeleton. Run "
                f"--fetch-meshes --model {self.model} to render the real arm."
            )
        line = f"  meshes: {self.mesh_dir}"
        if self.missing_meshes:
            line += (
                f"\n  missing upstream ({len(self.missing_meshes)}): "
                + ", ".join(self.missing_meshes)
            )
        return line

    def add_gui(self, *, folder: str | None = None) -> None:
        """Add this arm's model summary and joint sliders.

        Each arm gets its own top-level folder so several can share one server.
        """
        gui = self.server.gui
        with gui.add_folder(folder or self.name):
            gui.add_text("Model", initial_value=self.model, disabled=True)
            gui.add_text("Effector", initial_value=self.effector_model or "none", disabled=True)
            gui.add_text("Render", initial_value=self.render_mode, disabled=True)
            if self.missing_meshes:
                gui.add_text(
                    "Missing meshes",
                    initial_value=", ".join(self.missing_meshes),
                    disabled=True,
                )
            for index, spec in enumerate(self._joints):
                slider = gui.add_slider(
                    spec.name,
                    min=spec.lower,
                    max=spec.upper,
                    step=0.005,
                    initial_value=float(self._positions[index]),
                )
                slider.on_update(lambda _: self._apply_sliders())
                self._sliders.append(slider)
            reset = gui.add_button("Reset to rest")

            @reset.on_click
            def _(_: Any) -> None:
                self.reset()

    def reset(self) -> None:
        """Return to the rest pose, moving the sliders with it."""
        if self._sliders:
            for slider, spec in zip(self._sliders, self._joints, strict=True):
                slider.value = spec.rest
            return
        self.update([spec.rest for spec in self._joints])

    def _apply_sliders(self) -> None:
        self.update([float(slider.value) for slider in self._sliders])

    def run(self) -> None:
        """Serve until interrupted. Sliders drive the pose; no hardware is touched."""
        print(f"{self.name} ({self.render_mode}) on {self.url} — ctrl-c to stop")
        print(self.mesh_status())
        _serve_until_interrupted(self.server)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One arm's model selection and where its base sits in the scene frame.

    Usually built for you from a packaged rig by
    :meth:`ArmSceneVisualizer.from_rig`; construct it directly to compose a
    scene the packaged rigs do not cover.
    """

    model: str
    effector_model: str | None = None
    instance_config: Path | None = None
    urdf: Path | None = None
    mesh_dir: Path | None = None
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    color: tuple[int, int, int] | None = None


class ArmSceneVisualizer:
    """Several arms in one viser scene, each with its own base pose.

    Each arm keeps its own :class:`ArmVisualizer`, so per-arm joint order,
    limits, and render mode work exactly as in the single-arm case. Like
    :class:`ArmVisualizer` this opens no bus and starts no native node.

        scene = ArmSceneVisualizer({
            "left":  ArmSpec("Yam"),
            "right": ArmSpec("Yam", base_position=(0.0, -0.61, 0.0)),
        })
        scene.update("left", left_follower.read_state().joints.position_rad)
    """

    def __init__(
        self,
        arms: Mapping[str, ArmSpec],
        *,
        label: str | None = None,
        server: Any | None = None,
        port: int = 8080,
        show_grid: bool = True,
        show_labels: bool = True,
        up_axis: str = "+z",
        grid_size_m: float | None = None,
        initial_camera_position: Sequence[float] = (-1.1, -0.9, 0.8),
        initial_camera_look_at: Sequence[float] = (0.25, 0.0, 0.2),
    ) -> None:
        if not arms:
            raise ConfigurationError("an arm scene needs at least one arm")
        self.label = label
        mods = import_viz_modules()
        self.server = server if server is not None else mods.viser.ViserServer(port=port)
        self.server.scene.set_up_direction(up_axis)

        self.arms: dict[str, ArmVisualizer] = {}
        for index, (name, spec) in enumerate(arms.items()):
            self.arms[name] = ArmVisualizer(
                spec.model,
                name=name,
                effector_model=spec.effector_model,
                instance_config=spec.instance_config,
                urdf=spec.urdf,
                mesh_dir=spec.mesh_dir,
                server=self.server,
                base_position=spec.base_position,
                base_rotation_wxyz=spec.base_rotation_wxyz,
                color=spec.color or _ARM_COLORS[index % len(_ARM_COLORS)],
                show_grid=False,
                show_label=show_labels,
                up_axis=None,
            )

        if show_grid:
            span = grid_size_m if grid_size_m is not None else self._grid_span()
            self.server.scene.add_grid(
                "/ground", width=span, height=span, cell_size=0.05, section_size=0.25
            )

        position = np.asarray(initial_camera_position, dtype=np.float32)
        look_at = np.asarray(initial_camera_look_at, dtype=np.float32)

        @self.server.on_client_connect
        def _(client: Any) -> None:
            client.camera.position = position
            client.camera.look_at = look_at

    @classmethod
    def from_rig(
        cls, rig: Rig, *, mesh_dir: Path | None = None, **kwargs: Any
    ) -> ArmSceneVisualizer:
        """Build a scene from a packaged rig, honouring each arm's base pose.

        The rig owns the arm names, so ``scene.update("left", ...)`` takes the
        same name the operator CLI and the session use for that arm -- one
        vocabulary across the three, rather than a scene key that has to be
        mapped back to a bus by hand.
        """
        specs = {
            arm.name: ArmSpec(
                model=arm.model,
                effector_model=arm.effector_model,
                instance_config=arm.instance_config,
                mesh_dir=mesh_dir,
                base_position=arm.base_position,
                base_rotation_wxyz=arm.base_rotation_wxyz,
            )
            for arm in rig.arms
        }
        kwargs.setdefault("label", rig.name)
        return cls(specs, **kwargs)

    def _grid_span(self) -> float:
        """A grid wide enough to sit under every arm base, with a margin."""
        reach = max(
            float(np.linalg.norm(np.asarray(arm.base_frame.position)[:2]))
            for arm in self.arms.values()
        )
        return max(1.5, 2.0 * (reach + 0.6))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.arms)

    @property
    def url(self) -> str:
        return f"http://localhost:{self.server.get_port()}"

    def __getitem__(self, name: str) -> ArmVisualizer:
        try:
            return self.arms[name]
        except KeyError:
            raise ConfigurationError(
                f"unknown arm {name!r}; the scene holds: {', '.join(self.names)}"
            ) from None

    def update(
        self, name: str, positions: Sequence[float] | Mapping[str, float] | np.ndarray
    ) -> None:
        """Render one arm's pose. See :meth:`ArmVisualizer.update`."""
        self[name].update(positions)

    def update_all(
        self, poses: Mapping[str, Sequence[float] | Mapping[str, float] | np.ndarray]
    ) -> None:
        """Render several arms at once. Arms left out keep their last pose."""
        for name, positions in poses.items():
            self.update(name, positions)

    def add_gui(self) -> None:
        """One folder of sliders per arm, plus a scene-wide reset."""
        for name, arm in self.arms.items():
            arm.add_gui(folder=f"{name} ({arm.model})")
        reset = self.server.gui.add_button("Reset all arms")

        @reset.on_click
        def _(_: Any) -> None:
            for arm in self.arms.values():
                arm.reset()

    def run(self) -> None:
        """Serve until interrupted. No hardware is touched."""
        summary = ", ".join(f"{name}={arm.model}" for name, arm in self.arms.items())
        label = self.label or f"{len(self.arms)} arms"
        print(f"{label} ({summary}) on {self.url} — ctrl-c to stop")
        _serve_until_interrupted(self.server)


class CameraPanel:
    """Live camera thumbnails in the browser, one tile per camera.

    This takes readers that are *already open* and never opens one itself,
    which is what keeps this module honest: :mod:`openpi_control.viz` draws and
    holds no device, so it stays importable and testable on a box with no
    RealSense SDK, exactly as it stays usable against a dead cell. A reader is
    anything with a ``spec`` and a ``latest()`` -- see
    :class:`openpi_control.cameras.CameraReader`.

    Frames are pushed on the caller's clock, the same clock that pumps the
    poses, so a preview cannot drift into being its own thread fighting the
    render for the websocket. It is throttled to ``rate_hz`` and shrunk to
    ``max_width`` independently of that clock, because the mirror runs at 30 Hz
    and a preview has no reason to::

        panel = CameraPanel(scene.server, readers)
        while running:
            panel.step(dt)

    A camera that stops delivering stops being pushed: tiles are keyed off the
    reader's frame count, so a dead stream holds its last image instead of
    re-encoding it ten times a second.
    """

    def __init__(
        self,
        server: Any,
        readers: Mapping[str, Any],
        *,
        folder: str = "Cameras",
        max_width: int = _PREVIEW_MAX_WIDTH,
        rate_hz: float = _PREVIEW_RATE_HZ,
        jpeg_quality: int = _PREVIEW_JPEG_QUALITY,
    ) -> None:
        if max_width <= 0:
            raise ConfigurationError("a camera preview needs a positive max_width")
        if rate_hz <= 0:
            raise ConfigurationError("a camera preview needs a positive rate_hz")
        self.server = server
        self._readers = dict(readers)
        self._max_width = max_width
        self._period = 1.0 / rate_hz
        self._jpeg_quality = jpeg_quality
        self._since_push = self._period  # push on the first step, not a tick late
        self._pushed: dict[str, int] = {}
        self._tiles: dict[str, Any] = {}

        placeholder = np.full((3, 4, 3), _PREVIEW_PLACEHOLDER, dtype=np.uint8)
        with server.gui.add_folder(folder):
            for name, reader in self._readers.items():
                self._tiles[name] = server.gui.add_image(
                    placeholder,
                    label=self._label(name, reader),
                    format="jpeg",
                    jpeg_quality=jpeg_quality,
                )

    @staticmethod
    def _label(name: str, reader: Any) -> str:
        """``name`` first: it is the word the rig, the CLI, and datasets use."""
        description = getattr(getattr(reader, "spec", None), "label", None)
        return f"{name} — {description}" if description else name

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._readers)

    def step(self, dt: float) -> None:
        """Push each camera's newest frame, at most once per preview period."""
        self._since_push += dt
        if self._since_push < self._period:
            return
        self._since_push = 0.0
        for name, reader in self._readers.items():
            frame = reader.latest()
            if frame is None:
                continue
            # frames_read is the cheap "is this the same picture again" test;
            # a reader that does not keep one is simply pushed every period.
            count = getattr(reader, "frames_read", None)
            if count is not None:
                if self._pushed.get(name) == count:
                    continue
                self._pushed[name] = count
            self._tiles[name].image = _thumbnail(
                frame, self._max_width, pixel_format=getattr(reader, "pixel_format", "bgr8")
            )

    def remove(self) -> None:
        """Drop every tile from the GUI. The readers are the caller's to close."""
        for tile in self._tiles.values():
            tile.remove()
        self._tiles.clear()


def _thumbnail(frame: np.ndarray, max_width: int, *, pixel_format: str = "bgr8") -> np.ndarray:
    """A small RGB copy of one capture frame.

    Subsampling by stride rather than resampling is what lets a preview cost one
    numpy copy and no OpenCV: capture already refuses to depend on cv2, and a
    thumbnail whose only job is to show aim and lighting gains nothing from
    interpolation.

    The channel order is asked for, not assumed -- a reader opened as ``rgb8``
    (which is what the dataset recorder does) would come out with red and blue
    swapped if this flipped unconditionally.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ConfigurationError(
            f"a camera preview wants an HxWx3 frame, got shape {frame.shape}"
        )
    stride = max(1, -(-frame.shape[1] // max_width))
    step = 1 if pixel_format == "rgb8" else -1
    return np.ascontiguousarray(frame[::stride, ::stride, ::step])


def _serve_until_interrupted(server: Any) -> None:
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m openpi_control.viz",
        description="Serve a packaged arm model in the browser with viser. Opens no bus.",
    )
    parser.add_argument(
        "--model", default=None, help=f"one of: {', '.join(SUPPORTED_MODELS)} (default Yam)"
    )
    parser.add_argument(
        "--rig",
        default=None,
        help=(
            f"draw a whole packaged rig instead of one arm: {', '.join(rig_names())}. "
            "Still opens no bus -- the sliders drive the render."
        ),
    )
    parser.add_argument(
        "--effector", default=None, help=f"one of: {', '.join(SUPPORTED_EFFECTORS)}"
    )
    parser.add_argument("--instance-config", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=None, help="override the packaged URDF")
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=None,
        help="directory holding the URDF's meshes; enables full mesh rendering",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument(
        "--list", action="store_true", help="list the models that ship a URDF and exit"
    )
    parser.add_argument(
        "--list-rigs", action="store_true", help="list the packaged rigs and exit"
    )
    parser.add_argument(
        "--fetch-meshes",
        action="store_true",
        help=(
            "download --model's visual meshes into ~/openpi-data/meshes and exit. "
            "Needs the network once; later runs render the real arm offline."
        ),
    )
    args = parser.parse_args(argv)
    if args.rig is not None and args.model is not None:
        parser.error("--rig draws a whole rig; it takes its models from the rig, not --model")
    model = args.model or "Yam"

    if args.list_rigs:
        for name in rig_names():
            rig = resolve_rig(name)
            print(f"{name:16s} {rig.description}")
            for arm in rig.arms:
                effector = arm.effector_model or "no effector"
                print(f"  {arm.name:8s} {arm.model:6s} {arm.interface:8s} {effector}")
        return 0

    if args.list:
        for model in SUPPORTED_MODELS:
            try:
                assets = resolve_model_assets(model)
            except ConfigurationError as err:
                print(f"{model:18s} unavailable: {err}")
                continue
            urdf = assets.urdf.name if assets.urdf is not None else "no URDF"
            print(f"{model:18s} {urdf}")
        return 0

    if args.fetch_meshes:
        try:
            print(meshes.fetch_meshes(model, urdf=args.urdf).summary())
        except ConfigurationError as err:
            parser.error(str(err))
        return 0

    scene: ArmVisualizer | ArmSceneVisualizer
    try:
        if args.rig is not None:
            scene = ArmSceneVisualizer.from_rig(
                resolve_rig(args.rig),
                mesh_dir=args.mesh_dir,
                port=args.port,
                show_grid=not args.no_grid,
            )
        else:
            scene = ArmVisualizer(
                model,
                effector_model=args.effector,
                instance_config=args.instance_config,
                urdf=args.urdf,
                mesh_dir=args.mesh_dir,
                port=args.port,
                show_grid=not args.no_grid,
            )
    except ConfigurationError as err:
        parser.error(str(err))
    scene.add_gui()
    scene.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
