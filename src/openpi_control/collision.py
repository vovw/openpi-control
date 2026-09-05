"""Collision geometry for the packaged arms, and the gate that acts on it.

A bimanual cell has two arms that can reach the same volume, and nothing
between them but the operator's attention. This module is the geometry half of
closing that gap: it turns each packaged URDF's *collision* meshes into a
sphere cover, runs forward kinematics on a candidate joint pose, and reports
the smallest clearance in the cell.

It is hardware-free in the same sense :mod:`openpi_control.viz` is -- it reads
the packaged URDF and the cached meshes and nothing else -- and it is
*viser*-free too, so a collision model can be built and tested headless::

    from openpi_control.collision import CellCollisionModel
    from openpi_control.rigs import resolve_rig

    model = CellCollisionModel.from_rig(resolve_rig("yam_bimanual"))
    report = model.check({"left": left_q, "right": right_q})
    print(report.clearance_m, report.contacts)

:mod:`openpi_control.viz` draws what this reports (``CollisionOverlay``) and
:mod:`openpi_control.viser_control` refuses commands on it (:class:`CollisionGate`).
Neither owns the model: it is built once and handed to both, so the picture on
the page and the rule the gate applies are the same geometry.

Why spheres
-----------
Each link is covered by a small set of spheres fitted to its collision mesh, so
the whole cell reduces to one ``(N, 3)`` array of centres and one ``(N,)`` array
of radii, and a check is a single broadcast distance matrix. Measured on the
packaged ``yam_bimanual`` (two YAM arms, 7 links each, 8 spheres per link):
112 spheres, and FK plus the full 112x112 clearance matrix costs well under a
millisecond -- affordable twice per tick inside the 33 ms budget at the 30 Hz
mirror rate.

The alternative -- one enclosing capsule per link -- is simpler still but much
looser. Measured against each YAM link's convex hull by voxel occupancy, a
single capsule claims 2.1x to 9.8x the hull's volume where an 8-sphere cover
claims 1.3x to 2.1x. That padding is not free: it is reported as lost clearance
and it is what makes a gate cry wolf.

The cover is *conservative* by construction -- every point of every collision
triangle is inside some sphere -- so a real contact is never missed, only
predicted early. See :func:`fit_sphere_cover` for how that is guaranteed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from .exceptions import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence


_EXTRA_HINT = (
    "collision checking needs the optional 'viz' extra: "
    "uv sync --extra viz  (or: pip install 'openpi-control[viz]')"
)

# How many spheres cover one link. Eight is where the tightening stops paying
# for itself on the packaged YAM: measured by voxel occupancy against each
# link's convex hull, 4 spheres claim 1.7-3.9x the hull, 8 claim 1.3-2.1x, and
# 16 claim 1.0-1.8x for double the pair count. Padding is reported as lost
# clearance, so it is worth buying down -- but not at 4x the check cost.
_SPHERES_PER_LINK = 8

# The cover is fitted to mesh *vertices*, which says nothing about the middle
# of a large triangle: the packaged YAM collision meshes reach a 190 mm edge
# where a flat face is triangulated coarsely. So each mesh is first subdivided
# until no edge exceeds this length, and every sphere is then grown by
# _SURFACE_PADDING_M below.
_MESH_SUBDIVISION_M = 0.010

# Half-covers a subdivided triangle: the farthest a point inside an equilateral
# triangle of side L can be from its nearest vertex is L/sqrt(3). With the
# subdivision above that is 5.8 mm, rounded up here. Applied to every sphere,
# this is what turns "covers the vertices" into "covers the surface".
_SURFACE_PADDING_M = 0.006

# The mesh-free fallback. A packaged wheel ships URDFs and no meshes, so a
# fresh checkout has only the kinematic tree to work from -- the same tree
# viz.py draws as a skeleton. Links become sphere chains of this radius along
# their bones. It is a nominal tabletop-arm link radius, not a measurement of
# any arm: the YAM's own mesh-fitted spheres run 0.02-0.06 m. Erring fat means
# false stops rather than missed contacts, which is the right direction, but a
# model built this way says so in `geometry_source` and the GUI shows it.
_SKELETON_SPHERE_RADIUS_M = 0.04

# Random poses used to find link pairs that are in contact in *every* pose --
# geometry that overlaps by construction, which would otherwise alarm forever.
# See ArmCollisionModel._self_pair_mask. Few are needed: one non-touching
# sample is enough to keep a pair enabled, so this is not the large sample a
# "never touches, stop checking it" rule would need.
_AUTO_DISABLE_SAMPLES = 200
_AUTO_DISABLE_SEED = 20260902

# Clearance the gate stops at, in metres. Browser control walks a joint at
# 0.5 rad/s (viser_control.ControlLimits), which moves a YAM flange about
# 0.25 m/s at full reach -- 8 mm in one 33 ms tick. Stopping 20 mm out leaves
# roughly two ticks of headroom over the tick that discovers the problem.
_DEFAULT_STOP_CLEARANCE_M = 0.02
# Where the page turns amber. Far enough out to be a warning rather than a
# report of the stop that is about to happen.
_DEFAULT_WARN_CLEARANCE_M = 0.05

GROUND = "ground"


@dataclass(frozen=True, slots=True)
class GeometryModules:
    """The lazily imported geometry dependencies.

    Deliberately *not* :func:`openpi_control.viz.import_viz_modules`: building a
    collision model must not need viser. They ride in on the same extra, but a
    headless check should fail on a missing trimesh, not on a missing browser
    library it never calls.
    """

    trimesh: ModuleType
    yourdfpy: ModuleType


def import_geometry_modules() -> GeometryModules:
    """Import trimesh/yourdfpy, raising a ConfigurationError naming the extra."""
    try:
        import trimesh
        import yourdfpy
    except ImportError as err:
        raise ConfigurationError(f"{_EXTRA_HINT} (missing: {err.name})") from err
    return GeometryModules(trimesh, yourdfpy)


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #


class LinkChain:
    """Forward kinematics for one URDF, evaluated without mutating it.

    ``yourdfpy`` can do this with ``update_cfg`` plus ``get_transform``, and
    this deliberately does not use it, for two reasons:

    * **It must not disturb the render.** The gate evaluates a *candidate* pose
      -- where the arm would be next tick -- against a URDF object that may be
      the one drawing the measured pose. ``update_cfg`` is a write;
      :meth:`openpi_control.viz.ArmVisualizer.world_end_effector_position` has
      to save and restore the configuration around every call for exactly this
      reason, and a gate doing that on the control thread is a race waiting to
      happen.
    * **It is called twice per armed arm per tick.** Measured on the packaged
      YAM, this evaluates all 7 link transforms in ~40 us against ~700 us for
      ``update_cfg`` plus a ``get_transform`` per link -- the difference between
      a rounding error and 4% of a 33 ms tick for a two-arm cell.

    Revolute, continuous, prismatic and fixed joints are handled; anything else
    is refused rather than silently treated as fixed.
    """

    def __init__(self, urdf: Any, joint_order: Sequence[str]) -> None:
        self.base_link = str(urdf.base_link)
        index = {name: position for position, name in enumerate(joint_order)}

        parent: dict[str, str] = {}
        joint_of: dict[str, Any] = {}
        for joint in urdf.robot.joints:
            parent[joint.child] = joint.parent
            joint_of[joint.child] = joint

        # Parents before children, so one forward pass composes the chain.
        ordered = [self.base_link]
        placed = {self.base_link}
        remaining = set(urdf.link_map) - placed
        while remaining:
            progressed = False
            for link in sorted(remaining):
                if parent.get(link) in placed:
                    ordered.append(link)
                    placed.add(link)
                    remaining.discard(link)
                    progressed = True
            if not progressed:
                raise ConfigurationError(
                    "URDF links unreachable from the base link: "
                    + ", ".join(sorted(remaining))
                )
        self.links: tuple[str, ...] = tuple(ordered)
        self._link_index = {link: position for position, link in enumerate(self.links)}

        # One row per link, in `self.links` order. The base link is its own
        # parent with an identity origin, which makes the forward pass uniform.
        self._parent = np.zeros(len(self.links), dtype=np.intp)
        self._origin = np.tile(np.eye(4), (len(self.links), 1, 1))
        self._axis = np.zeros((len(self.links), 3), dtype=np.float64)
        self._joint_index = np.full(len(self.links), -1, dtype=np.intp)
        self._prismatic = np.zeros(len(self.links), dtype=bool)
        for position, link in enumerate(self.links):
            joint = joint_of.get(link)
            if joint is None:
                continue
            self._parent[position] = self._link_index[joint.parent]
            if joint.origin is not None:
                self._origin[position] = np.asarray(joint.origin, dtype=np.float64)
            if joint.type in ("fixed", "floating"):
                continue
            if joint.type not in ("revolute", "continuous", "prismatic"):
                raise ConfigurationError(
                    f"joint {joint.name!r} has unsupported type {joint.type!r}"
                )
            if joint.name not in index:
                # An actuated joint nobody supplies a value for would be pinned
                # at zero, quietly checking a pose the arm is not standing in.
                raise ConfigurationError(
                    f"joint {joint.name!r} is actuated but absent from the joint order"
                )
            # yourdfpy leaves axis as None when the URDF omits the tag; the URDF
            # spec defaults it to +x.
            axis = np.asarray(
                (1.0, 0.0, 0.0) if joint.axis is None else joint.axis, dtype=np.float64
            )
            norm = float(np.linalg.norm(axis))
            if norm < 1e-9:
                raise ConfigurationError(f"joint {joint.name!r} has a zero-length axis")
            self._axis[position] = axis / norm
            self._joint_index[position] = index[joint.name]
            self._prismatic[position] = joint.type == "prismatic"

    def transforms(self, positions: np.ndarray) -> np.ndarray:
        """``(len(links), 4, 4)`` link poses in the base-link frame."""
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        out = np.empty((len(self.links), 4, 4), dtype=np.float64)
        for position in range(len(self.links)):
            local = self._origin[position]
            joint = int(self._joint_index[position])
            if joint >= 0:
                value = float(values[joint])
                step = np.eye(4)
                if self._prismatic[position]:
                    step[:3, 3] = self._axis[position] * value
                else:
                    step[:3, :3] = _rotation(self._axis[position], value)
                local = local @ step
            out[position] = local if position == 0 else out[self._parent[position]] @ local
        return out


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation about a unit ``axis``."""
    cos, sin = np.cos(angle), np.sin(angle)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + sin * skew + (1.0 - cos) * (skew @ skew)
