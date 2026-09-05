"""MolmoAct2 HTTP inference and bimanual YAM observation/action mapping.

The robot process deliberately stays a small policy client.  A GPU host runs
the MolmoAct2 server and this module sends the three RGB observations over the
server's ``json_numpy`` HTTP protocol, then turns each returned 14-D action
into the native six-joint-plus-effector commands used by this package.

Every default here is the one the reference deployment runs, because the
checkpoint's behaviour is sensitive to how it is driven and those values were
settled on hardware rather than guessed.  The reference is
``examples/yam/{host_server_yam.py,molmoact_client.py,launch_molmoact_inference.py}``
in the MolmoAct2 tree, and each constant below names the measurement it came
from.  Deviations from it are marked ``deviation:`` and there are only three:
a bounded request timeout, URDF joint-limit clipping, and clipping the gripper
into [0, 1] -- all of which this package needs and none of which change what
the policy is asked to do.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .exceptions import ConfigurationError, PiControlError, StaleStateError
from .types import ArmState, PositionCommand

MOLMOACT_STATE_DIM = 14
MOLMOACT_ACTION_DIM = 14
MOLMOACT_ARM_DOF = 6
MOLMOACT_GRIPPER_DOF = 1
MOLMOACT_CAMERA_NAMES = ("top", "left_wrist", "right_wrist")
MOLMOACT_ARM_NAMES = ("left", "right")
DEFAULT_MOLMOACT_SERVER = "http://127.0.0.1:8202/act"

# Health routes, tried in this order before the legacy ``GET /act`` probe.
# ``serve_unified.py`` routes /act to POST only, so the legacy probe comes back
# "405 Method Not Allowed" there and reads as a server that is up but sick --
# which is what an operator pointed at a unified server sees today.
HEALTH_PATHS = ("/healthz", "/health")

# How an encoded frame rides the payload, because the two server generations
# disagree. ``json_numpy`` base64s an ndarray, so host_server_yam.py reads a
# 1-D uint8 array as JPEG bytes; serve_unified.py instead reads a base64
# *string* and answers an array with an opaque HTTP 500 ("Inference failed --
# see server logs"), which says nothing about the frame format being the
# problem. Measured against the pi05 unified server on this cell with three
# 360x640 views, the base64 JPEG payload is 1.09 MB against 2.77 MB raw and
# spends 0.016 s on the wire against 0.039 s -- on noise frames, which is the
# worst case for JPEG, so a real scene gains more. Either way the round trip
# is dominated by the ~0.07-0.15 s the GPU spends, which is why this is a
# compatibility choice first and a bandwidth one second.
FRAME_ENCODING_ARRAY = "array"
FRAME_ENCODING_BASE64 = "base64"
FRAME_ENCODINGS = (FRAME_ENCODING_ARRAY, FRAME_ENCODING_BASE64)

# Health-payload keys only serve_unified.py sends. Either one identifies the
# generation that wants base64-string frames.
UNIFIED_HEALTH_KEYS = ("backend", "camera_order")

# The server's own normalization statistics key. Sent for wire fidelity with
# the reference client and, more usefully, checked against the health payload:
# it is the one field that tells a bimanual-YAM server apart from the DROID
# server, which answers the same /act route with a different state dimension.
MOLMOACT_NORM_TAG = "yam_dual_molmoact2"

# Flow-matching denoising steps. 10 is the server's default and what the model
# card documents; the reference client leaves the key off the payload entirely
# to inherit it.
DEFAULT_MOLMOACT_NUM_STEPS = 10

# The checkpoint's ``max_action_horizon`` (config.json), so a healthy server
# returns exactly this many actions per call. Recorded rather than enforced:
# the whole chunk is executed whatever its length, because executing only a
# prefix is what this number exists to warn against (see BoundedChunkExecutor).
MOLMOACT_ACTION_HORIZON = 30

# JPEG-encode frames on the way out. Three raw 360x640 frames are ~2.8 MB of
# base64 per request, which dominates round-trip time on anything short of
# gigabit ethernet; q95 cuts that ~25x and measured the same wall-clock as q85
# (0.215 s vs 0.205 s, inside the jitter) while keeping the fidelity, so the
# reference client defaults high. 0 disables and sends raw frames.
DEFAULT_MOLMOACT_JPEG_QUALITY = 95

# deviation: the reference client passes no timeout at all. A robot-side client
# that can block forever holds torque on two energized arms while it waits, so
# there is a bound here -- just a generous one. A cold or non-CUDA-graph call
# takes 9-16 s, so anything near the old 10 s default timed out exactly when
# the server was slowest.
DEFAULT_REQUEST_TIMEOUT_S = 60.0

# Safety clamp for the default executor: the furthest a joint may be commanded
# from its previously commanded target in one control tick. Deliberately
# tighter than the reference deployment's 0.3, because this package drives the
# arms through the native node rather than i2rt's direct CAN writes, and
# measured on this rig the gentler clamp behaves better.
DEFAULT_MAX_STEP_RAD = 0.10

# The same clamp for the gripper, which gets its own budget because it is not a
# joint travelling through space -- it is a jaw that has to finish closing
# *before* the arm moves on. Sharing the joints' 0.10 caps a full open-to-close
# at ten ticks, a third of a second at 30 Hz, which is most of a grasp: the jaw
# is still travelling while the plan has already moved the hand away. Measured
# on a fold run against this checkpoint the effector, not the arms, was what
# most of the clamp count was refusing. 0.30 reaches either stop in four ticks
# (~0.13 s) and still refuses a single garbage action that would slam the jaw.
DEFAULT_MAX_EFFECTOR_STEP = 0.30

# Chunk playback speed. 1.0 consumes one action per control tick, which is the
# rate the demonstrations were recorded at and the rate the checkpoint plans
# for. Below 1.0 the same chunk is played over proportionally more ticks, with
# the commands linearly interpolated between the policy's actions.
#
# This is the knob for a run that lunges: it slows the arms down without
# shortening what they do. A tighter ``max_step_rad`` looks like the same
# thing and is not -- the clamp files each action down and then moves on to the
# next one, so a plan that asks for more than the clamp allows is executed as a
# shrunken copy of itself and the reach never lands. Time-scaling keeps every
# millimetre of the path and only spends longer on it. It also holds the policy
# back: a chunk that takes twice as long is re-planned half as often, so an
# observation gets acted on to its conclusion instead of being overtaken.
DEFAULT_CHUNK_SPEED = 1.0

# Safety clamp for the reference executor (``ReachingChunkExecutor``): the
# furthest any single value may be commanded from its *measured* position in
# one action. 0.3 rad is the reference deployment's ``--max-joint-step-rad``.
DEFAULT_MAX_JOINT_STEP_RAD = 0.3

# Sub-tick interpolation: one command per ~0.01 rad of the largest move, at
# most 100 of them, 1 ms apart. This is what makes the arm *arrive* at each
# action in the chunk (~0.13 s per action in practice) instead of low-pass
# filtering the plan into a fragment of it.
SUB_STEP_RAD = 0.01
MAX_SUB_STEPS = 100
SUB_STEP_PERIOD_S = 0.001

# How early a prefetch fires: the next chunk is requested once the motion still
# queued is shorter than the measured inference latency plus this margin. It
# covers the jitter the EMA has already smoothed away -- too small and the arm
# runs dry before the chunk lands, too large and the observation the policy is
# handed is stale by the time it acts on it.
DEFAULT_PREFETCH_MARGIN_S = 0.05

# The pose every training demonstration begins from: the median first frame of
# allenai/20122025-foldclo-05 (10/10 episodes), gripper held open.
#
# This matters more than it looks. The obvious alternative -- norm_stats'
# state_stats.q50 -- is the median over every frame of every episode, i.e. a
# mid-task pose with cloth already in hand. Starting there hands the policy a
# state saying "arms raised, mid-fold" while the cameras show an untouched
# table: a combination that never occurs in training, and the policy then runs
# on proprioception alone and mills aimlessly. Beginning inference from the
# distribution the demonstrations begin from is not a nicety.
MOLMOACT_START_JOINTS: dict[str, tuple[float, ...]] = {
    "left": (0.021, 0.015, 0.049, -0.133, 0.174, 0.234),
    "right": (0.015, 0.016, 0.130, -0.418, -0.029, -0.023),
}
# i2rt gripper convention, which is the native one here too: 1.0 is open.
# Training episode starts show ~0.99.
MOLMOACT_START_EFFECTOR = 1.0


class InferenceError(PiControlError):
    """The policy server or its response could not be used safely."""


class InferenceConnectionError(InferenceError):
    """The server could not be reached; startup may retry this failure."""


class EncodedFramesUnsupported(InferenceError):
    """The server decodes raw HxWx3 frames only, not JPEG-encoded ones.

    Servers predating the encoded-frame branch in their ``_to_pil`` reject a
    1-D uint8 array outright. Raised as its own type so a caller that has
    already energized the arms can drop to raw frames and keep the run rather
    than lose it to a payload format. Told apart from a current server's
    complaint about a genuinely bad frame by :func:`server_rejects_encoded_frames`.
    """


# The two server generations answer an unusable frame with messages that differ
# by one clause, and *both* contain "HxWx3":
#
#   raw-only:  image must be HxWx3, got shape (271814,)
#   current:   image must be HxWx3 (raw) or 1-D uint8 (encoded), got shape (360, 640, 4)
#
# Only the second advertises the encoded form, so only its absence means the
# server cannot decode a JPEG. Matching "HxWx3" alone reads a current server
# complaining about a frame that is genuinely malformed -- an RGBA frame after a
# camera reconfigure, a 2-D grayscale one -- as an out-of-date server, which
# drops the run to raw frames for a fault raw frames cannot fix and points the
# operator at the wrong box.
_RAW_FRAME_COMPLAINT = "HxWx3"
_ENCODED_FRAME_CLAUSE = "1-D uint8"
# serve_unified.py rejects the array form with a third message, which also
# contains "HxWx3" and advertises the encoded form it *does* take:
#
#   Could not interpret image of shape (139363,) as HxWx3 RGB (...). Send
#   HxWx3 / HxW / CxHxW uint8, a PNG/JPEG as base64, or a PIL image.
#
# That is not a server which needs raw frames -- it needs the other encoded
# container, which :meth:`MolmoActClient.health` has already negotiated. Left
# to the raw-only branch it would cost the run 2.5x the payload for nothing.
_BASE64_FRAME_CLAUSE = "base64"


def server_rejects_encoded_frames(response_body: str) -> bool:
    """True when a frame complaint came from a raw-only ``_to_pil``.

    Deliberately biased towards "the server understands encoded frames": a
    wrong "unsupported" silently halves the run's throughput for the rest of
    the episode and blames the server, while a wrong "supported" surfaces the
    server's own message, which says what is actually wrong with the frame.
    A complaint that names an encoded form it accepts -- either generation's --
    is therefore never read as a server that cannot decode one.
    """
    if _RAW_FRAME_COMPLAINT not in response_body:
        return False
    advertised = (_ENCODED_FRAME_CLAUSE, _BASE64_FRAME_CLAUSE)
    return not any(clause in response_body for clause in advertised)


def normalize_server_url(server: str | None) -> str:
    """Return a full MolmoAct ``/act`` URL from a host, URL, or empty value."""
    value = (server or DEFAULT_MOLMOACT_SERVER).strip().rstrip("/")
    if not value:
        value = DEFAULT_MOLMOACT_SERVER
    if "://" not in value:
        value = "http://" + value
    if not value.endswith("/act"):
        value += "/act"
    return value


def health_urls(act_url: str) -> tuple[str, ...]:
    """The health routes to try, in order, for one ``/act`` URL.

    The dedicated routes come first and the legacy ``GET /act`` probe stays
    last, so a unified server answers on the first try and a reference server
    costs two cheap 404s before the probe that works.
    """
    base = act_url[: -len("/act")] if act_url.endswith("/act") else act_url
    return tuple(base + path for path in HEALTH_PATHS) + (act_url,)


def _json_numpy() -> Any:
    try:
        import json_numpy
    except ImportError as err:  # pragma: no cover - depends on optional extra
        raise ConfigurationError(
            "MolmoAct inference needs the optional 'inference' extra: uv sync --extra inference"
        ) from err
    return json_numpy


def _pil_image() -> Any:
    try:
        from PIL import Image
    except ImportError as err:  # pragma: no cover - depends on optional extra
        raise ConfigurationError(
            "JPEG frame transport needs the optional 'inference' extra: "
            "uv sync --extra inference (or pass --raw-frames)"
        ) from err
    return Image


def _new_session() -> Any:
    """A keep-alive HTTP session, so one TCP connection serves every call.

    A fresh connection per inference pays a handshake each time, which is real
    money when the round trip is tens of milliseconds and the policy is asked
    for a chunk several times a second.
    """
    try:
        import requests
    except ImportError as err:  # pragma: no cover - depends on optional extra
        raise ConfigurationError(
            "MolmoAct inference needs the optional 'inference' extra: uv sync --extra inference"
        ) from err
    return requests.Session()


def _as_rgb(frame: np.ndarray, *, pixel_format: str = "rgb8") -> np.ndarray:
    """Return a contiguous uint8 RGB camera frame for the wire payload."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise InferenceError(f"camera frame must be HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if pixel_format == "bgr8":
        image = image[..., ::-1]
    elif pixel_format != "rgb8":
        raise InferenceError(f"unsupported camera pixel format {pixel_format!r}")
    return np.ascontiguousarray(image)


def encode_frame(
    frame: np.ndarray, *, jpeg_quality: int, encoding: str = FRAME_ENCODING_ARRAY
) -> np.ndarray | str:
    """Return one frame in wire form: JPEG bytes, or the raw array at quality 0.

    ``json_numpy`` base64s any ndarray, so a 1-D uint8 array of JPEG bytes
    rides the existing payload with no new keys and no format negotiation --
    the reference server treats 1-D uint8 as an encoded image and HxWx3 as a
    raw frame. ``encoding=FRAME_ENCODING_BASE64`` sends the same JPEG as a
    base64 string instead, which is the form serve_unified.py decodes; the
    picture is identical either way, only its container differs.

    The frame is validated and cast exactly the way the server's ``_to_pil``
    treats a raw one, so the transport is not also a change of picture. Pillow
    reads the buffer against ``mode="RGB"`` and raises a bare ``TypeError`` on
    anything but uint8, so without this JPEG would fail on a frame that raw
    transport delivers fine.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise InferenceError(f"camera frame must be HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if jpeg_quality <= 0:
        return image
    if encoding not in FRAME_ENCODINGS:
        raise ConfigurationError(
            f"frame encoding must be one of {FRAME_ENCODINGS}, got {encoding!r}"
        )
    buffer = io.BytesIO()
    _pil_image().fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=jpeg_quality)
    jpeg = buffer.getvalue()
    if encoding == FRAME_ENCODING_BASE64:
        return base64.b64encode(jpeg).decode("ascii")
    return np.frombuffer(jpeg, dtype=np.uint8)


def decode_wire_frame(frame: np.ndarray | str) -> np.ndarray:
    """One wire frame as the server decodes it: RGB ``HxWx3`` uint8.

    The mirror of :func:`encode_frame`, and it exists so a preview can show the
    picture the model was handed -- compression artifacts and all -- instead of
    the pre-encoding original. A raw frame passes straight through, which is
    what the server's own decoder does with it too.
    """
    if isinstance(frame, str):
        # A base64-string frame, optionally as a data URI: both are forms the
        # unified server accepts, so both have to come back as a picture.
        payload = frame.split(",", 1)[1] if frame.startswith("data:") else frame
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (ValueError, TypeError) as err:
            raise InferenceError(f"a base64 wire frame did not decode: {err}") from err
        image = _pil_image().open(io.BytesIO(decoded)).convert("RGB")
        return np.asarray(image, dtype=np.uint8)
    array = np.asarray(frame)
    if array.ndim == 3:
        return array
    if array.ndim != 1:
        raise InferenceError(f"a wire frame is 1-D encoded or HxWx3 raw, got {array.shape}")
    image = _pil_image().open(io.BytesIO(array.tobytes())).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def served_frames(client: MolmoActClient) -> dict[str, np.ndarray]:
    """What the policy last saw, keyed by the rig's camera names."""
    return {
        name: decode_wire_frame(frame) for name, frame in client.last_wire_frames.items()
    }


@dataclass(frozen=True, slots=True)
class BimanualObservation:
    """One policy observation in the MolmoAct2 bimanual YAM contract."""

    top_cam: np.ndarray
    left_cam: np.ndarray
    right_cam: np.ndarray
    state: np.ndarray

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32).reshape(-1)
        if state.shape != (MOLMOACT_STATE_DIM,):
            raise ConfigurationError(
                f"MolmoAct2 state must have shape ({MOLMOACT_STATE_DIM},), got {state.shape}"
            )
        if not np.all(np.isfinite(state)):
            raise ConfigurationError("MolmoAct2 state contains non-finite values")
        object.__setattr__(self, "state", state)


class MolmoActClient:
    """HTTP client for ``host_server_yam.py`` or a compatible server."""

    def __init__(
        self,
        server: str | None = None,
        *,
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        num_steps: int = DEFAULT_MOLMOACT_NUM_STEPS,
        jpeg_quality: int = DEFAULT_MOLMOACT_JPEG_QUALITY,
        enable_cuda_graph: bool = True,
        frame_encoding: str | None = None,
        session: Any | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ConfigurationError("inference request timeout must be positive")
        if num_steps <= 0:
            raise ConfigurationError("inference num_steps must be positive")
        self.url = normalize_server_url(server)
        self.timeout_s = timeout_s
        self.num_steps = int(num_steps)
        self.jpeg_quality = int(jpeg_quality)
        # On by default, and sent on every request rather than left to the
        # server's launch flags: CUDA-graph capture in the action expert took
        # inference from 9-16 s to ~0.5 s with real 360x640 frames (0.31 s ->
        # 0.18 s of pure GPU time). Sending it explicitly is what makes the
        # fast path independent of how the operator started the server -- and
        # sending False would *override* a server launched with --cuda-graph.
        # Safe under a shared server because it serializes predict_action
        # behind a lock; graph capture is not concurrency-safe.
        self.enable_cuda_graph = bool(enable_cuda_graph)
        # Which encoded-frame container this server understands. Left to
        # :meth:`health` to settle from the server's own payload, because
        # guessing it wrong costs an HTTP 500 that names no cause; passing one
        # explicitly pins it, for a server that reports something new.
        if frame_encoding is not None and frame_encoding not in FRAME_ENCODINGS:
            raise ConfigurationError(
                f"frame encoding must be one of {FRAME_ENCODINGS}, got {frame_encoding!r}"
            )
        self.frame_encoding = frame_encoding or FRAME_ENCODING_ARRAY
        self._frame_encoding_pinned = frame_encoding is not None
        # The last health payload, kept so the operator banner can name the
        # checkpoint that actually answered.
        self.last_health: dict[str, Any] = {}
        # Per-call latency, split the way the reference client reports it: the
        # GPU/transport split is the number that diagnoses a slow call.
        self.last_latency: dict[str, float] = {}
        # The frames exactly as they went out, kept so a preview can show what
        # the server was handed rather than a fresh read taken beside it.
        self.last_wire_frames: dict[str, np.ndarray | str] = {}
        self._session = session

    @property
    def session(self) -> Any:
        """The keep-alive session, opened on first use."""
        if self._session is None:
            self._session = _new_session()
        return self._session

    def close(self) -> None:
        """Drop the pooled connection. Safe to call on a client never used."""
        session, self._session = self._session, None
        closer = getattr(session, "close", None)
        if closer is not None:
            closer()

    def health(self) -> dict[str, Any]:
        """Check the server before any motors are energized.

        The payload is validated, not just the status: a bimanual-YAM server
        and a DROID server both answer with ``{"status": "ok"}`` and differ
        only in the fields below, and finding that out from the arms'
        behaviour is a worse way to find it out.

        Answering also settles how frames go on the wire, since the generation
        that owns the route that answered is the one that says which container
        it can decode (see :data:`FRAME_ENCODING_ARRAY`).
        """
        missing: list[str] = []
        for url in health_urls(self.url):
            try:
                response = self.session.get(url, timeout=self.timeout_s)
            except (OSError, TimeoutError) as err:
                raise InferenceConnectionError(
                    f"cannot reach MolmoAct server {url}: {err}"
                ) from err
            status = int(getattr(response, "status_code", 200))
            # 404: this generation does not route it. 405: the route is there
            # but POST-only, which is what GET /act answers on a unified
            # server. Neither is a sick server, so try the next candidate.
            if status in (404, 405):
                missing.append(f"{url} HTTP {status}")
                continue
            payload = self._validated_health(url, _response_text(response))
            self._adopt_frame_encoding(payload)
            self.last_health = payload
            return payload
        raise InferenceError(
            f"no health route answered on {self.url}: {', '.join(missing)}"
        )

    def _validated_health(self, url: str, body: str) -> dict[str, Any]:
        """Parse one health body and check it against the bimanual contract."""
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as err:
            raise InferenceError(f"MolmoAct health response was not JSON: {err}") from err
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise InferenceError(f"MolmoAct server is not ready: {payload!r}")
        for key, expected in (
            ("norm_tag", MOLMOACT_NORM_TAG),
            ("state_dim", MOLMOACT_STATE_DIM),
            ("action_dim", MOLMOACT_ACTION_DIM),
            ("num_cameras", len(MOLMOACT_CAMERA_NAMES)),
        ):
            actual = payload.get(key)
            if actual is not None and actual != expected:
                raise InferenceError(
                    f"MolmoAct server at {url} reports {key}={actual!r}, "
                    f"but the bimanual YAM contract needs {expected!r}"
                )
        # camera_order names the same three views, but in the server's own
        # labels ("left" for what this package calls left_wrist), so only the
        # count is comparable -- and the count is what a single-view or DROID
        # checkpoint gets wrong.
        order = payload.get("camera_order")
        if isinstance(order, Sequence) and not isinstance(order, str):
            if len(order) != len(MOLMOACT_CAMERA_NAMES):
                raise InferenceError(
                    f"MolmoAct server at {url} serves {len(order)} cameras "
                    f"{list(order)!r}, but the bimanual YAM contract needs "
                    f"{len(MOLMOACT_CAMERA_NAMES)}"
                )
        return payload

    def _adopt_frame_encoding(self, payload: Mapping[str, Any]) -> None:
        """Adopt the encoded-frame container the answering server decodes.

        Skipped when the constructor pinned one, so an operator keeps the last
        word against a server that reports something this table has not seen.
        """
        if self._frame_encoding_pinned:
            return
        unified = any(key in payload for key in UNIFIED_HEALTH_KEYS)
        self.frame_encoding = FRAME_ENCODING_BASE64 if unified else FRAME_ENCODING_ARRAY

    def infer(
        self,
        observation: BimanualObservation,
        instruction: str,
        *,
        num_steps: int | None = None,
        enable_cuda_graph: bool | None = None,
    ) -> np.ndarray:
        """POST an observation and return a validated ``(N, 14)`` action chunk."""
        if not instruction.strip():
            raise ConfigurationError("inference instruction must not be empty")
        steps = self.num_steps if num_steps is None else int(num_steps)
        if steps <= 0:
            raise ConfigurationError("inference num_steps must be positive")
        graph = self.enable_cuda_graph if enable_cuda_graph is None else bool(enable_cuda_graph)
        json_numpy = _json_numpy()
        quality = self.jpeg_quality
        encoding = self.frame_encoding
        wire_frames = {
            "top": encode_frame(observation.top_cam, jpeg_quality=quality, encoding=encoding),
            "left_wrist": encode_frame(
                observation.left_cam, jpeg_quality=quality, encoding=encoding
            ),
            "right_wrist": encode_frame(
                observation.right_cam, jpeg_quality=quality, encoding=encoding
            ),
        }
        self.last_wire_frames = wire_frames
        payload = {
            "top_cam": wire_frames["top"],
            "left_cam": wire_frames["left_wrist"],
            "right_cam": wire_frames["right_wrist"],
            "instruction": instruction,
            "state": np.asarray(observation.state, dtype=np.float32),
            "timestamp": time.time(),
            "num_steps": steps,
            "normalization_tag": MOLMOACT_NORM_TAG,
            "enable_cuda_graph": graph,
        }
        body = json_numpy.dumps(payload)
        started = time.perf_counter()
        try:
            response = self.session.post(
                self.url,
                data=body.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_s,
            )
        except (OSError, TimeoutError) as err:
            raise InferenceError(f"MolmoAct inference request failed: {err}") from err
        round_trip_s = time.perf_counter() - started
        status = getattr(response, "status_code", 200)
        response_body = _response_text(response)
        if status != 200:
            # The server's own complaint about the frame's shape is the only
            # signal it gives that it cannot decode an encoded one.
            if quality > 0 and server_rejects_encoded_frames(response_body):
                raise EncodedFramesUnsupported(
                    f"MolmoAct server at {self.url} cannot decode JPEG frames "
                    f"(HTTP {status}: {response_body.strip()})"
                )
            raise InferenceError(f"MolmoAct server returned HTTP {status}: {response_body}")

        try:
            result = json_numpy.loads(response_body)
            actions = np.asarray(result["actions"], dtype=np.float64)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as err:
            raise InferenceError(f"invalid MolmoAct response: {err}") from err
        if actions.ndim != 2 or actions.shape[1] != MOLMOACT_ACTION_DIM or actions.shape[0] == 0:
            raise InferenceError(
                f"MolmoAct actions must have shape (N, {MOLMOACT_ACTION_DIM}), got {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise InferenceError("MolmoAct actions contain non-finite values")
        gpu_s = float(result.get("dt_ms") or 0.0) / 1000.0 if isinstance(result, dict) else 0.0
        self.last_latency = {
            "round_trip_s": round_trip_s,
            "gpu_s": gpu_s,
            "transport_s": max(0.0, round_trip_s - gpu_s),
            "payload_mb": len(body) / 1e6,
            "n_actions": float(actions.shape[0]),
        }
        return actions


def _response_text(response: Any) -> str:
    """Read a response body from a requests-like or file-like response."""
    text = getattr(response, "text", None)
    if text is not None:
        return text
    read = response.read()
    return read.decode("utf-8") if isinstance(read, bytes) else str(read)


class ChunkPrefetcher:
    """Compute the next action chunk while the current one is still executing.

    Only ever one request is in flight: the server serializes ``predict_action``
    behind a lock, so a second concurrent call would queue rather than overlap.
    The win is hiding inference latency behind arm motion, not parallelism.

    The observation is always built by the caller, on the caller's thread, so
    the freshness checks that guard it stay where they can stop the loop.
    """

    def __init__(self, client: MolmoActClient, *, initial_latency_s: float = 0.35) -> None:
        self._client = client
        self.latency_s = float(initial_latency_s)
        self.used_pending = False
        self._lock = threading.Lock()
        # A plain daemon thread rather than a pool: a discarded inference should
        # be abandoned at exit, not joined, so a Ctrl-C landing mid-request does
        # not surface as a traceback out of threading's shutdown.
        self._thread: threading.Thread | None = None
        self._result: np.ndarray | None = None
        self._error: BaseException | None = None

    @property
    def busy(self) -> bool:
        """True while a chunk is in flight *or* computed but not yet consumed."""
        return self._thread is not None

    def submit(self, observation: BimanualObservation, instruction: str) -> None:
        """Start inference for the next chunk. A no-op while one is pending."""
        if self._thread is not None:
            return
        with self._lock:
            self._result = self._error = None

        def _run() -> None:
            started = time.perf_counter()
            try:
                actions = self._client.infer(observation, instruction)
            except BaseException as err:  # noqa: BLE001 - re-raised from take()
                with self._lock:
                    self._error = err
                return
            with self._lock:
                self._result = actions
                # 0.3 EMA: follows a changing link quickly without letting one
                # slow call dominate the trigger point.
                self.latency_s = 0.3 * (time.perf_counter() - started) + 0.7 * self.latency_s

        self._thread = threading.Thread(target=_run, name="molmoact-prefetch", daemon=True)
        self._thread.start()

    def take(self, observation: BimanualObservation, instruction: str) -> np.ndarray:
        """Return the next chunk, falling back to a synchronous call."""
        self.used_pending = self._thread is not None
        if self._thread is None:
            self.submit(observation, instruction)
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            error, result = self._error, self._result
            self._result = self._error = None
        if error is not None:
            raise error
        if result is None:
            raise InferenceError("prefetched inference returned no action chunk")
        return result

    def drop(self) -> None:
        """Abandon an in-flight or unconsumed chunk; it is stale now."""
        self._thread = None
        with self._lock:
            self._result = self._error = None

    def close(self) -> None:
        """Release state. Never blocks: the worker is a daemon thread."""
        self.drop()


def _state_for_arm(arm: Any, *, name: str, max_age_s: float) -> ArmState:
    state = arm.latest_state
    if state is None:
        raise StaleStateError(f"{name} has published no state")
    if not state.is_fresh(max_age_s):
        raise StaleStateError(f"{name} state is {state.age_s * 1e3:.0f} ms old")
    if state.joints.position_rad.size != MOLMOACT_ARM_DOF:
        raise ConfigurationError(
            f"{name} must publish {MOLMOACT_ARM_DOF} arm joints, "
            f"got {state.joints.position_rad.size}"
        )
    if state.effector is None:
        raise ConfigurationError(f"{name} must publish a YAM gripper state")
    return state


def measured_vector(states: Mapping[str, ArmState]) -> np.ndarray:
    """The 14-value ``[left, right]`` vector the state and actions both use."""
    values: list[float] = []
    for name in MOLMOACT_ARM_NAMES:
        state = states.get(name)
        if state is None:
            raise ConfigurationError(f"MolmoAct2 bimanual inference needs arm {name!r}")
        if state.effector is None:
            raise ConfigurationError(f"{name} must publish a YAM gripper state")
        values.extend(float(value) for value in state.joints.position_rad)
        values.append(float(state.effector.position))
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (MOLMOACT_STATE_DIM,):
        raise ConfigurationError(
            f"measured state must have shape ({MOLMOACT_STATE_DIM},), got {vector.shape}"
        )
    return vector


def command_lag(
    commands: Mapping[str, PositionCommand], states: Mapping[str, ArmState]
) -> float:
    """How far the arms are running behind the joint targets they were given.

    The largest single-joint gap, in rad, between a commanded target and the
    measured pose sampled beside it. It is the number that separates the two
    ways a run can look wrong: a chunk whose clamp count is low but whose lag
    is large is a plan the *hardware* is not keeping up with, and slowing
    playback down (``time_scale``) is the lever; a chunk with the opposite
    split is the clamp refusing the plan, and ``max_step_rad`` is.
    """
    worst = 0.0
    for name, command in commands.items():
        state = states.get(name)
        if state is None:
            continue
        measured = np.asarray(state.joints.position_rad, dtype=np.float64)
        target = np.asarray(command.position_rad, dtype=np.float64)
        if measured.shape != target.shape:
            continue
        worst = max(worst, float(np.abs(target - measured).max()))
    return worst


def build_observation(
    live_arms: Mapping[str, Any],
    readers: Mapping[str, Any],
    *,
    max_age_s: float = 0.25,
) -> BimanualObservation:
    """Build the exact left/right and top/left/right ordering the checkpoint uses."""
    missing_arms = set(MOLMOACT_ARM_NAMES).difference(live_arms)
    if missing_arms:
        raise ConfigurationError(
            "MolmoAct2 bimanual inference needs arms: " + ", ".join(sorted(missing_arms))
        )
    states = {
        name: _state_for_arm(live_arms[name], name=name, max_age_s=max_age_s)
        for name in MOLMOACT_ARM_NAMES
    }

    frames: dict[str, np.ndarray] = {}
    for name in MOLMOACT_CAMERA_NAMES:
        reader = readers.get(name)
        if reader is None:
            raise ConfigurationError(f"MolmoAct2 inference needs camera {name!r}")
        frame = reader.latest()
        if frame is None:
            raise InferenceError(f"camera {name!r} has not delivered a frame")
        frames[name] = _as_rgb(frame, pixel_format=getattr(reader, "pixel_format", "rgb8"))

    return BimanualObservation(
        top_cam=frames["top"],
        left_cam=frames["left_wrist"],
        right_cam=frames["right_wrist"],
        state=np.asarray(measured_vector(states), dtype=np.float32),
    )


def split_action(action: Sequence[float] | np.ndarray) -> dict[str, np.ndarray]:
    """Split one model action into native arm-plus-effector vectors."""
    vector = np.asarray(action, dtype=np.float64).reshape(-1)
    if vector.shape != (MOLMOACT_ACTION_DIM,):
        raise ConfigurationError(
            f"MolmoAct2 action must have shape ({MOLMOACT_ACTION_DIM},), got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ConfigurationError("MolmoAct2 action contains non-finite values")
    return {
        "left": vector[:7].copy(),
        "right": vector[7:].copy(),
    }


def split_chunk(actions: Sequence[Sequence[float]] | np.ndarray) -> dict[str, np.ndarray]:
    """Split a whole action chunk into per-arm blocks, columns kept intact.

    The per-arm layout lives in :func:`split_action`; this is the same split
    applied to every row at once, for consumers that want the chunk rather than
    one action -- the Viser overlay, which draws the trail a chunk predicts.
    """
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != MOLMOACT_ACTION_DIM:
        raise ConfigurationError(
            f"MolmoAct2 chunk must have shape (N, {MOLMOACT_ACTION_DIM}), got {values.shape}"
        )
    return {"left": values[:, :7].copy(), "right": values[:, 7:].copy()}


def time_scale(
    actions: Sequence[Sequence[float]] | np.ndarray, speed: float = DEFAULT_CHUNK_SPEED
) -> np.ndarray:
    """Resample a chunk so one control tick advances ``speed`` of one action.

    The policy's actions are absolute poses sampled at the training rate, so
    playing the chunk slower is a resample and nothing more: tick ``t`` samples
    the plan at action-time ``t * speed`` and interpolates between the two
    actions it falls between. The path, its shape, and its endpoint are
    untouched -- only the time spent on it changes. ``speed`` above 1.0 drops
    ticks and runs the chunk fast; 1.0 returns the chunk itself.

    See ``DEFAULT_CHUNK_SPEED`` for why this and the step clamp are not
    interchangeable.
    """
    if speed <= 0:
        raise ConfigurationError("chunk speed must be positive")
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != MOLMOACT_ACTION_DIM:
        raise ConfigurationError(
            f"MolmoAct2 chunk must have shape (N, {MOLMOACT_ACTION_DIM}), got {values.shape}"
        )
    if speed == 1.0 or len(values) < 2:
        return values
    last = len(values) - 1
    # One tick past the final action's arrival, so the chunk always ends on the
    # action the policy planned to end on rather than short of it.
    steps = int(np.ceil(last / speed)) + 1
    positions = np.minimum(np.arange(steps, dtype=np.float64) * speed, last)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, last)
    frac = (positions - lower)[:, None]
    return values[lower] * (1.0 - frac) + values[upper] * frac


def _commands_from_row(row: np.ndarray) -> dict[str, PositionCommand]:
    """One interpolation row as a per-arm native command."""
    split = split_action(row)
    return {
        name: PositionCommand(goal[:MOLMOACT_ARM_DOF], effector=float(goal[-1]))
        for name, goal in split.items()
    }


def interpolate(
    measured: np.ndarray,
    target: np.ndarray,
    *,
    sub_step_rad: float = SUB_STEP_RAD,
    max_sub_steps: int = MAX_SUB_STEPS,
) -> list[dict[str, PositionCommand]]:
    """Walk from the measured pose to ``target`` in bounded sub-steps.

    One row per ``sub_step_rad`` of the largest single-value move, capped at
    ``max_sub_steps``; the caller paces them ``SUB_STEP_PERIOD_S`` apart. This
    is the primitive both the chunk executor and the start-pose ramp use, and
    it is why a commanded target is reached rather than approached.
    """
    travel = float(np.abs(np.asarray(target) - np.asarray(measured)).max())
    steps = min(int(travel / sub_step_rad), max_sub_steps)
    rows = (
        np.asarray(target, dtype=np.float64).reshape(1, -1)
        if steps <= 1
        else np.linspace(measured, target, steps)
    )
    return [_commands_from_row(row) for row in rows]


def start_pose_plan(
    states: Mapping[str, ArmState],
    *,
    sub_step_rad: float = SUB_STEP_RAD,
    max_sub_steps: int = MAX_SUB_STEPS,
) -> list[dict[str, PositionCommand]]:
    """Ramp from where the arms stand to the pose the demonstrations start at.

    Every rollout then begins from the same in-distribution initial condition,
    which is the point: see ``MOLMOACT_START_JOINTS``.
    """
    target = np.concatenate(
        [
            np.asarray(
                (*MOLMOACT_START_JOINTS[name], MOLMOACT_START_EFFECTOR), dtype=np.float64
            )
            for name in MOLMOACT_ARM_NAMES
        ]
    )
    return interpolate(
        measured_vector(states),
        target,
        sub_step_rad=sub_step_rad,
        max_sub_steps=max_sub_steps,
    )


# A gripper that is not tracking: how far the commanded effector has to have
# swept *across the run*, against how little the measured one moved, before it
# is called out. 0.25 is a quarter of the normalized stroke; 0.02 is just above
# the E_Yam's own position noise (pos_error_margin is 0.1 rad of the stroke).
GRIPPER_STALL_COMMAND_TRAVEL = 0.25
GRIPPER_STALL_MEASURED_TRAVEL = 0.02


@dataclass(slots=True)
class GripperWatch:
    """Watches the gripper actually do what the policy asked it to.

    Every freshness check in this package is about the *message*: ``is_fresh``
    times the published state, and the node publishes on its own clock whether
    or not the gripper servo moved. An inert gripper therefore looks completely
    healthy to the arm state, to the policy, and to the operator -- the joints
    keep tracking and only the gripper is dead. Nothing else here would notice.

    The spans are cumulative over the whole run, which is what makes the test
    both sensitive and safe. Per chunk it was neither: a policy that walks the
    gripper open in small steps never sweeps far enough inside one chunk to
    trip anything, so the failure this exists for went unreported for a whole
    session. Cumulatively it shows up immediately. And a gripper clamped on an
    object -- the obvious false positive, since it too sits still while being
    commanded shut -- had to travel to reach the object, so its measured span
    is already wide by the time it is holding.
    """

    command_travel: float = GRIPPER_STALL_COMMAND_TRAVEL
    measured_travel: float = GRIPPER_STALL_MEASURED_TRAVEL
    commanded: dict[str, float] = field(default_factory=dict)
    measured: dict[str, float] = field(default_factory=dict)
    _cmd_span: dict[str, tuple[float, float]] = field(default_factory=dict)
    _meas_span: dict[str, tuple[float, float]] = field(default_factory=dict)

    def observe(
        self,
        commands: Mapping[str, PositionCommand],
        states: Mapping[str, ArmState],
    ) -> None:
        for name, command in commands.items():
            if command.effector is not None:
                self.commanded[name] = float(command.effector)
                self._extend(self._cmd_span, name, float(command.effector))
        for name, state in states.items():
            if state.effector is not None:
                self.measured[name] = float(state.effector.position)
                self._extend(self._meas_span, name, float(state.effector.position))

    @staticmethod
    def _extend(spans: dict[str, tuple[float, float]], name: str, value: float) -> None:
        low, high = spans.get(name, (value, value))
        spans[name] = (min(low, value), max(high, value))

    def span(self, spans: Mapping[str, tuple[float, float]], name: str) -> float:
        low, high = spans.get(name, (0.0, 0.0))
        return high - low

    def stalled(self) -> list[str]:
        """Arms whose gripper has been swept a long way and never once moved."""
        names = []
        for name in self._cmd_span:
            if name not in self._meas_span:
                continue
            if (
                self.span(self._cmd_span, name) >= self.command_travel
                and self.span(self._meas_span, name) <= self.measured_travel
            ):
                names.append(name)
        return sorted(names)

    def render(self) -> str:
        """One compact ``arm cmd/measured`` field per arm for the chunk log."""
        parts = []
        for name in MOLMOACT_ARM_NAMES:
            command = self.commanded.get(name)
            measured = self.measured.get(name)
            if command is None and measured is None:
                continue
            parts.append(
                f"{name[0]}{'--' if command is None else f'{command:.2f}'}"
                f"/{'--' if measured is None else f'{measured:.2f}'}"
            )
        return " ".join(parts)


@dataclass(slots=True)
class BoundedChunkExecutor:
    """Apply absolute action chunks with Karma-style bounded interpolation.

    One command per control tick, each within ``max_step_rad`` of the previously
    commanded target. This is the default because it is what behaves well on
    this rig: the arms are driven through the native node, and walking to every
    action as fast as the reference deployment does (see
    :class:`ReachingChunkExecutor`) proved worse here in practice.
    """

    max_step_rad: float = DEFAULT_MAX_STEP_RAD
    max_effector_step: float = DEFAULT_MAX_EFFECTOR_STEP
    carry_targets: bool = False
    clamped: int = field(default=0, init=False)
    _targets: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.max_step_rad <= 0 or self.max_effector_step <= 0:
            raise ConfigurationError("chunk execution step limits must be positive")

    def reset(self, states: Mapping[str, ArmState]) -> None:
        """Re-seed the per-arm command target at the start of a chunk.

        The joints are re-seeded from the measured pose every time, so the
        bounded step is always taken from where the arm actually is. That also
        makes the command stream discontinuous once per chunk: an arm that is
        running behind its target has the target pulled back to it and then
        re-accelerates, a sawtooth in commanded velocity at every boundary, and
        on hardware that is the jerk the operator feels. ``carry_targets``
        keeps the integrator instead, which is smooth across the boundary and
        is the same argument the gripper already wins below -- it is off by
        default because re-seeding is also what bounds how far the command can
        march past an arm that has stopped against something.

        The gripper is not, after the first call. A gripper that is squeezing
        something *always* reads short of the value it was commanded -- that is
        what holding an object looks like -- so re-seeding the effector target
        from the measurement gives back a slice of the grip on every chunk
        boundary, and the arm re-closes the same slice on the ticks that
        follow. Worse, a gripper whose feedback is not tracking at all (a servo
        zeroed at the wrong stop reads pinned at one end) pins the commanded
        value with it, and the policy's gripper output stops reaching the
        hardware entirely while the joints keep moving normally. The commanded
        effector is this executor's own integrator; it carries forward.
        """
        previous = self._targets
        if self.carry_targets and previous is not None:
            # Nothing to re-seed: every value is this executor's own
            # integrator, so the next chunk continues from the last command.
            return
        self._targets = {}
        for name in MOLMOACT_ARM_NAMES:
            state = states[name]
            if state.effector is None:
                raise ConfigurationError(f"{name} must publish a YAM gripper state")
            held = (
                float(state.effector.position)
                if previous is None
                else float(previous[name][-1])
            )
            self._targets[name] = np.concatenate(
                (np.asarray(state.joints.position_rad, dtype=np.float64), [held])
            )

    def step(
        self,
        action: Sequence[float] | np.ndarray,
        *,
        limits: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> dict[str, PositionCommand]:
        if self._targets is None:
            raise ConfigurationError("chunk executor must be reset from measured state first")
        split = split_action(action)
        commands: dict[str, PositionCommand] = {}
        filed_down = False
        for name, goal in split.items():
            previous = self._targets[name]
            joint_goal = goal[:MOLMOACT_ARM_DOF]
            if limits is not None and name in limits:
                lower, upper = limits[name]
                joint_goal = np.clip(joint_goal, lower, upper)
            joint_move = joint_goal - previous[:MOLMOACT_ARM_DOF]
            bounded_move = np.clip(joint_move, -self.max_step_rad, self.max_step_rad)
            joint_target = previous[:MOLMOACT_ARM_DOF] + bounded_move
            effector_move = float(goal[-1] - previous[-1])
            bounded_effector = float(
                np.clip(effector_move, -self.max_effector_step, self.max_effector_step)
            )
            # Counted per action, not per value: the log line answers "how many
            # actions did the limit refuse", so one action the clamp touched
            # anywhere is one, however many of its 14 values it filed down.
            if not np.array_equal(bounded_move, joint_move) or bounded_effector != effector_move:
                filed_down = True
            effector_target = previous[-1] + bounded_effector
            effector_target = float(np.clip(effector_target, 0.0, 1.0))
            target = np.concatenate((joint_target, [effector_target]))
            self._targets[name] = target
            commands[name] = PositionCommand(joint_target, effector=effector_target)
        self.clamped += int(filed_down)
        return commands

    @property
    def targets(self) -> Mapping[str, np.ndarray]:
        return self._targets or {}


@dataclass(slots=True)
class ReachingChunkExecutor:
    """The reference deployment's executor: walk to every action in sub-steps.

    Opt-in (``--reach-actions``) rather than the default: it is one-to-one with
    the reference deployment, but on this rig it drives the native node far
    harder than :class:`BoundedChunkExecutor` does, and behaved worse.

    Each action is clamped to within ``max_joint_step_rad`` of the *measured*
    pose -- so one outlier cannot lunge the arm, and the clamp cannot drift
    away from where the arm actually is -- and then walked to in sub-steps.

    Reaching every action matters as much as bounding it. The checkpoint plans
    a 30-action chunk and expects all of it to run; a hardware A/B on the
    cloth-fold task, everything else equal, gave path/net travel 15.5 for full
    chunks against 26.2 for the first 8 actions of each. Commanding one clipped
    step per action and moving on is the same failure in a different disguise:
    the arm repeats the opening fragment of a reach and abandons it, so the
    part where the gripper arrives and closes never executes.
    """

    max_joint_step_rad: float = DEFAULT_MAX_JOINT_STEP_RAD
    max_effector_step: float = DEFAULT_MAX_JOINT_STEP_RAD
    sub_step_rad: float = SUB_STEP_RAD
    max_sub_steps: int = MAX_SUB_STEPS
    clamped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_joint_step_rad <= 0 or self.max_effector_step <= 0:
            raise ConfigurationError("chunk execution step limits must be positive")
        if self.sub_step_rad <= 0 or self.max_sub_steps < 1:
            raise ConfigurationError("chunk interpolation granularity must be positive")

    def plan(
        self,
        action: Sequence[float] | np.ndarray,
        states: Mapping[str, ArmState],
        *,
        limits: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> list[dict[str, PositionCommand]]:
        """The commands that take the arms from their measured pose to ``action``."""
        measured = measured_vector(states)
        target = self._bounded_target(action, measured, limits)
        return interpolate(
            measured,
            target,
            sub_step_rad=self.sub_step_rad,
            max_sub_steps=self.max_sub_steps,
        )

    def _bounded_target(
        self,
        action: Sequence[float] | np.ndarray,
        measured: np.ndarray,
        limits: Mapping[str, tuple[np.ndarray, np.ndarray]] | None,
    ) -> np.ndarray:
        goal = np.concatenate(
            [split_action(action)[name] for name in MOLMOACT_ARM_NAMES]
        )
        if limits is not None:
            # deviation: the reference deployment does not clip to joint limits.
            # Kept because this package knows the URDF's, and clipping an
            # in-range action is a no-op, so it only ever removes a command the
            # arm could not have honoured anyway.
            for index, name in enumerate(MOLMOACT_ARM_NAMES):
                bounds = limits.get(name)
                if bounds is None:
                    continue
                lower, upper = bounds
                block = slice(index * 7, index * 7 + MOLMOACT_ARM_DOF)
                goal[block] = np.clip(goal[block], lower, upper)
        step = np.empty(MOLMOACT_ACTION_DIM, dtype=np.float64)
        step[:MOLMOACT_ARM_DOF] = step[7 : 7 + MOLMOACT_ARM_DOF] = self.max_joint_step_rad
        step[MOLMOACT_ARM_DOF] = step[-1] = self.max_effector_step
        target = np.clip(goal, measured - step, measured + step)
        if not np.array_equal(target, goal):
            self.clamped += 1
        # deviation: PositionCommand requires a normalized effector, so the
        # gripper is clipped into range rather than rejected mid-chunk.
        target[MOLMOACT_ARM_DOF] = np.clip(target[MOLMOACT_ARM_DOF], 0.0, 1.0)
        target[-1] = np.clip(target[-1], 0.0, 1.0)
        return target
