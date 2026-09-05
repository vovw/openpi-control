# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`openpi-basic-control` drives robot arms (YAM, ARX, FR3, SO-101, Trossen) from
Python through a C++ control runtime. Each arm runs one `pi_control_node`
process; the Python process spawns it and talks to it over ZeroMQ. Everything
else — the operator CLI, the viser scene, the LeRobot recorder, the MolmoAct2
inference client — sits on top of that one boundary.

## Commands

```bash
uv sync                                            # installs and builds the native node
uv run --no-sync ruff check .                      # lint
uv run --no-sync pytest -q tests --ignore=tests/sil # the Python suite (~500 tests, seconds)
uv run --no-sync pytest tests/test_cli.py -k live   # one file / one selection
```

`uv sync` is an *exact* sync, and the `cell` dependency-group in
`pyproject.toml` is a default group, so a plain `uv sync` installs every extra a
live cell needs (dev, viz, cameras, inference, vr). Do not "fix" a missing
import by running `uv sync --extra <one>` — that uninstalls the rest. `lerobot`
is the only extra held back (it pulls torch): `uv sync --extra lerobot`.

The editable install carries a built `pi_control_node` at
`.venv/lib/python*/site-packages/openpi_control/bin/`. `OPENPI_CONTROL_NODE`
overrides which binary is used (see `native.native_executable`).

### Native C++

```bash
sudo ./scripts/install_build_deps_ubuntu.sh   # once: host packages
./scripts/build_deps.sh                       # once: pinned Pinocchio, ZMQ, Trossen, libfranka, libmodbus into .deps/

cmake -S . -B build-native -G Ninja -DOPENPI_CONTROL_BUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --target pi_control_tests pi_topic_zmq_tests
ctest --test-dir build-native --output-on-failure
ctest --test-dir build-native -R 'DriverCanMit.*'          # subset by ctest regex
./build-native/native/pi_control/tests/pi_control_tests --gtest_filter='MathOps.*'
```

`pi_control_tests` compiles only the sources it needs and requires neither
Pinocchio nor ZMQ, so it builds with `-DOPENPI_CONTROL_BUILD_NATIVE=OFF` — that
is how the valgrind CI job gets a test binary without building `.deps`.
Sanitizer builds come from `-DOPENPI_CONTROL_SANITIZER=address` (or
`undefined`, `thread`) applied globally; the fuzz harness needs Clang plus
`-DOPENPI_CONTROL_BUILD_FUZZERS=ON`. `.github/workflows/ci.yml` holds the exact
per-job test regexes — sanitizers and valgrind run *targeted* subsets, not the
whole suite. `native/pi_control/.clang-tidy` exists but its header comment is
stale: no CI job currently runs clang-tidy.

### SIL (real node, fake CAN servos)

```bash
sudo modprobe vcan && sudo ip link add dev vcan0 type vcan && sudo ip link set up vcan0
OPENPI_CONTROL_NODE=$PWD/build-native/native/pi_control/pi_control_node \
  OPENPI_SIL_VCAN=vcan0 OPENPI_SIL_VCAN2=vcan1 \
  uv run --no-sync pytest -q tests/sil
```

`tests/sil` skips itself unless `OPENPI_SIL_VCAN` names a live vcan interface.
Pair tests need a second one.

### Wheel

`uv build --wheel` → `dist/`. CI additionally checks the packaged node's
architecture and that `libpinocchio`/`libzmq` resolve out of the wheel's own
`libs/` directory.

## Architecture

### The process boundary

`protocol.py` is the ABI. Its `struct.Struct` formats mirror the C structs in
`native/pi_control/include/pi_topic_zmq.hpp` field for field, and
`PROTOCOL_VERSION` is checked in a handshake at connect. **A change on either
side is a change on both**, plus a version bump; the Python decode will
otherwise silently misread bytes. Topic names are session-scoped
(`openpi.<session_id>.<logical_name>.*`) so two sessions on one box cannot
cross-talk.

Python layering, innermost first:

- `types.py` — immutable, unit-explicit state/command dataclasses. Arrays are
  made read-only on construction; positions are rad, effort Nm, current A.
- `config.py` — connection kinds (`SocketCanConnection`, `EthernetConnection`,
  `SerialConnection`, `FR3Connection`, `RobotiqConnection`), `ArmConfig`, and
  packaged-asset resolution.
- `protocol.py` → `backend.py` (the `ArmBackend` ABC) → `native.py`
  (`NativeArmBackend`: spawns the node, owns the ZMQ sockets, reader thread,
  heartbeat, log tee).
- `arms.py` — `FollowerArm` / `LeaderArm`, the role-specific public handles.
- `session.py` — `ArmSession` owns several arms and makes connect/close
  all-or-nothing; `teleop.py` — `TeleopPair` adds alignment-gated bilateral
  engagement.
- `rigs.py` — a whole cell (which arms, which bus, base poses, which cameras).
  Pure config: resolving a rig opens nothing.

`ArmBackend` is the seam every hardware-free test goes through:
`tests/fake_arm_backend.py` stands in for the whole backend (used by the `live`,
record, and inference tests), while `tests/fake_native_node.py` is a real ZMQ
peer speaking the actual ABI (used by `test_native_backend.py`, with failure
modes selected by `FAKE_NODE_BEHAVIOR`). The Python CI job installs with
`OPENPI_CONTROL_BUILD_NATIVE=OFF`, so **no test outside `tests/sil` may need the
native binary**.

### Packaged model assets

`src/openpi_control/models/` is three layers, resolved by
`config.resolve_model_assets` and passed to the node as file paths:

- `arms/<Model>/<Model>.json` — model-wide: joint torque/velocity limits,
  gains, `servo_model` strings, catalog baud rate, default effector.
- `arms/<Model>/<Model>_01.json` — per *instance*: `zero_pos`, `home_pos`,
  `pos_min`/`pos_max`, reverse flags. This is the file `openpi-control zero`
  and per-robot calibration touch.
- `arms/<Model>/<Model>.urdf` — shipped for the gravity model (link inertias,
  joint origins), *not* for rendering. Meshes are fetched at runtime into
  `~/openpi-data/meshes/` by `meshes.py`.

Effectors mirror this under `effectors/<E_*>/` with an extra `_mass.json`.
`urdf_inertial.prepare_merged_urdf` *replaces* the arm URDF's `end_link`
inertial block with the effector's mass model (or zero mass when none is
attached) before handing the merged file to the node — that is what keeps the
gravity model from double-counting or missing the gripper.

`servos/__init__.SERVO_ZERO_DRIVERS` maps the *exact* `servo_model` string from
those JSONs to a per-family zeroing driver. There is deliberately no
translation layer: a new servo family means the same canonical string in the
model JSON and in that table.

### Native runtime

`native/pi_control/` is a Device / Driver / Servo hierarchy: `Device` (→
`DeviceArm` → `DeviceArmCan`/`DeviceArmSerial`, plus `DeviceFR3` and
`DeviceEffector*`), `Driver` (→ `DriverCan` → `DriverCanMit` →
`DriverArxEncoder`, `DriverSerial` → `DriverFt`, `DriverController` →
`DriverTrossen`, `DriverFR3`), and `Servo` (→ `ServoDm`, `ServoFt`,
`ServoController`, `ServoCanPassiveEncoder`), wired together from the model
JSON by `pi_device_config`. `pi_control.hpp` holds the safety
constants (temperature derating ramp, velocity-bounded move-to-ready speeds,
position-difference thresholds) with the rationale in comments next to each.
`pi_topic_zmq` is the transport, `pi_algo_pino` the Pinocchio gravity model.

### Safety and lifecycle invariants

These are load-bearing; several tests exist only to protect them.

- **An arm cannot outlive the process that energized it.** The node holds a
  parent-liveness pipe fd (`--parent_liveness_fd`) and SIGTERMs itself when it
  closes. There is no `up`/`down` pair of commands — one foreground process owns
  the whole arc and ctrl-c is the way out.
- **Only `live`, `infer`, and `record` energize arms.** `doctor` and `cameras`
  are read-only (`--probe` reads, never writes); `zero` writes servo firmware
  and confirms first; `viz.py` opens no bus at all.
- **Park before de-energizing.** Exit paths park at the instance JSON's
  `home_pos` rather than dropping the arm where it stands.
- **Teardown always reaches every arm.** `ArmSession.close` gives each arm a
  `close()` attempt and re-raises the first error afterwards; a raising
  `disengage` must not orphan an energized node holding its ZMQ ports.
- **Draw and command are separate modules.** `viz.py` only renders (safe
  against a dead cell); `viser_control.py` is the half that can move an arm, and
  its arming gate refuses stale, missing, or disagreeing state.
- **Gripper polarity is inverted between worlds.** This package uses `1.0` =
  open; LeRobot datasets and the Quest trigger use `0.0` = open.
  `record.to_native_gripper` / `to_dataset_gripper` are the only two places that
  flip, and they are named for their direction.

### Inference

`inference.py` is a thin MolmoAct2 `/act` HTTP client (14-D bimanual state and
action, three RGB views, `json_numpy` wire codec). Wire-level defaults
deliberately match the reference deployment; *execution* semantics deliberately
do not — the default is one bounded command per control tick, and
`--reach-actions`, `--prefetch`, `--reset-start-pose` switch on the reference
runtime piece by piece. `docs/inference.md` has the table of what differs and
why. Do not "align with the reference" here without reading it.

## Conventions

- Comments and module docstrings carry the *why*, and cite the measurement when
  a default came from one (frame rates, encode costs, thresholds). Match that
  density; a constant that came from hardware should say so.
- `docs/*.md` is where rationale that outgrows a docstring lives, and README
  links each one. A behavioral change usually means a docs edit too.
- Ruff: line length 100, `select = ["E", "F", "I", "UP", "B"]`, target py311.
  Python is `>=3.11,<3.14` (trossen-arm wheel availability).
- Commits: short imperative subject, then a prose body explaining the reasoning
  and any measurements behind the change.
- Logs land under `~/openpi-data/logs/` (`OPENPI_LOG_DIR` overrides); every CLI
  command attaches `runlog.setup_run_logging`, which also captures native
  crashes.
