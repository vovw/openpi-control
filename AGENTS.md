# Repository instructions

## Scope and workflow

- Keep user documentation command-first: a runnable command and a short explanation.
- Update affected docs when commands or behavior change. Avoid duplicated walkthroughs.
- Use `external/vr-teleop-kit`, a pinned Git submodule, for the VR adapter and relay.
  Do not edit its contents or advance its revision unless the task requires it.
- Keep logs, datasets, build artifacts, and local environment files out of commits.
- Run `uv run --no-sync ruff check .` and relevant Python tests after code changes.
  The full Python check is `uv run --no-sync pytest -q tests --ignore=tests/sil`.
- Native changes need the corresponding C++ tests. Build and SIL commands are in
  `docs/development.md`. Do not run hardware commands merely to validate docs.

## Environment

`uv sync` installs the default `cell` group: development, visualization, cameras,
inference, and VR runtime dependencies. `uv sync --extra lerobot` adds recording
and requires Python 3.12 or 3.13. Use `uv run --no-sync` for checks in an already
prepared environment. The VR relay has its own environment in the submodule.

## Code map

- `src/openpi_control/types.py`: immutable states and commands; radians, Nm, amperes.
- `config.py`, `rigs.py`: connections, model assets, and cell configuration.
- `protocol.py`, `native.py`: binary protocol, node process, sockets, and heartbeat.
- `arms.py`, `session.py`: arm handles and multi-arm lifecycle.
- `cli.py`: operator commands; `viser_control.py`: browser motion controls.
- `viz.py`: rendering; `cameras.py`: capture; `record.py`: LeRobot episodes.
- `inference.py`, `inference_record.py`: policy transport, execution, and recording.
- `teleop_vr.py`, `teleop_hardware.py`, `teleop_sim.py`: Quest adapter and runners.
- `android/quest-streamer`: native OpenXR controller input app.
- `native/pi_control`: C++ control runtime, drivers, and tests.

## Invariants

- Python protocol structs must match `native/pi_control/include/pi_topic_zmq.hpp`.
  Update both ends and the protocol version for ABI changes.
- Each energized arm belongs to one foreground owner. Preserve parent-liveness
  shutdown, stale-state checks, bounded commands, and parking on exit.
- Teardown must attempt to close every arm even when another close fails.
- `doctor` and camera discovery are read-only. `zero` writes calibration.
  `live`, hardware `teleop`, `infer`, `rollout`, and `record` can power motors.
  `record --dry-run` suppresses dataset writes, not hardware operation.
- Keep rendering separate from hardware commands. Simulation must not create
  a hardware session.
- Native grippers use 1 = open; datasets and Quest triggers use 0 = open.
  Preserve conversion at the recorder/adapter boundary.
- Tests outside `tests/sil` must work without a built native binary. Use
  `tests/fake_arm_backend.py` or `tests/fake_native_node.py` as appropriate.
- Preserve per-instance calibration in model `_01.json` files. Canonical
  `servo_model` strings must match the zero-driver registry.
