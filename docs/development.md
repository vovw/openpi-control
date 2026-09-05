# Build and test

## Build the runtime

```bash
# Install Ubuntu system dependencies and pinned native libraries.
sudo ./scripts/install_build_deps_ubuntu.sh
./scripts/build_deps.sh

# Build and install the editable package with operator dependencies.
uv sync

# Build a distributable wheel in dist/.
uv build --wheel
```

## Python checks

```bash
# Lint project code.
uv run --no-sync ruff check .

# Run tests that do not require real hardware or the native runtime.
uv run --no-sync pytest -q tests --ignore=tests/sil

# Run a selected test file.
uv run --no-sync pytest -q tests/test_teleop_vr.py
```

## Native checks

```bash
# Configure and build the native test binaries.
cmake -S . -B build-native -DOPENPI_CONTROL_BUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --target pi_control_tests pi_topic_zmq_tests

# Run the native suite.
ctest --test-dir build-native --output-on-failure
```

## Software-in-the-loop tests

```bash
# Create virtual CAN buses once per host boot.
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link add dev vcan1 type vcan
sudo ip link set up vcan0
sudo ip link set up vcan1

# Run the real node against fake CAN servos.
OPENPI_CONTROL_NODE="$PWD/build-native/native/pi_control/pi_control_node" \
OPENPI_SIL_VCAN=vcan0 OPENPI_SIL_VCAN2=vcan1 \
uv run --no-sync pytest -q tests/sil
```

## Update the VR checkout

```bash
# Restore the revision pinned by this repository.
git submodule update --init --recursive

# Inspect the pinned commit.
git submodule status
```
