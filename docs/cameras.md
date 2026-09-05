# Camera commands

`uv sync` installs the RealSense capture and image-encoding dependencies.

```bash
# List configured cameras and their detected device paths.
uv run --no-sync openpi-control cameras

# Open every stream and report captured frame information.
uv run --no-sync openpi-control cameras --probe

# Save one PNG per camera to /tmp/cameras.
uv run --no-sync openpi-control cameras --probe --snapshot /tmp/cameras

# Select the top and right wrist cameras only.
uv run --no-sync openpi-control cameras --only right

# Override the top camera's device path.
uv run --no-sync openpi-control cameras --camera top=/dev/video4 --probe

# Show camera previews alongside real arms in Viser; powers the arms.
uv run --no-sync openpi-control live

# Power arms without opening cameras.
uv run --no-sync openpi-control live --no-cameras
```

The packaged rig identifies `top`, `left_wrist`, and `right_wrist` by serial
number. Check the saved views before recording or inference. Only one process
should capture each camera at a time.

`--camera NAME=DEVICE` is repeatable and also works with `doctor`, `live`,
`record`, `infer`, and `rollout`. Edit the rig in
[`rigs.py`](../src/openpi_control/rigs.py) for permanent camera assignments.
