# Viser commands

## View models without connecting hardware

```bash
# Download YAM rendering meshes to the local cache.
uv run --no-sync openpi-control-viz --fetch-meshes --model Yam

# View one arm with joint sliders.
uv run --no-sync openpi-control-viz --model Yam --effector E_Yam

# View both arms in their configured base poses.
uv run --no-sync openpi-control-viz --rig yam_bimanual

# List available models or rigs.
uv run --no-sync openpi-control-viz --list
uv run --no-sync openpi-control-viz --list-rigs
```

Open the printed URL, normally `http://ROBOT_HOST:8080`.
`--port 8081` changes the port; `--mesh-dir PATH` selects local meshes.
Downloaded meshes are stored under `~/openpi-data/meshes/`.

## View or control live hardware

```bash
# Power the arms and display measured joint states plus camera previews.
uv run --no-sync openpi-control live

# Add motion sliders; confirm the measured pose and click Arm first.
uv run --no-sync openpi-control live --control

# Show policy execution, predicted paths, and live camera previews.
uv run --no-sync openpi-control infer \
    --server http://POLICY_HOST:8202 --instruction "fold the towel"
```

Ctrl-C parks and powers down live hardware. The standalone
`openpi-control-viz` viewer only changes the rendered pose.

## VR preview

After [VR setup](vr.md), run:

```bash
# Drive the virtual arms from Quest controllers.
uv run --no-sync openpi-control teleop --backend sim
```

This is an IK preview without contact physics. Gripper values appear in a
panel; the packaged URDF fingers are fixed.
