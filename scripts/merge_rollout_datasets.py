#!/usr/bin/env python3
"""Merge local OpenPI-control LeRobot v3 rollout datasets.

The rollout recorder writes one LeRobot dataset per experiment. This utility
keeps each source data/video file intact, rewrites the global indices, and
creates one LeRobot dataset with a per-episode provenance sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OBJECT_BY_EPISODE = {
    0: "pink towel",
    1: "blue towel",
    2: "black T-shirt",
}
NUMERIC_FEATURES = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/yambox/openpi-data/rollouts"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    return parser.parse_args()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def set_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    field = table.schema.field(name)
    return table.set_column(index, field, pa.array(values, type=field.type))


def shift_episode_stats(row: dict[str, object], prefix: str, offset: int) -> None:
    for statistic in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
        key = f"stats/{prefix}/{statistic}"
        if key in row:
            row[key] = [float(value) + offset for value in row[key]]  # type: ignore[index]


def scalar_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def aggregate_image_stats(source_stats: list[dict[str, object]]) -> dict[str, object]:
    means = [np.asarray(stats["mean"], dtype=np.float64).reshape(-1) for stats in source_stats]
    stds = [np.asarray(stats["std"], dtype=np.float64).reshape(-1) for stats in source_stats]
    mins = [np.asarray(stats["min"], dtype=np.float64).reshape(-1) for stats in source_stats]
    maxes = [np.asarray(stats["max"], dtype=np.float64).reshape(-1) for stats in source_stats]
    quantiles = {
        name: [
            np.asarray(stats[name], dtype=np.float64).reshape(-1)
            for stats in source_stats
        ]
        for name in ("q01", "q10", "q50", "q90", "q99")
    }
    weights = np.asarray(
        [int(np.asarray(stats["count"]).reshape(-1)[0]) for stats in source_stats],
        dtype=np.float64,
    )
    total = float(np.sum(weights))

    mean = np.sum(np.stack(means) * weights[:, None], axis=0) / total
    second_moment = np.sum(
        (np.stack(stds) ** 2 + np.stack(means) ** 2) * weights[:, None],
        axis=0,
    ) / total
    std = np.sqrt(np.maximum(second_moment - mean**2, 0.0))

    def reshape_like(values: np.ndarray) -> object:
        return values.reshape(np.asarray(source_stats[0]["mean"]).shape).tolist()

    result: dict[str, object] = {
        "min": reshape_like(np.min(np.stack(mins), axis=0)),
        "max": reshape_like(np.max(np.stack(maxes), axis=0)),
        "mean": reshape_like(mean),
        "std": reshape_like(std),
        "count": [int(total)],
    }
    for name, values in quantiles.items():
        # This is a weighted approximation. Decoding all video frames merely
        # to recompute pixel quantiles would make the merge needlessly costly.
        result[name] = reshape_like(
            np.sum(np.stack(values) * weights[:, None], axis=0) / total
        )
    return result


def build_readme(
    *,
    repo_id: str,
    source_rows: list[dict[str, object]],
    total_frames: int,
    total_episodes: int,
    fps: int,
) -> str:
    lines = [
        "---",
        "pretty_name: OpenPI MolmoAct2 fold-towel rollout ablation",
        "library_name: lerobot",
        "tags:",
        "- robotics",
        "- robot-learning",
        "- lerobot",
        "- openpi-control",
        "task_categories:",
        "- robotics",
        "---",
        "",
        "# OpenPI MolmoAct2 fold-towel rollout ablation",
        "",
        f"Merged LeRobot v3 dataset: `{repo_id}`.",
        "",
        "This dataset contains the hardware inference rollouts collected with "
        "`openpi-control rollout` on the bimanual YAM cell. Every source run "
        "contributes all saved episodes, including partial episodes saved after "
        "Ctrl-C. The `y/n` labels are the operator's task-success labels.",
        "",
        "## Contents",
        "",
        f"- {total_episodes} episodes and {total_frames:,} frames",
        f"- Nominal recording rate: {fps} FPS",
        "- Three video views per frame: top, left wrist, right wrist",
        "- State/action vector: 14 values (6 joints + gripper per arm)",
        "- Canonical LeRobot metadata under `meta/` and trajectory data under `data/`",
        "- Per-episode configuration/provenance: `meta/rollout_metadata.jsonl`",
        "- Original recorder manifests: `provenance/`",
        "",
        "## Configurations tested",
        "",
        "The effective duration is computed from the implementation's interpolation "
        "rule: `ceil((chunk_size - 1) / speed) + 1` control ticks.",
        "",
        "| Source run | Speed | Chunk size | Approx. chunk duration | Episodes | Success labels |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in source_rows:
        speed = float(row["speed"])
        chunk = int(row["chunk_size"])
        ticks = math.ceil((chunk - 1) / speed) + 1
        labels = str(row["labels"])
        lines.append(
            f"| `{row['source_directory']}` | {speed:g} | {chunk} | "
            f"{ticks / fps:.2f} s | {row['episodes']} | `{labels}` |"
        )
    lines += [
        "",
        "### Episode mapping",
        "",
        "The original recorder stores the natural-language prompt, but not the "
        "towel color. The object names below follow the operator's episode-order "
        "annotation for these runs:",
        "",
        "| Local episode slot | Object annotation | Prompt in dataset |",
        "|---:|---|---|",
        "| 1 | pink towel | `fold the towel` |",
        "| 2 | blue towel | `fold the towel` |",
        "| 3 | black T-shirt | `fold the tshirt` |",
        "",
        "## Important notes",
        "",
        "- The original source manifests report all 21 episodes as unsuccessful (`n`).",
        "- The object annotation is provenance supplied by the operator; it is not "
        "part of the original per-frame task string.",
        "- The directory `fold-towel-speed10-chunk15` is named as chunk 15, but its "
        "source manifest records `speed=1.0` and `chunk_size=30`; the manifest is "
        "treated as the ground truth.",
        "- Dataset timestamps use the declared 30 FPS. The saved episodes are about "
        "110–115 nominal seconds each even when the wall-clock limit was 120 seconds.",
        "- The rollout runtime used the PyAV video backend. This dataset preserves "
        "the recorded videos and does not recompress them.",
        "",
        "## Loading",
        "",
        "The dataset is structured as a standard LeRobot v3 dataset. The sidecar "
        "configuration can be joined by `global_episode_index`:",
        "",
        "```python",
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset",
        "",
        f'dataset = LeRobotDataset("{repo_id}")',
        "```",
        "",
        "The trajectory rows retain the original prompt in LeRobot's `task` field. "
        "Use `meta/rollout_metadata.jsonl` when filtering by speed, chunk size, "
        "object annotation, success label, or source run.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists():
        if any(output.iterdir()):
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    else:
        output.mkdir(parents=True)

    source_dirs = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and (path / "openpi_control_rollouts.json").is_file()
    )
    if not source_dirs:
        raise SystemExit(f"no rollout datasets found under {source_root}")

    first_info = json.loads((source_dirs[0] / "meta/info.json").read_text())
    fps = int(first_info["fps"])
    video_keys = [
        name
        for name, feature in first_info["features"].items()
        if feature.get("dtype") == "video"
    ]
    first_tasks = pq.read_table(source_dirs[0] / "meta/tasks.parquet")
    task_rows = first_tasks.to_pylist()

    (output / "data/chunk-000").mkdir(parents=True)
    (output / "meta/episodes/chunk-000").mkdir(parents=True)
    (output / "videos").mkdir(parents=True)
    (output / "provenance").mkdir(parents=True)

    episode_rows: list[dict[str, object]] = []
    rollout_metadata: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    numeric_values: dict[str, list[np.ndarray]] = {name: [] for name in NUMERIC_FEATURES}
    image_stats: dict[str, list[dict[str, object]]] = {name: [] for name in video_keys}
    global_frame_offset = 0
    global_episode_offset = 0

    for source_index, source_dir in enumerate(source_dirs):
        manifest = json.loads((source_dir / "openpi_control_rollouts.json").read_text())
        info = json.loads((source_dir / "meta/info.json").read_text())
        if int(info["fps"]) != fps:
            raise SystemExit(f"FPS mismatch in {source_dir}")

        source_manifest_path = output / "provenance" / f"{source_index:02d}-{source_dir.name}.json"
        json_dump(source_manifest_path, manifest)

        data_path = source_dir / "data/chunk-000/file-000.parquet"
        data_table = pq.read_table(data_path)
        source_episode_table = pq.read_table(
            source_dir / "meta/episodes/chunk-000/file-000.parquet"
        )
        source_episode_rows = source_episode_table.to_pylist()
        manifest_episodes = {
            int(entry["episode_index"]): entry for entry in manifest["episodes"]
        }
        expected_source_frames = sum(int(row["length"]) for row in source_episode_rows)
        if expected_source_frames != data_table.num_rows:
            raise SystemExit(
                f"episode/data frame mismatch in {source_dir}: "
                f"{expected_source_frames} != {data_table.num_rows}"
            )

        episode_count = len(source_episode_rows)
        speed = float(manifest["speed"])
        chunk_size = int(manifest["chunk_size"])
        labels = "".join(
            str(manifest_episodes[int(row["episode_index"])].get("label", "?"))
            for row in source_episode_rows
        )
        source_rows.append(
            {
                "source_directory": source_dir.name,
                "source_repo_id": manifest.get("repo_id"),
                "speed": speed,
                "chunk_size": chunk_size,
                "episodes": episode_count,
                "labels": labels,
            }
        )

        shifted = set_column(
            data_table,
            "episode_index",
            np.asarray(data_table["episode_index"].to_numpy()) + global_episode_offset,
        )
        shifted = set_column(
            shifted,
            "index",
            np.asarray(shifted["index"].to_numpy()) + global_frame_offset,
        )
        output_data_path = output / f"data/chunk-000/file-{source_index:03d}.parquet"
        pq.write_table(shifted, output_data_path)

        for name in NUMERIC_FEATURES:
            values = np.asarray(shifted[name].to_pylist(), dtype=np.float64)
            if values.ndim == 1:
                values = values[:, None]
            numeric_values[name].append(values)

        source_stats = json.loads((source_dir / "meta/stats.json").read_text())
        for name in video_keys:
            image_stats[name].append(source_stats[name])

        for name in video_keys:
            source_video = source_dir / f"videos/{name}/chunk-000/file-000.mp4"
            target_video = output / f"videos/{name}/chunk-000/file-{source_index:03d}.mp4"
            target_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, target_video)

        for source_episode_row in source_episode_rows:
            local_ep = int(source_episode_row["episode_index"])
            row = dict(source_episode_row)
            row["episode_index"] = local_ep + global_episode_offset
            row["data/file_index"] = source_index
            row["dataset_from_index"] = int(row["dataset_from_index"]) + global_frame_offset
            row["dataset_to_index"] = int(row["dataset_to_index"]) + global_frame_offset
            for name in video_keys:
                row[f"videos/{name}/file_index"] = source_index
            shift_episode_stats(row, "episode_index", global_episode_offset)
            shift_episode_stats(row, "index", global_frame_offset)
            episode_rows.append(row)

            manifest_entry = manifest_episodes[local_ep]
            task_prompt = str(manifest_entry.get("prompt", row["tasks"][0]))
            metadata = {
                "global_episode_index": local_ep + global_episode_offset,
                "source_episode_index": local_ep,
                "source_directory": source_dir.name,
                "source_repo_id": manifest.get("repo_id"),
                "speed": speed,
                "chunk_size": chunk_size,
                "episode_seconds_limit": float(manifest.get("episode_seconds", 120.0)),
                "fps": fps,
                "object_annotation": OBJECT_BY_EPISODE.get(
                    local_ep, f"episode slot {local_ep + 1}"
                ),
                "prompt": task_prompt,
                "success": bool(manifest_entry.get("success", False)),
                "label": manifest_entry.get("label"),
                "saved": bool(manifest_entry.get("saved", True)),
                "aborted": bool(manifest_entry.get("aborted", False)),
                "length": int(row["length"]),
                "stored_duration_s": (int(row["length"]) - 1) / fps,
            }
            rollout_metadata.append(metadata)

        global_frame_offset += data_table.num_rows
        global_episode_offset += episode_count

    # The source task table is identical across the runs: the prompt remains the
    # canonical LeRobot task label, while configuration is in the sidecar.
    pq.write_table(first_tasks, output / "meta/tasks.parquet")

    episode_schema = source_episode_table.schema
    pq.write_table(
        pa.Table.from_pylist(episode_rows, schema=episode_schema),
        output / "meta/episodes/chunk-000/file-000.parquet",
    )

    merged_stats: dict[str, object] = {}
    for name, arrays in numeric_values.items():
        merged_stats[name] = scalar_stats(np.concatenate(arrays, axis=0))
    for name, stats in image_stats.items():
        merged_stats[name] = aggregate_image_stats(stats)
    json_dump(output / "meta/stats.json", merged_stats)

    merged_info = dict(first_info)
    merged_info.update(
        {
            "total_episodes": len(episode_rows),
            "total_frames": global_frame_offset,
            "total_tasks": len(task_rows),
            "data_files_size_in_mb": round(
                sum(path.stat().st_size for path in (output / "data").rglob("*.parquet"))
                / 1_000_000,
                2,
            ),
            "video_files_size_in_mb": round(
                sum(path.stat().st_size for path in (output / "videos").rglob("*.mp4"))
                / 1_000_000,
                2,
            ),
            "splits": {"train": f"0:{len(episode_rows)}"},
            "repo_id": args.repo_id,
            "rollout_metadata_path": "meta/rollout_metadata.jsonl",
        }
    )
    json_dump(output / "meta/info.json", merged_info)

    with (output / "meta/rollout_metadata.jsonl").open("w") as handle:
        for metadata in rollout_metadata:
            handle.write(json.dumps(metadata) + "\n")

    combined_manifest = {
        "format": "openpi-control.merged-inference-rollout-v1",
        "dataset_format": "LeRobot v3.0",
        "repo_id": args.repo_id,
        "source_dataset_count": len(source_dirs),
        "total_episodes": len(episode_rows),
        "total_frames": global_frame_offset,
        "fps": fps,
        "partial_episodes_saved_on_ctrl_c": all(
            bool(json.loads((path / "openpi_control_rollouts.json").read_text()).get(
                "partial_episodes_saved_on_ctrl_c", False
            ))
            for path in source_dirs
        ),
        "source_datasets": source_rows,
        "episodes": rollout_metadata,
    }
    json_dump(output / "openpi_control_rollouts.json", combined_manifest)
    (output / ".gitattributes").write_text(
        "*.mp4 filter=lfs diff=lfs merge=lfs -text\n"
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
    )
    (output / "README.md").write_text(
        build_readme(
            repo_id=args.repo_id,
            source_rows=source_rows,
            total_frames=global_frame_offset,
            total_episodes=len(episode_rows),
            fps=fps,
        )
    )

    print(f"merged {len(source_dirs)} source datasets")
    print(f"episodes: {len(episode_rows)}")
    print(f"frames: {global_frame_offset}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
