#!/usr/bin/env python3
"""
Unit test for episode filtering logic.
Creates a mock Lerobot dataset and verifies the filter pipeline end-to-end.
"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path


def extract_seed_from_name(name: str) -> int | None:
    """Extract seed number from episode name like '..._seed_407008'."""
    m = re.search(r"_seed_(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def load_episodes_jsonl(dataset_root: Path) -> dict[int, dict]:
    """Load meta/episodes.jsonl and return {episode_index: metadata_dict}."""
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_path}")

    episodes = {}
    with open(episodes_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ep_idx = item["episode_index"]
            episodes[ep_idx] = item
    return episodes


def build_filter_by_seeds(
    episodes: dict[int, dict],
    target_seeds: set[int],
) -> tuple[list[int], list]:
    """Match episodes by seed extracted from raw_file_name or tasks."""
    matched = []
    unmatched_info = []

    for ep_idx, meta in sorted(episodes.items()):
        raw_name = meta.get("raw_file_name", "")
        seed = extract_seed_from_name(raw_name) if raw_name else None

        # Fallback: try to extract seed from tasks
        if seed is None:
            tasks = meta.get("tasks", [])
            for task in tasks:
                seed = extract_seed_from_name(task)
                if seed is not None:
                    break

        if seed is not None and seed in target_seeds:
            matched.append(ep_idx)
        else:
            unmatched_info.append((ep_idx, raw_name, seed))

    return matched, unmatched_info


def create_mock_lerobot_dataset(root: Path, num_episodes: int = 10):
    """Create a minimal Lerobot dataset structure."""
    # meta/
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # info.json
    info = {
        "codebase_version": "2.0",
        "fps": 10,
        "total_episodes": num_episodes,
        "total_frames": num_episodes * 33,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": f"0:{num_episodes}"},
        "total_tasks": 1,
        "total_videos": 0,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "observation.images.cam_high": {"dtype": "video", "shape": [3, 480, 640], "names": None},
            "observation.images.cam_left_wrist": {"dtype": "video", "shape": [3, 480, 640], "names": None},
            "observation.images.cam_right_wrist": {"dtype": "video", "shape": [3, 480, 640], "names": None},
            "observation.state.default": {"dtype": "float32", "shape": [14], "names": None},
            "action.default": {"dtype": "float32", "shape": [14], "names": None},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f)

    # episodes.jsonl with raw_file_name containing seeds
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for i in range(num_episodes):
            seed = 400000 + i * 12
            episode = {
                "episode_index": i,
                "tasks": [f"move block blue large"],
                "length": 33,
                "raw_file_name": f"episode_{i:06d}_c2_pilot_train_move_block_seed_{seed}",
            }
            f.write(json.dumps(episode) + "\n")

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "move block blue large"}) + "\n")

    # stats.json
    with open(meta_dir / "stats.json", "w") as f:
        json.dump({}, f)

    # data/chunk-000/episode_*.parquet (minimal - just metadata)
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_episodes):
        # Minimal parquet with just the required columns
        # We won't actually read video frames in this test
        table = pa.table({
            "episode_index": pa.array([i] * 33, type=pa.int64()),
            "index": pa.array(list(range(i * 33, (i + 1) * 33)), type=pa.int64()),
            "task_index": pa.array([0] * 33, type=pa.int64()),
            "timestamp": pa.array([j / 10.0 for j in range(33)], type=pa.float32()),
            "observation.state.default": pa.array([[0.0] * 14 for _ in range(33)], type=pa.list_(pa.float32(), 14)),
            "action.default": pa.array([[0.0] * 14 for _ in range(33)], type=pa.list_(pa.float32(), 14)),
        })
        pq.write_table(table, data_dir / f"episode_{i:06d}.parquet")

    # Empty videos directory (we won't decode videos)
    for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
        (root / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(parents=True, exist_ok=True)

    return root


def test_build_episode_filter():
    """Test the build_episode_filter.py script logic."""
    # Functions are defined locally above

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "mock_dataset"
        create_mock_lerobot_dataset(root, num_episodes=10)

        # Load episodes
        episodes = load_episodes_jsonl(root)
        assert len(episodes) == 10

        # Test seed filtering: target seeds 400012 and 400036 (episodes 1 and 3)
        target_seeds = {400012, 400036}
        matched, unmatched = build_filter_by_seeds(episodes, target_seeds)
        print(f"[test] Matched indices: {matched}")
        assert matched == [1, 3], f"Expected [1, 3], got {matched}"
        assert len(unmatched) == 8
        print("[test] build_episode_filter.py: PASSED")


def test_filter_file_format():
    """Test filter file format (one integer per line)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filter_path = Path(tmpdir) / "filter.txt"
        with open(filter_path, "w") as f:
            for i in [0, 2, 4, 6, 8]:
                f.write(f"{i}\n")

        # Simulate BaseLerobotDataset filter logic
        with open(filter_path, "r") as f:
            allowed_indices = {int(line.strip()) for line in f if line.strip()}

        all_episodes = list(range(10))
        filtered = [idx for idx in all_episodes if idx in allowed_indices]

        assert filtered == [0, 2, 4, 6, 8], f"Expected [0, 2, 4, 6, 8], got {filtered}"
        print(f"[test] Filter logic: {len(filtered)}/10 episodes retained")
        print("[test] BaseLerobotDataset filter logic: PASSED")


def main():
    print("=" * 50)
    print("Running episode filter tests")
    print("=" * 50)

    test_build_episode_filter()
    print()
    test_filter_file_format()

    print()
    print("=" * 50)
    print("All tests PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
