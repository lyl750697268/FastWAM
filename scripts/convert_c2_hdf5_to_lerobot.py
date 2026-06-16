#!/usr/bin/env python3
"""
Convert RoboTwin C2 HDF5 data to FastWAM Lerobot format.

Reads raw HDF5 episodes from /mnt/data/mixiangju/nas_data/Data/RoboTwin/C2-v0/,
extracts 3-camera video + action/state, and writes Lerobot-compatible dataset.

Usage:
    python scripts/convert_c2_hdf5_to_lerobot.py \
        --raw-root /mnt/data/mixiangju/nas_data/Data/RoboTwin/C2-v0 \
        --seed-list data/c2_train_seeds.txt \
        --output-dir data/robotwin2.0_c2/train \
        --action-mode 16d_full

    python scripts/convert_c2_hdf5_to_lerobot.py \
        --raw-root /mnt/data/mixiangju/nas_data/Data/RoboTwin/C2-v0 \
        --seed-list data/c2_test_seeds.txt \
        --output-dir data/robotwin2.0_c2/test \
        --action-mode 16d_full
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


CAMERA_KEYS = ["head_camera", "left_camera", "right_camera"]
CAMERA_KEY_MAP = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}
FPS = 10  # RoboTwin default


def decode_jpeg_bytes(jpeg_bytes):
    """Decode JPEG bytes to RGB numpy array [H, W, 3]."""
    if isinstance(jpeg_bytes, bytes):
        img = Image.open(io.BytesIO(jpeg_bytes))
    else:
        img = Image.open(io.BytesIO(bytes(jpeg_bytes)))
    img = img.convert("RGB")
    return np.array(img)


def extract_seed_from_dirname(dirname: str) -> int | None:
    m = re.search(r"_seed_(\d+)", dirname)
    return int(m.group(1)) if m else None


def discover_episodes(raw_root: Path, split: str) -> list[Path]:
    split_root = raw_root / split
    if not split_root.exists():
        raise FileNotFoundError(split_root)
    episodes = []
    for task_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        for ep_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            episodes.append(ep_dir)
    return episodes


def filter_episodes_by_seed(episodes: list[Path], seed_set: set[int]) -> list[Path]:
    matched = []
    for ep_dir in episodes:
        seed = extract_seed_from_dirname(ep_dir.name)
        if seed is not None and seed in seed_set:
            matched.append(ep_dir)
    return matched


def load_scene_instruction(episode_dir: Path) -> str:
    scene_path = episode_dir / "scene_info.json"
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    meta = scene.get("episode_0", {}).get("c2_metadata", {})
    return meta.get("instruction", "")


def encode_video_ffmpeg(frames: list[np.ndarray], out_path: Path, fps: int = FPS) -> None:
    """Encode frames to MP4 using ffmpeg. Writes to a local temp file first, then moves to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]

    # Use local temp file to avoid network/fsync issues during encoding
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="fwam_vid_")
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        str(tmp_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(frame.astype(np.uint8).tobytes())
        stderr, _ = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode()[:500]}")
    except Exception:
        proc.kill()
        tmp_path.unlink(missing_ok=True)
        raise

    # Quick sanity check before moving
    if tmp_path.stat().st_size < 1024:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg produced empty/invalid file: {tmp_path}")

    # Validate mp4 container
    import av
    try:
        container = av.open(str(tmp_path))
        container.close()
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg output is not a valid MP4: {e}")

    shutil.move(str(tmp_path), str(out_path))


def build_action(hdf5_file: h5py.File, mode: str) -> np.ndarray:
    """Extract and format action vector from HDF5."""
    vector = hdf5_file["joint_action/vector"][()]  # [T, 16]
    if mode == "16d_full":
        return vector.astype(np.float32)
    elif mode == "14d_no_gripper":
        # left_arm(7) + right_arm(7)
        left_arm = vector[:, :7]
        right_arm = vector[:, 8:15]
        return np.concatenate([left_arm, right_arm], axis=1).astype(np.float32)
    elif mode == "14d_left7_right6_gripper1":
        # left_arm(7) + right_arm(6) + right_gripper(1) — speculative
        left_arm = vector[:, :7]
        right_arm = vector[:, 8:14]
        right_grip = vector[:, 15:16]
        return np.concatenate([left_arm, right_arm, right_grip], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown action mode: {mode}")


def build_state(hdf5_file: h5py.File, mode: str) -> np.ndarray:
    """Extract and format state (proprioception) from HDF5."""
    # Use endpose as state — [left_endpose(7), right_endpose(7)] = 14D
    left = hdf5_file["endpose/left_endpose"][()]  # [T, 7]
    right = hdf5_file["endpose/right_endpose"][()]  # [T, 7]
    state = np.concatenate([left, right], axis=1).astype(np.float32)
    if mode == "16d_full":
        # Pad or use joint state instead
        # For 16D, use full joint vector as both action and state
        return hdf5_file["joint_action/vector"][()].astype(np.float32)
    return state


def convert_episode(
    episode_dir: Path,
    episode_index: int,
    task_index: int,
    action_mode: str,
    output_root: Path,
    chunk_size: int = 1000,
) -> dict:
    """Convert a single episode to Lerobot format. Returns episode metadata."""
    hdf5_path = episode_dir / "data" / "episode0.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        n_frames = f["joint_action/vector"].shape[0]
        instruction = load_scene_instruction(episode_dir)

        # Extract action and state
        actions = build_action(f, action_mode)
        states = build_state(f, action_mode)
        assert actions.shape[0] == n_frames
        assert states.shape[0] == n_frames

        action_dim = actions.shape[1]
        state_dim = states.shape[1]

        # Extract and encode video for each camera
        for cam_key in CAMERA_KEYS:
            hdf5_cam = cam_key
            frames = []
            for frame_idx in range(n_frames):
                jpeg = f[f"observation/{hdf5_cam}/rgb"][frame_idx]
                if isinstance(jpeg, bytes):
                    img = decode_jpeg_bytes(jpeg)
                else:
                    img = decode_jpeg_bytes(bytes(jpeg))
                frames.append(img)

            lerobot_cam_key = CAMERA_KEY_MAP[cam_key]
            chunk = episode_index // chunk_size
            video_path = (
                output_root
                / "videos"
                / f"chunk-{chunk:03d}"
                / lerobot_cam_key
                / f"episode_{episode_index:06d}.mp4"
            )
            encode_video_ffmpeg(frames, video_path, fps=FPS)

        # Build parquet data
        timestamps = np.arange(n_frames, dtype=np.float32) / FPS
        frame_indices = np.arange(n_frames, dtype=np.int64)
        episode_indices = np.full(n_frames, episode_index, dtype=np.int64)
        indices = np.arange(n_frames, dtype=np.int64)
        task_indices = np.full(n_frames, task_index, dtype=np.int64)

        table = pa.table({
            "index": pa.array(indices),
            "episode_index": pa.array(episode_indices),
            "frame_index": pa.array(frame_indices),
            "timestamp": pa.array(timestamps),
            "task_index": pa.array(task_indices),
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), state_dim)),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), action_dim)),
        })

        chunk = episode_index // chunk_size
        parquet_path = (
            output_root
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)

        return {
            "episode_index": episode_index,
            "tasks": [instruction],
            "length": n_frames,
            "raw_file_name": episode_dir.name,
        }


def create_info_json(output_root: Path, num_episodes: int, action_dim: int, state_dim: int) -> None:
    """Create Lerobot info.json."""
    features = {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": [None, None, None],  # [T, H, W, C] — filled by video info
            "names": None,
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": [None, None, None],
            "names": None,
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": [None, None, None],
            "names": None,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": None,
        },
    }
    info = {
        "codebase_version": "2.0",
        "fps": FPS,
        "robot_type": "RoboTwin",
        "total_episodes": num_episodes,
        "total_frames": 0,  # Will be updated
        "total_chunks": max(1, (num_episodes + 999) // 1000),
        "chunks_size": 1000,
        "total_tasks": 0,  # Will be updated
        "total_videos": 3,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "splits": {"train": f"0:{num_episodes}"},
    }
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=str, required=True,
                        help="Path to raw RoboTwin C2 data (e.g., .../RoboTwin/C2-v0)")
    parser.add_argument("--seed-list", type=str, default=None,
                        help="Text file with one seed per line")
    parser.add_argument("--episode-list", type=str, default=None,
                        help="Text file with one episode name per line (alternative to --seed-list)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output Lerobot dataset directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval_unseen"])
    parser.add_argument("--action-mode", type=str, default="16d_full",
                        choices=["16d_full", "14d_no_gripper", "14d_left7_right6_gripper1"])
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="Max episodes to convert (0=all)")
    args = parser.parse_args()

    if not args.seed_list and not args.episode_list:
        print("[error] Must provide either --seed-list or --episode-list")
        sys.exit(1)

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load target seeds
    if args.seed_list:
        with open(args.seed_list, "r") as f:
            target_seeds = {int(line.strip()) for line in f if line.strip()}
    else:
        target_seeds = set()
        with open(args.episode_list, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seed = extract_seed_from_dirname(line)
                if seed is not None:
                    target_seeds.add(seed)
    print(f"[info] Target seeds: {len(target_seeds)}")

    # Discover and filter episodes
    all_episodes = discover_episodes(raw_root, args.split)
    print(f"[info] Total {args.split} episodes: {len(all_episodes)}")
    episodes = filter_episodes_by_seed(all_episodes, target_seeds)
    print(f"[info] Matched episodes: {len(episodes)}")

    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]
        print(f"[info] Limited to: {len(episodes)}")

    if not episodes:
        print("[error] No episodes matched. Check seed list and raw data path.")
        sys.exit(1)

    # Determine action/state dims from first episode
    with h5py.File(episodes[0] / "data" / "episode0.hdf5", "r") as f:
        actions = build_action(f, args.action_mode)
        states = build_state(f, args.action_mode)
        action_dim = actions.shape[1]
        state_dim = states.shape[1]
    print(f"[info] Action dim: {action_dim}, State dim: {state_dim}")

    # Create info.json
    create_info_json(output_root, len(episodes), action_dim, state_dim)

    # Collect unique instructions for tasks.jsonl
    task_to_idx = {}
    tasks = []
    total_frames = 0

    # Convert episodes
    for ep_idx, ep_dir in enumerate(tqdm(episodes, desc="Converting episodes")):
        ep_meta = convert_episode(
            episode_dir=ep_dir,
            episode_index=ep_idx,
            task_index=0,  # Will be updated after collecting tasks
            action_mode=args.action_mode,
            output_root=output_root,
        )
        total_frames += ep_meta["length"]

        # Track tasks
        for task in ep_meta["tasks"]:
            if task not in task_to_idx:
                task_to_idx[task] = len(tasks)
                tasks.append(task)

    # Update info.json with totals
    info_path = output_root / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Write tasks.jsonl
    with open(output_root / "meta" / "tasks.jsonl", "w") as f:
        for i, task in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")

    # Write episodes.jsonl
    # Re-convert to get correct task indices (simpler to re-read)
    with open(output_root / "meta" / "episodes.jsonl", "w") as f:
        for ep_idx, ep_dir in enumerate(episodes):
            instruction = load_scene_instruction(ep_dir)
            task_idx = task_to_idx.get(instruction, 0)
            with h5py.File(ep_dir / "data" / "episode0.hdf5", "r") as hf:
                n_frames = hf["joint_action/vector"].shape[0]
            ep_dict = {
                "episode_index": ep_idx,
                "tasks": [instruction],
                "length": n_frames,
                "raw_file_name": ep_dir.name,
            }
            f.write(json.dumps(ep_dict) + "\n")

    # Write empty stats.json (will be computed during training)
    with open(output_root / "meta" / "stats.json", "w") as f:
        json.dump({}, f)

    print(f"[ok] Converted {len(episodes)} episodes, {total_frames} frames")
    print(f"[ok] Output: {output_root}")
    print(f"[ok] Action mode: {args.action_mode} (dim={action_dim})")
    print(f"[ok] Next step: update robotwin.yaml config action/state raw_shape to {action_dim}/{state_dim}")


if __name__ == "__main__":
    main()
