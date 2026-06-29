#!/usr/bin/env python3
"""Convert RoboTwin C1 HDF5 episodes to FastWAM LeRobot format.

C1 layout differs from C2: each chunk directory contains multiple episodes:
    <raw_root>/<task>/<chunk>/data/episode*.hdf5
    <raw_root>/<task>/<chunk>/scene_info.json  (contains episode_N metadata)
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
FPS = 10


def decode_jpeg_bytes(jpeg_bytes):
    if isinstance(jpeg_bytes, bytes):
        img = Image.open(io.BytesIO(jpeg_bytes))
    else:
        img = Image.open(io.BytesIO(bytes(jpeg_bytes)))
    img = img.convert("RGB")
    return np.array(img)


def discover_hdf5_episodes(raw_root: Path) -> list[Path]:
    pattern = re.compile(r"episode\d+\.hdf5$")
    eps = []
    for path in raw_root.rglob("*.hdf5"):
        if pattern.search(path.name):
            eps.append(path)
    return sorted(eps)


def extract_seed_from_path(hdf5_path: Path) -> int | None:
    # Prefer explicit spec_seed from scene_info; fallback to dirname parsing.
    m = re.search(r"_seed_(\d+)", hdf5_path.as_posix())
    return int(m.group(1)) if m else None


def load_scene_instruction(hdf5_path: Path) -> str:
    """Load instruction for a C1 episode from chunk-level scene_info.json."""
    chunk_dir = hdf5_path.parent.parent
    scene_path = chunk_dir / "scene_info.json"
    if not scene_path.exists():
        return ""
    ep_index = int(re.search(r"episode(\d+)", hdf5_path.name).group(1))
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    wrapper = scene.get(f"episode_{ep_index}", {})
    for key in ("c1_metadata", "c2_metadata"):
        if key in wrapper:
            return wrapper[key].get("instruction", "")
    return ""


def encode_video_ffmpeg(frames: list[np.ndarray], out_path: Path, fps: int = FPS) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]

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

    if tmp_path.stat().st_size < 1024:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg produced empty/invalid file: {tmp_path}")

    import av
    try:
        container = av.open(str(tmp_path))
        container.close()
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg output is not a valid MP4: {e}")

    shutil.move(str(tmp_path), str(out_path))


def build_action(hdf5_file: h5py.File, mode: str) -> np.ndarray:
    vector = hdf5_file["joint_action/vector"][()]
    if mode == "16d_full":
        return vector.astype(np.float32)
    elif mode == "14d_no_gripper":
        left_arm = vector[:, :7]
        right_arm = vector[:, 8:15]
        return np.concatenate([left_arm, right_arm], axis=1).astype(np.float32)
    elif mode == "14d_left7_right6_gripper1":
        left_arm = vector[:, :7]
        right_arm = vector[:, 8:14]
        right_grip = vector[:, 15:16]
        return np.concatenate([left_arm, right_arm, right_grip], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown action mode: {mode}")


def build_state(hdf5_file: h5py.File, mode: str) -> np.ndarray:
    left = hdf5_file["endpose/left_endpose"][()]
    right = hdf5_file["endpose/right_endpose"][()]
    state = np.concatenate([left, right], axis=1).astype(np.float32)
    if mode == "16d_full":
        return hdf5_file["joint_action/vector"][()].astype(np.float32)
    return state


def convert_episode(
    hdf5_path: Path,
    episode_index: int,
    task_index: int,
    action_mode: str,
    output_root: Path,
    chunk_size: int = 1000,
) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        n_frames = f["joint_action/vector"].shape[0]
        instruction = load_scene_instruction(hdf5_path)

        actions = build_action(f, action_mode)
        states = build_state(f, action_mode)
        assert actions.shape[0] == n_frames
        assert states.shape[0] == n_frames

        action_dim = actions.shape[1]
        state_dim = states.shape[1]

        for cam_key in CAMERA_KEYS:
            frames = []
            for frame_idx in range(n_frames):
                jpeg = f[f"observation/{cam_key}/rgb"][frame_idx]
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
            "raw_file_name": hdf5_path.as_posix(),
            "stats": {
                "observation.state": {
                    "count": [int(n_frames)],
                    "min": states.min(axis=0).astype(float).tolist(),
                    "max": states.max(axis=0).astype(float).tolist(),
                    "mean": states.mean(axis=0).astype(float).tolist(),
                    "std": states.std(axis=0).astype(float).tolist(),
                },
                "action": {
                    "count": [int(n_frames)],
                    "min": actions.min(axis=0).astype(float).tolist(),
                    "max": actions.max(axis=0).astype(float).tolist(),
                    "mean": actions.mean(axis=0).astype(float).tolist(),
                    "std": actions.std(axis=0).astype(float).tolist(),
                },
            },
        }


def create_info_json(output_root: Path, num_episodes: int, action_dim: int, state_dim: int) -> None:
    features = {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.fps": FPS,
                "video.codec": "avc1",
            },
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.fps": FPS,
                "video.codec": "avc1",
            },
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.fps": FPS,
                "video.codec": "avc1",
            },
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
        "total_frames": 0,
        "total_chunks": max(1, (num_episodes + 999) // 1000),
        "chunks_size": 1000,
        "total_tasks": 0,
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
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--seed-list", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--action-mode", type=str, default="14d_no_gripper",
                        choices=["16d_full", "14d_no_gripper", "14d_left7_right6_gripper1"])
    parser.add_argument("--max-episodes", type=int, default=0, help="Max episodes to convert (0=all)")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_hdf5 = discover_hdf5_episodes(raw_root)
    print(f"[info] Total HDF5 episodes: {len(all_hdf5)}")

    if args.seed_list:
        with open(args.seed_list, "r") as f:
            target_seeds = {int(line.strip()) for line in f if line.strip()}
        hdf5_paths = [p for p in all_hdf5 if extract_seed_from_path(p) in target_seeds]
        print(f"[info] Matched by seed: {len(hdf5_paths)}")
    else:
        hdf5_paths = all_hdf5

    if args.max_episodes > 0:
        hdf5_paths = hdf5_paths[:args.max_episodes]
        print(f"[info] Limited to: {len(hdf5_paths)}")

    if not hdf5_paths:
        print("[error] No episodes matched.")
        sys.exit(1)

    with h5py.File(hdf5_paths[0], "r") as f:
        actions = build_action(f, args.action_mode)
        states = build_state(f, args.action_mode)
        action_dim = actions.shape[1]
        state_dim = states.shape[1]
    print(f"[info] Action dim: {action_dim}, State dim: {state_dim}")

    create_info_json(output_root, len(hdf5_paths), action_dim, state_dim)

    task_to_idx = {}
    tasks = []
    total_frames = 0
    episode_stats = []

    for ep_idx, hdf5_path in enumerate(tqdm(hdf5_paths, desc="Converting episodes")):
        ep_meta = convert_episode(
            hdf5_path=hdf5_path,
            episode_index=ep_idx,
            task_index=0,
            action_mode=args.action_mode,
            output_root=output_root,
        )
        total_frames += ep_meta["length"]
        episode_stats.append(ep_meta["stats"])
        for task in ep_meta["tasks"]:
            if task not in task_to_idx:
                task_to_idx[task] = len(tasks)
                tasks.append(task)

    info_path = output_root / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    with open(output_root / "meta" / "tasks.jsonl", "w") as f:
        for i, task in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")

    with open(output_root / "meta" / "episodes.jsonl", "w") as f:
        for ep_idx, hdf5_path in enumerate(hdf5_paths):
            instruction = load_scene_instruction(hdf5_path)
            task_idx = task_to_idx.get(instruction, 0)
            with h5py.File(hdf5_path, "r") as hf:
                n_frames = hf["joint_action/vector"].shape[0]
            ep_dict = {
                "episode_index": ep_idx,
                "tasks": [instruction],
                "length": n_frames,
                "raw_file_name": hdf5_path.as_posix(),
            }
            f.write(json.dumps(ep_dict) + "\n")

    with open(output_root / "meta" / "episodes_stats.jsonl", "w") as f:
        for ep_idx, stats in enumerate(episode_stats):
            f.write(json.dumps({"episode_index": ep_idx, "stats": stats}) + "\n")

    with open(output_root / "meta" / "stats.json", "w") as f:
        json.dump({}, f)

    print(f"[ok] Converted {len(hdf5_paths)} episodes, {total_frames} frames")
    print(f"[ok] Output: {output_root}")


if __name__ == "__main__":
    main()
