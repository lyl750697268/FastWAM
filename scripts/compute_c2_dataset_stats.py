#!/usr/bin/env python3
"""
Compute dataset stats for C2 Lerobot dataset.
Reads all parquet files, calculates action/state normalization statistics,
and writes dataset_stats.json for FastWAM training.

Usage:
    python scripts/compute_c2_dataset_stats.py \
        --dataset-root data/robotwin2.0_c2/train \
        --output data/robotwin2.0_c2/dataset_stats.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm


def compute_stats_from_parquets(dataset_root: Path, action_key: str, state_key: str):
    """Compute dataset stats from parquet files."""
    data_dir = dataset_root / "data"
    parquet_files = list(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"[info] Found {len(parquet_files)} parquet files")

    # First pass: collect per-episode stats and global data
    episode_actions = []
    episode_states = []

    for pf in tqdm(sorted(parquet_files), desc="Reading episodes"):
        table = pq.read_table(str(pf))
        actions = np.array(table.column(action_key).to_pylist(), dtype=np.float32)
        states = np.array(table.column(state_key).to_pylist(), dtype=np.float32)
        episode_actions.append(actions)
        episode_states.append(states)

    # Compute per-episode stats
    def compute_episode_stats(arrays):
        means = []
        stds = []
        mins = []
        maxs = []
        q01s = []
        q99s = []
        for arr in arrays:
            means.append(arr.mean(axis=0))
            stds.append(arr.std(axis=0))
            mins.append(arr.min(axis=0))
            maxs.append(arr.max(axis=0))
            q01s.append(np.percentile(arr, 1, axis=0))
            q99s.append(np.percentile(arr, 99, axis=0))
        return (
            np.stack(means), np.stack(stds),
            np.stack(mins), np.stack(maxs),
            np.stack(q01s), np.stack(q99s)
        )

    action_means, action_stds, action_mins, action_maxs, action_q01s, action_q99s = compute_episode_stats(episode_actions)
    state_means, state_stds, state_mins, state_maxs, state_q01s, state_q99s = compute_episode_stats(episode_states)

    # Global stats (across all frames)
    all_actions = np.concatenate(episode_actions, axis=0)
    all_states = np.concatenate(episode_states, axis=0)

    def make_stats(means, stds, mins, maxs, q01s, q99s, all_data):
        # Stepwise: aggregate per-episode stats
        stepwise_mean = means.mean(axis=0)
        stepwise_std = np.sqrt((stds**2 + (means - stepwise_mean)**2).mean(axis=0))
        stepwise_min = mins.min(axis=0)
        stepwise_max = maxs.max(axis=0)
        stepwise_q01 = q01s.min(axis=0)
        stepwise_q99 = q99s.max(axis=0)

        # Global: across all data
        global_mean = all_data.mean(axis=0)
        global_std = all_data.std(axis=0)
        global_min = all_data.min(axis=0)
        global_max = all_data.max(axis=0)
        global_q01 = np.percentile(all_data, 1, axis=0)
        global_q99 = np.percentile(all_data, 99, axis=0)

        return {
            "stepwise_mean": torch.from_numpy(stepwise_mean.astype(np.float32)),
            "stepwise_std": torch.from_numpy(stepwise_std.astype(np.float32)),
            "stepwise_min": torch.from_numpy(stepwise_min.astype(np.float32)),
            "stepwise_max": torch.from_numpy(stepwise_max.astype(np.float32)),
            "stepwise_q01": torch.from_numpy(stepwise_q01.astype(np.float32)),
            "stepwise_q99": torch.from_numpy(stepwise_q99.astype(np.float32)),
            "global_mean": torch.from_numpy(global_mean.astype(np.float32)),
            "global_std": torch.from_numpy(global_std.astype(np.float32)),
            "global_min": torch.from_numpy(global_min.astype(np.float32)),
            "global_max": torch.from_numpy(global_max.astype(np.float32)),
            "global_q01": torch.from_numpy(global_q01.astype(np.float32)),
            "global_q99": torch.from_numpy(global_q99.astype(np.float32)),
        }

    stats = {
        "action": {"default": make_stats(action_means, action_stds, action_mins, action_maxs, action_q01s, action_q99s, all_actions)},
        "state": {"default": make_stats(state_means, state_stds, state_mins, state_maxs, state_q01s, state_q99s, all_states)},
        "num_episodes": len(parquet_files),
        "num_transition": len(all_actions),
    }

    return stats


def save_stats(stats, output_path):
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(stats), f, indent=2)
    print(f"[ok] Saved stats to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--action-key", type=str, default="action")
    parser.add_argument("--state-key", type=str, default="observation.state")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    stats = compute_stats_from_parquets(root, args.action_key, args.state_key)
    save_stats(stats, args.output)


if __name__ == "__main__":
    main()
