#!/usr/bin/env python3
"""
Build episode filter index for FastWAM Lerobot dataset.

Reads a Lerobot dataset's meta/episodes.jsonl, matches episodes by seed or
episode name against our C1/C2 split lists, and outputs an integer index list
that BaseLerobotDataset can consume via episode_filter_path.

Usage:
    python scripts/build_episode_filter.py \
        --dataset-root ./data/robotwin2.0/robotwin2.0 \
        --seed-list ./data/c2_train_seeds.txt \
        --output ./data/c2_train_episode_indices.txt

    python scripts/build_episode_filter.py \
        --dataset-root ./data/robotwin2.0/robotwin2.0 \
        --episode-names ./data/c2_train_episodes.txt \
        --output ./data/c2_train_episode_indices.txt
"""
import argparse
import json
import re
import sys
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
) -> list[int]:
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


def build_filter_by_names(
    episodes: dict[int, dict],
    target_names: set[str],
) -> list[int]:
    """Match episodes by exact raw_file_name."""
    matched = []
    unmatched_info = []

    for ep_idx, meta in sorted(episodes.items()):
        raw_name = meta.get("raw_file_name", "")
        # Also try to match against a synthetic name if raw_file_name is empty
        name_to_match = raw_name if raw_name else f"episode_{ep_idx:06d}"

        if name_to_match in target_names:
            matched.append(ep_idx)
        else:
            unmatched_info.append((ep_idx, raw_name))

    return matched, unmatched_info


def main():
    parser = argparse.ArgumentParser(description="Build episode filter index for FastWAM")
    parser.add_argument("--dataset-root", required=True, help="Path to Lerobot dataset root")
    parser.add_argument("--seed-list", help="Text file with one seed per line")
    parser.add_argument("--episode-names", help="Text file with one episode name per line")
    parser.add_argument("--output", required=True, help="Output file path (one int index per line)")
    parser.add_argument("--verbose", action="store_true", help="Print unmatched episodes")
    args = parser.parse_args()

    if not args.seed_list and not args.episode_names:
        print("[error] Must provide either --seed-list or --episode-names", file=sys.stderr)
        sys.exit(1)

    dataset_root = Path(args.dataset_root)
    print(f"[info] Loading episodes from {dataset_root / 'meta/episodes.jsonl'}")
    episodes = load_episodes_jsonl(dataset_root)
    print(f"[info] Total episodes in dataset: {len(episodes)}")

    if args.seed_list:
        with open(args.seed_list, "r") as f:
            target_seeds = {int(line.strip()) for line in f if line.strip()}
        print(f"[info] Target seeds from {args.seed_list}: {len(target_seeds)}")
        matched, unmatched = build_filter_by_seeds(episodes, target_seeds)
    else:
        with open(args.episode_names, "r") as f:
            target_names = {line.strip() for line in f if line.strip()}
        print(f"[info] Target episode names from {args.episode_names}: {len(target_names)}")
        matched, unmatched = build_filter_by_names(episodes, target_names)

    print(f"[info] Matched episodes: {len(matched)} / {len(episodes)}")

    if args.verbose and unmatched:
        print(f"[info] Unmatched episodes (showing first 10):")
        for info in unmatched[:10]:
            print(f"  {info}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for idx in matched:
            f.write(f"{idx}\n")
    print(f"[ok] Wrote {len(matched)} episode indices to {output_path}")


if __name__ == "__main__":
    main()
