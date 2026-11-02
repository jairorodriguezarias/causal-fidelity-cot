"""Build reproducible, premise-disjoint SNLI contrast pairs from official splits."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

DATASET_ID = "stanfordnlp/snli"
OUTPUT_FILE = "artifacts/dataset.json"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def premise_key(text):
    return " ".join(text.casefold().split())


def select_pairs(rows, count, seed, excluded=()):
    groups = defaultdict(lambda: {0: set(), 2: set()})
    originals = {}
    excluded = set(excluded)
    for row in rows:
        premise, hypothesis, label = row["premise"].strip(), row["hypothesis"].strip(), row["label"]
        key = premise_key(premise)
        if label not in (0, 2) or not premise or not hypothesis or key in excluded:
            continue
        originals[key] = premise
        groups[key][label].add(hypothesis)
    candidates = sorted(key for key, g in groups.items() if g[0] and g[2])
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f"Need {count} disjoint contrast pairs, found {len(candidates)}")
    pairs = []
    for key in candidates[:count]:
        pairs.append({
            "id": hashlib.sha256(key.encode()).hexdigest()[:16],
            "premise": originals[key],
            "entail_hypothesis": rng.choice(sorted(groups[key][0])),
            "contra_hypothesis": rng.choice(sorted(groups[key][2])),
        })
    return pairs


def validate_dataset(data):
    seen = set()
    for name in ("train", "calibration", "test"):
        pairs = data["splits"][name]
        if not pairs:
            raise ValueError(f"Empty split: {name}")
        keys = [premise_key(p["premise"]) for p in pairs]
        if len(set(keys)) != len(keys) or seen.intersection(keys):
            raise ValueError("Premise leakage or duplicate within a split")
        seen.update(keys)
    if fingerprint(data["splits"]) != data["sha256"]:
        raise ValueError("Dataset checksum mismatch")


def build_contrast_dataset(n_pairs=500, seed=42, n_calibration=50, n_test=50, output=OUTPUT_FILE):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info(DATASET_ID).sha
    splits, seen = {}, set()
    for name, source, count in (("train", "train", n_pairs),
                                ("calibration", "validation", n_calibration),
                                ("test", "test", n_test)):
        rows = load_dataset(DATASET_ID, split=source, revision=revision)
        splits[name] = select_pairs(rows, count, seed, excluded=seen)
        seen.update(premise_key(p["premise"]) for p in splits[name])
        print(f"{name}: {len(splits[name])} pairs", flush=True)
    data = {"schema_version": 4, "source": DATASET_ID, "revision": revision,
            "seed": seed, "source_splits": {"train": "train", "calibration": "validation", "test": "test"},
            "splits": splits, "sha256": fingerprint(splits)}
    validate_dataset(data)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return data


def load_contrast_dataset(path=OUTPUT_FILE):
    data = json.loads(Path(path).read_text())
    validate_dataset(data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--calibration", type=int, default=50)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    build_contrast_dataset(args.train, args.seed, args.calibration, args.test, args.output)
