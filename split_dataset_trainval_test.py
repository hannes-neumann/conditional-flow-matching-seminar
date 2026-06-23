from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata
import numpy as np


def enable_nullable_string_writes() -> None:
    if hasattr(anndata.settings, "allow_write_nullable_strings"):
        anndata.settings.allow_write_nullable_strings = True


def split_trainval_test_by_leiden(
    leiden: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    labels, counts = np.unique(leiden.astype(str), return_counts=True)
    rng = np.random.default_rng(seed)
    tie_break = rng.permutation(len(labels))
    order = np.lexsort((tie_break, -counts))

    n_cells = len(leiden)
    target_sizes = {
        "train_val": n_cells * (1.0 - test_fraction),
        "test": n_cells * test_fraction,
    }
    split_sizes = {"train_val": 0, "test": 0}
    split_clusters: dict[str, list[str]] = {"train_val": [], "test": []}

    for i in order:
        label = str(labels[i])
        count = int(counts[i])
        split = max(
            split_sizes,
            key=lambda name: target_sizes[name] - split_sizes[name],
        )
        split_clusters[split].append(label)
        split_sizes[split] += count

    leiden_str = leiden.astype(str)
    train_val_idx = np.flatnonzero(np.isin(leiden_str, split_clusters["train_val"]))
    test_idx = np.flatnonzero(np.isin(leiden_str, split_clusters["test"]))

    return train_val_idx, test_idx, split_clusters


def write_subset(adata: anndata.AnnData, indices: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adata[indices].copy().write_h5ad(path)


def main() -> None:
    enable_nullable_string_writes()

    parser = argparse.ArgumentParser(
        description=(
            "Split an AnnData file into train_val and test files while keeping "
            "Leiden clusters intact."
        )
    )
    parser.add_argument("--input", default="data/whole_dataset.h5ad")
    parser.add_argument("--train-val-output", default="data/whole_dataset_train_val.h5ad")
    parser.add_argument("--test-output", default="data/whole_dataset_test.h5ad")
    parser.add_argument("--metadata-output", default="data/whole_dataset_split.json")
    parser.add_argument("--leiden-key", default="leiden")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    train_val_path = Path(args.train_val_output)
    test_path = Path(args.test_output)
    metadata_path = Path(args.metadata_output)

    print(f"Loading {input_path} ...")
    adata = anndata.read_h5ad(input_path)
    if args.leiden_key not in adata.obs:
        raise KeyError(f"Missing Leiden column {args.leiden_key!r} in adata.obs")

    leiden = adata.obs[args.leiden_key].astype(str).to_numpy()
    train_val_idx, test_idx, split_clusters = split_trainval_test_by_leiden(
        leiden=leiden,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    metadata = {
        "input": str(input_path),
        "leiden_key": args.leiden_key,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "n_cells": int(adata.n_obs),
        "splits": {
            "train_val": {
                "path": str(train_val_path),
                "n_cells": int(len(train_val_idx)),
                "clusters": split_clusters["train_val"],
            },
            "test": {
                "path": str(test_path),
                "n_cells": int(len(test_idx)),
                "clusters": split_clusters["test"],
            },
        },
    }

    print("Split:")
    for name in ("train_val", "test"):
        info = metadata["splits"][name]
        print(
            f"  {name:9s}: {info['n_cells']:7,d} cells | "
            f"{len(info['clusters']):2d} clusters | {', '.join(info['clusters'])}"
        )

    print(f"Writing {train_val_path} ...")
    write_subset(adata, train_val_idx, train_val_path)
    print(f"Writing {test_path} ...")
    write_subset(adata, test_idx, test_path)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
