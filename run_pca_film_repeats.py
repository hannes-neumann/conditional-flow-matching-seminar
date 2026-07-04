from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(
    *,
    args: argparse.Namespace,
    c_dim: int,
    seed: int,
    output_dir: Path,
) -> list[str]:
    run_name = f"pca-film-{c_dim}pc-seed-{seed}"
    artifact_name = f"morphology-pca-film-{c_dim}pc-seed-{seed}"

    command = [
        sys.executable,
        "train_cfm_morphology.py",
        "--model_variant",
        "pca_film",
        "--data_path",
        args.data_path,
        "--output_dir",
        str(output_dir),
        "--n_steps",
        str(args.n_steps),
        "--batch_size",
        str(args.batch_size),
        "--hidden_dim",
        str(args.hidden_dim),
        "--c_dim",
        str(c_dim),
        "--n_res_blocks",
        str(args.n_res_blocks),
        "--lr",
        str(args.lr),
        "--seed",
        str(seed),
        "--log_every",
        str(args.log_every),
        "--val_every",
        str(args.val_every),
        "--artifact_checkpoint_step",
        str(args.artifact_checkpoint_step),
        "--device",
        args.device,
        "--wandb_mode",
        args.wandb_mode,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        run_name,
        "--wandb_artifact_name",
        artifact_name,
    ]
    if args.skip_eval:
        command.append("--skip_eval")
    if args.no_wandb:
        command.append("--no_wandb")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated PCA-FiLM morphology trainings for seed-level comparison "
            "between different numbers of conditioning PCs."
        )
    )
    parser.add_argument("--data_path", default="data/whole_dataset_train_val.h5ad")
    parser.add_argument("--output_root", default="outputs/morphology_repeats")
    parser.add_argument(
        "--c_dims",
        nargs="+",
        type=int,
        default=[30, 50],
        help="PCA condition dimensions to compare.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[43, 44],
        help="Additional random seeds to train. Defaults to two extra repeats.",
    )
    parser.add_argument("--n_steps", type=int, default=30_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_res_blocks", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--artifact_checkpoint_step", type=int, default=30_000)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_group", default="pca_film_repeats")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    planned_runs: list[tuple[int, int, Path, list[str]]] = []

    for c_dim in args.c_dims:
        for seed in args.seeds:
            output_dir = output_root / f"pca_film_{c_dim}pc_seed_{seed}"
            command = build_command(
                args=args,
                c_dim=c_dim,
                seed=seed,
                output_dir=output_dir,
            )
            planned_runs.append((c_dim, seed, output_dir, command))

    print("Planned PCA-FiLM repeat runs:")
    for c_dim, seed, output_dir, command in planned_runs:
        print(f"  c_dim={c_dim:2d} | seed={seed:4d} | output={output_dir}")
        if args.dry_run:
            print("    " + " ".join(command))

    if args.dry_run:
        return

    for i, (c_dim, seed, output_dir, command) in enumerate(planned_runs, start=1):
        print("\n" + "=" * 88, flush=True)
        print(
            f"Run {i}/{len(planned_runs)}: PCA-FiLM with c_dim={c_dim}, seed={seed}",
            flush=True,
        )
        print(" ".join(command), flush=True)
        print("=" * 88 + "\n", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
