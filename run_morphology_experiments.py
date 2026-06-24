from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_VARIANTS = ["pca_concat", "pca_film", "full_film"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run morphology CFM experiments sequentially."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=DEFAULT_VARIANTS,
        help="Model variants to run in order.",
    )
    parser.add_argument("--data_path", default="data/whole_dataset_train_val.h5ad")
    parser.add_argument("--output_root", default="outputs/morphology")
    parser.add_argument("--n_steps", type=int, default=30_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_res_blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)

    for variant in args.variants:
        output_dir = output_root / variant
        command = [
            sys.executable,
            "train_cfm_morphology.py",
            "--model_variant",
            variant,
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
            "--n_res_blocks",
            str(args.n_res_blocks),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--log_every",
            str(args.log_every),
            "--val_every",
            str(args.val_every),
            "--device",
            args.device,
            "--wandb_mode",
            args.wandb_mode,
            "--wandb_group",
            variant,
            "--wandb_artifact_name",
            f"morphology-{variant.replace('_', '-')}-model",
        ]
        if args.skip_eval:
            command.append("--skip_eval")
        if args.no_wandb:
            command.append("--no_wandb")

        print("\n" + "=" * 80, flush=True)
        print(f"Running morphology experiment: {variant}", flush=True)
        print(" ".join(command), flush=True)
        print("=" * 80 + "\n", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
