"""
Generate and cache morphology predictions for trained CFM models on the held-out test set.

This script is intentionally separate from metric/plot notebooks: ODE sampling is the
expensive step, so predictions are saved once and can then be reused for analysis.

Examples:
  uv run python predict_morphology_test.py --device cpu --limit 128 --force
  uv run python predict_morphology_test.py --model_specs model_specs.json --device cuda

Model spec JSON can either be a list of specs or {"models": [...]}:
[
  {
    "name": "50pc_film_model",
    "artifact": "entity/project/artifact-name:v0"
  },
  {
    "name": "local_pca_film",
    "checkpoint_path": "outputs/morphology/pca_film/run/model_ema.pt",
    "config_path": "outputs/morphology/pca_film/run/config.json",
    "y_stats_path": "outputs/morphology/pca_film/run/y_stats.pt"
  }
]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import anndata
import numpy as np
import torch
import torch.nn as nn
import torchdiffeq

from train_cfm_morphology import (
    Config,
    build_model,
    get_device,
    get_morphology_columns,
    impute_nonfinite_features,
)

try:
    import wandb
except ImportError:
    wandb = None


DEFAULT_MODEL_SPECS: list[dict[str, Any]] = [
    {
        "name": "unconditioned",
        "model_variant": "baseline",
        "checkpoint_path": "models/morphology/dark-feather-17.pt",
        "y_stats_path": "outputs/y_stats.pt",
        "hidden_dim": 512,
        "n_res_blocks": 6,
        "time_emb_dim": 128,
        "y_dim": 56,
    },
    {
        "name": "50pc_film_model",
        "artifact": "hneumann-university-of-mannheim/cfm-morphology/morphology-pca-film-model:v0",
    },
    {
        "name": "50pc_concat_model",
        "artifact": "hneumann-university-of-mannheim/cfm-morphology/morphology-pca-concat-model:v0",
    },
    {
        "name": "full_film_model",
        "artifact": "hneumann-university-of-mannheim/cfm-morphology/morphology-full-film-model:v0",
    },
    {
        "name": "30pc_film",
        "artifact": "hneumann-university-of-mannheim/cfm-morphology/morphology-pca-film-model:v3",
    },
]


class BaselineVectorField(nn.Module):
    def __init__(
        self,
        y_dim: int,
        hidden_dim: int,
        n_res_blocks: int,
        time_emb_dim: int,
    ) -> None:
        super().__init__()
        from train_cfm_morphology import ConcatResBlock, sinusoidal_embedding

        self.sinusoidal_embedding = sinusoidal_embedding
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.input_proj = nn.Linear(y_dim + time_emb_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ConcatResBlock(hidden_dim) for _ in range(n_res_blocks)]
        )
        self.output_head = nn.Linear(hidden_dim, y_dim)

    def forward(self, y_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_sin = self.sinusoidal_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_sin)
        h = self.input_proj(torch.cat([y_t, t_emb], dim=-1))
        for block in self.res_blocks:
            h = block(h)
        return self.output_head(h)


def slugify(value: str) -> str:
    keep = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        elif char in {" ", ".", "/", ":"}:
            keep.append("-")
    return "".join(keep).strip("-_") or "model"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_model_specs(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(spec) for spec in DEFAULT_MODEL_SPECS]

    with open(path) as f:
        raw = json.load(f)
    specs = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
    if not isinstance(specs, list):
        raise ValueError("Model spec JSON must be a list or a dict with key 'models'")
    return [dict(spec) for spec in specs]


def artifact_to_spec(spec: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    if wandb is None:
        raise ImportError("wandb is required to load W&B artifacts")

    artifact_ref = spec["artifact"]
    print(f"Downloading W&B artifact {artifact_ref} ...")
    api = wandb.Api()
    artifact = api.artifact(artifact_ref, type="model")
    safe_artifact_dir = artifact_ref.replace("/", "__").replace(":", "__")
    download_dir = Path(artifact.download(root=str(artifact_dir / safe_artifact_dir)))

    checkpoint_filename = spec.get("checkpoint_filename", "model_ema.pt")
    checkpoint_path = download_dir / checkpoint_filename
    config_path = download_dir / spec.get("config_filename", "config.json")
    y_stats_path = download_dir / spec.get("y_stats_filename", "y_stats.pt")

    if not checkpoint_path.exists() and checkpoint_filename == "model_ema.pt":
        step_checkpoints = sorted(download_dir.glob("model_ema_step_*.pt"))
        if step_checkpoints:
            checkpoint_path = step_checkpoints[-1]

    for path, label in [
        (checkpoint_path, "checkpoint"),
        (config_path, "config.json"),
        (y_stats_path, "y_stats.pt"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Artifact {artifact_ref} is missing {label}: {path}")

    resolved = {
        **spec,
        "name": spec.get("name", artifact.name),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "y_stats_path": str(y_stats_path),
    }
    return resolved


def resolve_model_specs(
    specs: list[dict[str, Any]],
    artifact_dir: Path,
) -> list[dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved = []
    for spec in specs:
        resolved.append(artifact_to_spec(spec, artifact_dir) if "artifact" in spec else spec)
    return resolved


def config_from_spec(spec: dict[str, Any], root: Path) -> Config:
    cfg = Config()

    config_path = spec.get("config_path")
    if config_path:
        with open(resolve_path(config_path, root)) as f:
            train_cfg = json.load(f)
        for key, value in train_cfg.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    for key, value in spec.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    if spec.get("model_variant") == "baseline":
        cfg.model_variant = "baseline"
    return cfg


def build_model_for_spec(
    spec: dict[str, Any],
    cfg: Config,
    input_dim: int,
) -> nn.Module:
    if cfg.model_variant == "baseline":
        return BaselineVectorField(
            y_dim=cfg.y_dim,
            hidden_dim=cfg.hidden_dim,
            n_res_blocks=cfg.n_res_blocks,
            time_emb_dim=cfg.time_emb_dim,
        )
    return build_model(cfg, input_dim=input_dim)


def load_test_data(data_path: Path, y_dim: int, leiden_key: str) -> dict[str, Any]:
    print(f"Loading test data from {data_path} ...")
    adata = anndata.read_h5ad(data_path)
    morph_cols = get_morphology_columns(list(adata.obs.columns), y_dim)

    y_raw = torch.tensor(adata.obs[morph_cols].values.astype(np.float32), dtype=torch.float32)
    y_raw = impute_nonfinite_features(y_raw, morph_cols)
    y_log = torch.sign(y_raw) * torch.log1p(torch.abs(y_raw))

    x_pca = torch.tensor(np.asarray(adata.obsm["X_pca"], dtype=np.float32), dtype=torch.float32)
    x_full_np = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    x_full = torch.tensor(np.asarray(x_full_np, dtype=np.float32), dtype=torch.float32)

    if leiden_key not in adata.obs:
        raise KeyError(f"Missing Leiden column {leiden_key!r} in test data")

    return {
        "adata": adata,
        "morph_cols": morph_cols,
        "y_raw": y_raw,
        "y_log": y_log,
        "x_pca": x_pca,
        "x_full": x_full,
        "leiden": adata.obs[leiden_key].astype(str).to_numpy(),
        "obs_names": adata.obs_names.astype(str).to_numpy(),
    }


def input_for_model(cfg: Config, data: dict[str, Any]) -> tuple[torch.Tensor | None, int]:
    if cfg.model_variant == "baseline":
        return None, 0
    if cfg.model_variant in {"pca_concat", "pca_film"}:
        if data["x_pca"].shape[1] < cfg.c_dim:
            raise ValueError(
                f"Requested top {cfg.c_dim} PCs, but X_pca has only {data['x_pca'].shape[1]}"
            )
        x = data["x_pca"][:, : cfg.c_dim]
        return x, int(x.shape[1])
    if cfg.model_variant == "full_film":
        x = data["x_full"]
        return x, int(x.shape[1])
    raise ValueError(f"Unknown model_variant {cfg.model_variant!r}")


def load_y_stats(spec: dict[str, Any], checkpoint_path: Path, morph_cols: list[str], root: Path):
    candidates = []
    if spec.get("y_stats_path"):
        candidates.append(resolve_path(spec["y_stats_path"], root))
    candidates.append(checkpoint_path.parent / "y_stats.pt")
    candidates.append(root / "outputs" / "y_stats.pt")

    for path in candidates:
        if path.exists():
            stats = torch.load(path, map_location="cpu")
            columns = list(stats.get("columns", []))
            if columns and columns != morph_cols:
                raise ValueError(f"Column mismatch between {path} and test morphology columns")
            return stats["mean"].float(), stats["std"].float().clamp(min=1e-8), path

    raise FileNotFoundError(
        f"No y_stats.pt found for {spec['name']}. Add y_stats_path to the model spec."
    )


@torch.no_grad()
def sample_chunk(
    model: nn.Module,
    cfg: Config,
    cond: torch.Tensor | None,
    n: int,
    device: torch.device,
    method: str,
    rtol: float,
    atol: float,
) -> torch.Tensor:
    y0 = torch.randn(n, cfg.y_dim, device=device)
    cond_device = None if cond is None else cond.to(device, non_blocking=True)

    def ode_fn(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_batch = t.expand(y.shape[0])
        if cond_device is None:
            return model(y, t_batch)
        return model(y, t_batch, cond_device)

    traj = torchdiffeq.odeint(
        ode_fn,
        y0,
        torch.tensor([0.0, 1.0], device=device),
        method=method,
        rtol=rtol,
        atol=atol,
        options={"dtype": torch.float32},
    )
    return traj[-1].cpu()


def inverse_morphology_transform(y_log: torch.Tensor) -> np.ndarray:
    return (torch.sign(y_log) * torch.expm1(torch.abs(y_log))).numpy()


def save_targets_once(output_dir: Path, data: dict[str, Any]) -> None:
    target_path = output_dir / "test_targets.npz"
    if target_path.exists():
        return
    np.savez_compressed(
        target_path,
        y_real=data["y_raw"].numpy(),
        y_log=data["y_log"].numpy(),
        leiden=data["leiden"],
        obs_names=data["obs_names"],
        morph_cols=np.asarray(data["morph_cols"], dtype=object),
    )
    print(f"Saved shared test targets -> {target_path}")


def predict_one_model(
    spec: dict[str, Any],
    data: dict[str, Any],
    args: argparse.Namespace,
    root: Path,
    device: torch.device,
) -> None:
    model_name = spec["name"]
    model_dir = Path(args.output_dir) / slugify(model_name)
    pred_path = model_dir / "predictions.npz"
    meta_path = model_dir / "metadata.json"
    if pred_path.exists() and not args.force:
        print(f"Skipping {model_name}: predictions already exist at {pred_path}")
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = config_from_spec(spec, root)
    cfg.device = args.device
    cfg.ode_rtol = args.ode_rtol
    cfg.ode_atol = args.ode_atol

    cond_all, input_dim = input_for_model(cfg, data)
    model = build_model_for_spec(spec, cfg, input_dim=input_dim).to(device)

    checkpoint_path = resolve_path(spec["checkpoint_path"], root)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    y_mean, y_std, y_stats_path = load_y_stats(
        spec, checkpoint_path, data["morph_cols"], root
    )
    y_real_norm = (data["y_log"] - y_mean) / y_std

    n_total = int(len(y_real_norm))
    if args.limit is not None:
        n_total = min(n_total, args.limit)
    batch_size = min(args.batch_size, n_total)
    print(
        f"\nPredicting {model_name} ({cfg.model_variant}) | "
        f"{n_total:,} cells | batch_size={batch_size} | device={device}"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    y_gen_norm_parts = []
    t0 = time.time()
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        cond = None if cond_all is None else cond_all[start:end]
        y_gen_norm = sample_chunk(
            model=model,
            cfg=cfg,
            cond=cond,
            n=end - start,
            device=device,
            method=args.ode_method,
            rtol=args.ode_rtol,
            atol=args.ode_atol,
        )
        y_gen_norm_parts.append(y_gen_norm)
        elapsed = time.time() - t0
        cells_per_second = end / max(elapsed, 1e-8)
        print(
            f"  {end:7,d}/{n_total:7,d} cells "
            f"| {cells_per_second:,.1f} cells/s | elapsed {elapsed / 60:.1f} min"
        )

    y_gen_norm_all = torch.cat(y_gen_norm_parts, dim=0)
    y_gen_log = y_gen_norm_all * y_std + y_mean
    y_pred = inverse_morphology_transform(y_gen_log)
    y_real = data["y_raw"][:n_total].numpy()

    np.savez_compressed(
        pred_path,
        y_pred=y_pred.astype(np.float32),
        y_gen_norm=y_gen_norm_all.numpy().astype(np.float32),
        y_real=y_real.astype(np.float32),
        y_real_norm=y_real_norm[:n_total].numpy().astype(np.float32),
        leiden=data["leiden"][:n_total],
        obs_names=data["obs_names"][:n_total],
        morph_cols=np.asarray(data["morph_cols"], dtype=object),
    )

    metadata = {
        "model_name": model_name,
        "model_variant": cfg.model_variant,
        "checkpoint_path": str(checkpoint_path),
        "y_stats_path": str(y_stats_path),
        "config": asdict(cfg),
        "spec": spec,
        "data_path": str(resolve_path(args.data_path, root)),
        "n_cells": n_total,
        "batch_size": batch_size,
        "device": str(device),
        "ode_method": args.ode_method,
        "ode_rtol": args.ode_rtol,
        "ode_atol": args.ode_atol,
        "seed": args.seed,
        "seconds": time.time() - t0,
        "prediction_file": str(pred_path),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved predictions -> {pred_path}")
    print(f"Saved metadata    -> {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache CFM morphology predictions on the held-out test set"
    )
    parser.add_argument("--model_specs", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="data/whole_dataset_test.h5ad")
    parser.add_argument("--output_dir", type=str, default="outputs/test_predictions")
    parser.add_argument("--artifact_dir", type=str, default="outputs/test_predictions/artifacts")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test cell limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ode_method", type=str, default="dopri5")
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument("--ode_atol", type=float, default=1e-5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = project_root()
    args.data_path = str(resolve_path(args.data_path, root))
    args.output_dir = str(resolve_path(args.output_dir, root))
    args.artifact_dir = str(resolve_path(args.artifact_dir, root))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    specs = load_model_specs(args.model_specs)
    specs = resolve_model_specs(specs, Path(args.artifact_dir))

    data = load_test_data(
        data_path=Path(args.data_path),
        y_dim=56,
        leiden_key="leiden",
    )
    save_targets_once(Path(args.output_dir), data)

    for spec in specs:
        predict_one_model(spec, data, args, root, device)


if __name__ == "__main__":
    main()
