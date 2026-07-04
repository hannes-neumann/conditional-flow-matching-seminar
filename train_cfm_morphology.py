"""
Conditional Flow Matching for predicting cell morphology from spatial gene expression.
Self-contained Kaggle training script.

Usage:
  python train_cfm_morphology.py
  python train_cfm_morphology.py --n_steps 500 --batch_size 256   # smoke test
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, TensorDataset

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    data_path: str = "data/whole_dataset_train_val.h5ad"
    output_dir: str | None = None
    model_variant: str = "pca_film"

    # Architecture
    y_dim: int = 56
    c_dim: int = 30
    x_dim: int | None = None
    gene_encoder_hidden_dim: int = 512
    hidden_dim: int = 512
    n_res_blocks: int = 6
    time_emb_dim: int = 128

    # Training
    seed: int = 42
    batch_size: int = 512
    n_steps: int = 30_000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    ema_decay: float = 0.9999
    sigma: float = 0.0
    leiden_key: str = "leiden"
    val_fraction: float = 0.1

    # Logging / validation
    log_every: int = 100
    val_every: int = 1000
    artifact_checkpoint_step: int = 30_000
    use_wandb: bool = True
    wandb_project: str = "cfm-morphology"
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_mode: str = "online"
    wandb_artifact_name: str = ""

    # Eval
    n_eval_samples: int = 4096
    n_eval_samples_per_cluster: int = 2048
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-5
    skip_eval: bool = False
    device: str = "auto"
    max_nonfinite_batches: int = 20


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map scalar time t (shape (B,)) to a (B, dim) sinusoidal embedding."""
    if t.dim() == 0:
        t = t.unsqueeze(0)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConcatResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FiLMResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = self.linear1(F.silu(self.norm1(x)))
        h = (1 + gamma) * h + beta
        h = self.linear2(F.silu(self.norm2(h)))
        return x + h


class PCAConcatVectorField(nn.Module):
    """v_θ(y_t, t, c): R^56 × [0,1] × R^50 → R^56"""

    def __init__(
        self,
        y_dim: int,
        c_dim: int,
        hidden_dim: int,
        n_res_blocks: int,
        time_emb_dim: int,
    ) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Sinusoidal → small MLP → time feature
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        self.input_proj = nn.Linear(y_dim + time_emb_dim + c_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ConcatResBlock(hidden_dim) for _ in range(n_res_blocks)]
        )
        self.output_head = nn.Linear(hidden_dim, y_dim)

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_sin = sinusoidal_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_sin)
        x = torch.cat([y_t, t_emb, c], dim=-1)
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_head(x)


class PCAFiLMVectorField(nn.Module):
    def __init__(
        self,
        y_dim: int,
        c_dim: int,
        hidden_dim: int,
        n_res_blocks: int,
        time_emb_dim: int,
    ) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        cond_dim = c_dim + time_emb_dim
        self.input_proj = nn.Linear(y_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [FiLMResBlock(hidden_dim, cond_dim) for _ in range(n_res_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, y_dim)

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, x_pca: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_emb_dim))
        cond = torch.cat([x_pca, t_emb], dim=-1)

        h = self.input_proj(y_t)
        for block in self.res_blocks:
            h = block(h, cond)
        return self.output_head(self.output_norm(h))


class FullFiLMVectorField(nn.Module):
    def __init__(
        self,
        y_dim: int,
        x_dim: int,
        c_dim: int,
        gene_encoder_hidden_dim: int,
        hidden_dim: int,
        n_res_blocks: int,
        time_emb_dim: int,
    ) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.gene_encoder = nn.Sequential(
            nn.Linear(x_dim, gene_encoder_hidden_dim),
            nn.LayerNorm(gene_encoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(gene_encoder_hidden_dim, c_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        cond_dim = c_dim + time_emb_dim
        self.input_proj = nn.Linear(y_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [FiLMResBlock(hidden_dim, cond_dim) for _ in range(n_res_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, y_dim)

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, x_gene: torch.Tensor) -> torch.Tensor:
        c = self.gene_encoder(x_gene)
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_emb_dim))
        cond = torch.cat([c, t_emb], dim=-1)

        h = self.input_proj(y_t)
        for block in self.res_blocks:
            h = block(h, cond)
        return self.output_head(self.output_norm(h))


def build_model(cfg: Config, input_dim: int) -> nn.Module:
    if cfg.model_variant == "pca_concat":
        return PCAConcatVectorField(
            y_dim=cfg.y_dim,
            c_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            n_res_blocks=cfg.n_res_blocks,
            time_emb_dim=cfg.time_emb_dim,
        )
    if cfg.model_variant == "pca_film":
        return PCAFiLMVectorField(
            y_dim=cfg.y_dim,
            c_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            n_res_blocks=cfg.n_res_blocks,
            time_emb_dim=cfg.time_emb_dim,
        )
    if cfg.model_variant == "full_film":
        return FullFiLMVectorField(
            y_dim=cfg.y_dim,
            x_dim=input_dim,
            c_dim=cfg.c_dim,
            gene_encoder_hidden_dim=cfg.gene_encoder_hidden_dim,
            hidden_dim=cfg.hidden_dim,
            n_res_blocks=cfg.n_res_blocks,
            time_emb_dim=cfg.time_emb_dim,
        )
    raise ValueError(f"Unknown model_variant {cfg.model_variant!r}")


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_DROP_COLUMNS = {
    "Center_X", "Center_Y",
    "BoundingBoxMinimum_X", "BoundingBoxMaximum_X",
    "BoundingBoxMinimum_Y", "BoundingBoxMaximum_Y",
    "Orientation",
    "NormalizedMoment_1_0", "NormalizedMoment_0_0", "NormalizedMoment_0_1"
}



def get_morphology_columns(obs_columns: list[str], y_dim: int) -> list[str]:
    cols = [c for c in obs_columns if c[0].isupper()]
    cols = [c for c in cols if c not in _DROP_COLUMNS]
    cols = [c for c in cols if not c.startswith("SpatialMoment_")]
    assert len(cols) == y_dim, (
        f"Expected {y_dim} morphological features, got {len(cols)}.\n"
        f"Columns: {cols}"
    )
    return cols


def impute_nonfinite_features(y: torch.Tensor, cols: list[str]) -> torch.Tensor:
    finite_mask = torch.isfinite(y)
    if finite_mask.all():
        return y

    n_bad_cells = (~finite_mask).any(dim=1).sum().item()
    n_bad_vals = (~finite_mask).sum().item()
    print(
        f"Imputing {n_bad_vals} non-finite morphology values across "
        f"{n_bad_cells} cells ({n_bad_cells / len(y) * 100:.1f}%) with per-feature median"
    )

    y = y.clone()
    for j, col_name in enumerate(cols):
        col = y[:, j]
        valid = torch.isfinite(col)
        if valid.all():
            continue
        if not valid.any():
            raise ValueError(f"Morphology feature {col_name!r} has no finite values")
        median = col[valid].median()
        y[:, j] = torch.where(valid, col, median)
    return y


def split_train_val_stratified_by_leiden(
    leiden: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    labels, counts = np.unique(leiden.astype(str), return_counts=True)
    rng = np.random.default_rng(seed)
    leiden_str = leiden.astype(str)
    train_parts = []
    val_parts = []
    per_cluster = {}

    for label, count in zip(labels, counts):
        cluster_idx = np.flatnonzero(leiden_str == str(label))
        shuffled = rng.permutation(cluster_idx)
        n_val = int(round(len(cluster_idx) * val_fraction))
        if len(cluster_idx) > 1:
            n_val = min(max(n_val, 1), len(cluster_idx) - 1)
        val_cluster_idx = shuffled[:n_val]
        train_cluster_idx = shuffled[n_val:]
        train_parts.append(train_cluster_idx)
        val_parts.append(val_cluster_idx)
        per_cluster[str(label)] = {
            "train": int(len(train_cluster_idx)),
            "val": int(len(val_cluster_idx)),
        }

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    split_clusters: dict[str, list[str] | dict[str, dict[str, int]]] = {
        "train": [str(label) for label in labels],
        "val": [str(label) for label in labels],
        "per_cluster": per_cluster,
    }

    return train_idx, val_idx, split_clusters


def load_data(cfg: Config):
    print(f"Loading {cfg.data_path} ...")
    adata = anndata.read_h5ad(cfg.data_path)

    if cfg.model_variant in {"pca_concat", "pca_film"}:
        x_pca = torch.tensor(np.array(adata.obsm["X_pca"]), dtype=torch.float32)
        if x_pca.shape[1] < cfg.c_dim:
            raise ValueError(
                f"Requested top {cfg.c_dim} PCs, but X_pca only has {x_pca.shape[1]}"
            )
        x = x_pca[:, : cfg.c_dim]
        assert x.shape[1] == cfg.c_dim, (
            f"Expected sliced X_pca.shape[1] == {cfg.c_dim}, got {x.shape[1]}"
        )
        input_name = f"X_pca[:, :{cfg.c_dim}]"
    elif cfg.model_variant == "full_film":
        x_np = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        x = torch.tensor(np.asarray(x_np), dtype=torch.float32)
        cfg.x_dim = int(x.shape[1])
        input_name = "adata.X"
    else:
        raise ValueError(f"Unknown model_variant {cfg.model_variant!r}")

    if not torch.isfinite(x).all():
        raise ValueError(f"{input_name} contains NaN or infinite values")

    morph_cols = get_morphology_columns(list(adata.obs.columns), cfg.y_dim)
    y = torch.tensor(adata.obs[morph_cols].values.astype(np.float32), dtype=torch.float32)
    assert y.shape[1] == cfg.y_dim, f"Expected y.shape[1] == {cfg.y_dim}, got {y.shape[1]}"
    if cfg.leiden_key not in adata.obs:
        raise KeyError(f"Missing Leiden column {cfg.leiden_key!r} in adata.obs")
    leiden = adata.obs[cfg.leiden_key].astype(str).to_numpy()

    print(f"Loaded: x={tuple(x.shape)} from {input_name}, y={tuple(y.shape)}")

    y = impute_nonfinite_features(y, morph_cols)
    assert torch.isfinite(y).all(), "Non-finite values remain after imputation"

    # Raw CellProfiler features span many orders of magnitude and include negative values
    # (Hu moments, central moments). Signed log1p compresses the dynamic range while
    # preserving sign, making z-score standardization well-behaved.
    y = torch.sign(y) * torch.log1p(torch.abs(y))
    print(
        f"After signed log1p: min={y.min().item():.3f}, max={y.max().item():.3f}, "
        f"any NaN={torch.isnan(y).any().item()}, any inf={torch.isinf(y).any().item()}"
    )
    assert torch.isfinite(y).all(), "Signed log1p produced non-finite morphology values"

    train_idx, val_idx, split_clusters = split_train_val_stratified_by_leiden(
        leiden=leiden,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    print("Stratified Leiden train/val split:")
    for name, idx in (("train", train_idx), ("val", val_idx)):
        clusters = split_clusters[name]
        print(
            f"  {name:5s}: {len(idx):7,d} cells | "
            f"{len(clusters):2d} clusters | {', '.join(clusters)}"
        )

    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    leiden_train, leiden_val = leiden[train_idx], leiden[val_idx]

    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0).clamp(min=1e-8)

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    assert torch.isfinite(y_train).all(), "Non-finite values in standardized y_train"
    assert torch.isfinite(y_val).all(), "Non-finite values in standardized y_val"

    abs_max = y_train.abs().max(dim=0).values
    top = torch.topk(abs_max, k=5)
    print("Top-5 features by |max| after standardization:")
    for i, idx in enumerate(top.indices.tolist()):
        print(f"  {morph_cols[idx]:30s} |max|={top.values[i].item():.2e}")

    return (
        x_train, y_train, leiden_train,
        x_val, y_val, leiden_val,
        y_mean, y_std, morph_cols, split_clusters,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_lr_lambda(warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        return 1.0
    return lr_lambda


def get_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("MPS is available; using CPU by default. Pass --device mps to try Apple GPU.")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device {requested!r}; use auto, cuda, mps, or cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def init_wandb(cfg: Config, device: torch.device, parameter_count: int):
    if not cfg.use_wandb:
        return None
    if wandb is None:
        print("wandb is not installed; continuing without W&B logging.")
        return None

    run = wandb.init(
        project=cfg.wandb_project,
        group=cfg.wandb_group or f"cfm_h{cfg.hidden_dim}_r{cfg.n_res_blocks}",
        name=cfg.wandb_name,
        mode=cfg.wandb_mode,
        dir=cfg.output_dir,
        config={
            **asdict(cfg),
            "resolved_device": str(device),
            "parameter_count": parameter_count,
        },
        resume="allow",
    )
    if getattr(run, "url", None):
        print(f"W&B run: {run.url}")
    else:
        print(f"W&B run initialized in {cfg.wandb_mode!r} mode.")
    return run


def wandb_log(run, values: dict[str, object], step: int | None = None) -> None:
    if run is not None:
        run.log(values, step=step)


def wandb_image(path: str):
    if wandb is None:
        return None
    return wandb.Image(path)


def _slugify(value: str) -> str:
    keep = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        elif char in {" ", ".", "/"}:
            keep.append("-")
    slug = "".join(keep).strip("-_")
    return slug or "run"


def make_run_output_dir(cfg: Config, run) -> str:
    base_output_dir = cfg.output_dir
    if cfg.wandb_name:
        run_tag = _slugify(cfg.wandb_name)
    elif run is not None and getattr(run, "name", None):
        run_tag = _slugify(str(run.name))
    else:
        run_tag = time.strftime("%Y%m%d-%H%M%S")

    # Add run.id when available so manually reused W&B names do not overwrite each other.
    if run is not None and getattr(run, "id", None):
        run_tag = f"{run_tag}-{_slugify(str(run.id))}"

    return os.path.join(base_output_dir, run_tag)


def log_output_artifact(run, cfg: Config) -> None:
    if run is None or wandb is None:
        return

    artifact_name = cfg.wandb_artifact_name
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=asdict(cfg),
    )
    for filename in [
        "model_ema.pt",
        "config.json",
        "y_stats.pt",
        "split_clusters.json",
        "loss_curves.png",
        "validation_cluster_metrics.json",
        "validation_marginals.png",
        "validation_correlations.png",
    ]:
        path = os.path.join(cfg.output_dir, filename)
        if os.path.exists(path):
            artifact.add_file(path)

    run.log_artifact(artifact)
    print(f"Logged W&B artifact: {artifact_name}")


def save_step_artifact_checkpoint(
    *,
    run,
    cfg: Config,
    ema: EMA,
    input_dim: int,
    device: torch.device,
    step: int,
) -> None:
    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    ema_model = build_model(cfg, input_dim=input_dim).to(device)
    ema.apply_to(ema_model)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_ema_step_{step}.pt")
    torch.save(ema_model.state_dict(), checkpoint_path)
    print(f"Saved EMA step checkpoint → {checkpoint_path}")

    if run is None or wandb is None:
        return

    artifact_name = cfg.wandb_artifact_name
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata={**asdict(cfg), "checkpoint_step": step},
    )
    artifact.add_file(checkpoint_path, name=f"model_ema_step_{step}.pt")
    for filename in ["config.json", "y_stats.pt", "split_clusters.json"]:
        path = os.path.join(cfg.output_dir, filename)
        if os.path.exists(path):
            artifact.add_file(path)

    run.log_artifact(
        artifact,
        aliases=[f"step-{step}", f"{cfg.model_variant}-step-{step}"],
    )
    print(f"Logged W&B artifact checkpoint at step {step}: {artifact_name}")


def resolve_config(cfg: Config) -> Config:
    valid_variants = {"pca_concat", "pca_film", "full_film"}
    if cfg.model_variant not in valid_variants:
        raise ValueError(
            f"Unknown model_variant {cfg.model_variant!r}; "
            f"choose one of {sorted(valid_variants)}"
        )

    if cfg.output_dir is None:
        cfg.output_dir = os.path.join("outputs", "morphology", cfg.model_variant)
    if cfg.wandb_group is None:
        cfg.wandb_group = cfg.model_variant
    if not cfg.wandb_artifact_name:
        artifact_variant = cfg.model_variant.replace("_", "-")
        cfg.wandb_artifact_name = f"morphology-{artifact_variant}-model"
    return cfg


def train(cfg: Config) -> None:
    cfg = resolve_config(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = get_device(cfg.device)
    print(f"Device: {device}")
    print(f"Model variant: {cfg.model_variant}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    (
        c_train, y_train, _leiden_train,
        c_val, y_val, leiden_val,
        y_mean, y_std,
        morph_cols, split_clusters,
    ) = load_data(cfg)
    input_dim = int(c_train.shape[1])

    num_workers = 2 if device.type == "cuda" else 0
    pin = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(c_train, y_train),
        batch_size=cfg.batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        TensorDataset(c_val, y_val),
        batch_size=cfg.batch_size * 2, shuffle=False,
        drop_last=False, num_workers=num_workers, pin_memory=pin,
    )

    model = build_model(cfg, input_dim=input_dim).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameter_count:,}")
    run = init_wandb(cfg, device, parameter_count)
    cfg.output_dir = make_run_output_dir(cfg, run)
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Run output dir: {cfg.output_dir}")
    if run is not None:
        run.config.update({"output_dir": cfg.output_dir}, allow_val_change=True)

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    torch.save(
        {"mean": y_mean, "std": y_std, "columns": morph_cols},
        os.path.join(cfg.output_dir, "y_stats.pt"),
    )
    with open(os.path.join(cfg.output_dir, "split_clusters.json"), "w") as f:
        json.dump(split_clusters, f, indent=2)

    ema = EMA(model, decay=cfg.ema_decay)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=cfg.sigma)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(cfg.warmup_steps)
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else contextlib.nullcontext()
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_steps: list[int] = []
    val_steps: list[int] = []

    running_loss = 0.0
    running_count = 0
    skipped_nonfinite = 0
    logged_artifact_checkpoint = False
    step = 0
    t0 = time.time()
    data_iter = iter(train_loader)

    model.train()
    while step < cfg.n_steps:
        try:
            c_b, y1_b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            c_b, y1_b = next(data_iter)

        c_b = c_b.to(device, non_blocking=True)
        y1_b = y1_b.to(device, non_blocking=True)

        optimizer.zero_grad()
        with amp_ctx:
            y0 = torch.randn_like(y1_b)
            t, yt, ut = FM.sample_location_and_conditional_flow(y0, y1_b)
            pred = model(yt, t, c_b)
            loss = nn.functional.mse_loss(pred, ut)

        if not torch.isfinite(loss).item():
            skipped_nonfinite += 1
            print(
                f"Skipping non-finite loss batch "
                f"({skipped_nonfinite}/{cfg.max_nonfinite_batches}); "
                f"loss={loss.item()}"
            )
            optimizer.zero_grad(set_to_none=True)
            if skipped_nonfinite >= cfg.max_nonfinite_batches:
                raise RuntimeError(
                    "Too many non-finite training batches. Try --device cpu, "
                    "a smaller learning rate, or inspect morphology outliers."
                )
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        running_loss += loss.item()
        running_count += 1
        step += 1

        if (
            not logged_artifact_checkpoint
            and cfg.artifact_checkpoint_step > 0
            and step >= cfg.artifact_checkpoint_step
        ):
            save_step_artifact_checkpoint(
                run=run,
                cfg=cfg,
                ema=ema,
                input_dim=input_dim,
                device=device,
                step=step,
            )
            logged_artifact_checkpoint = True

        if step % cfg.log_every == 0:
            synchronize_device(device)
            elapsed = time.time() - t0
            avg_loss = running_loss / running_count
            lr_now = scheduler.get_last_lr()[0]
            ms_per_step = elapsed / running_count * 1000
            samples_per_second = cfg.batch_size * running_count / elapsed
            print(
                f"step {step:6d}/{cfg.n_steps}"
                f" | loss={avg_loss:.5f}"
                f" | lr={lr_now:.2e}"
                f" | {ms_per_step:.1f}ms/step"
                f" | {samples_per_second:,.0f} samples/s"
            )
            wandb_log(
                run,
                {
                    "train/loss": avg_loss,
                    "train/lr": lr_now,
                    "train/ms_per_step": ms_per_step,
                    "train/samples_per_second": samples_per_second,
                    "train/skipped_nonfinite_batches": skipped_nonfinite,
                },
                step=step,
            )
            train_losses.append(avg_loss)
            train_steps.append(step)
            running_loss = 0.0
            running_count = 0
            t0 = time.time()

        if step % cfg.val_every == 0:
            val_loss = _val_loss(model, val_loader, FM, device, amp_ctx)
            val_losses.append(val_loss)
            val_steps.append(step)
            print(f"  val loss={val_loss:.5f}")
            wandb_log(run, {"val/loss": val_loss}, step=step)
            model.train()

    # Save EMA weights
    ema_model = build_model(cfg, input_dim=input_dim).to(device)
    ema.apply_to(ema_model)
    ema_path = os.path.join(cfg.output_dir, "model_ema.pt")
    torch.save(ema_model.state_dict(), ema_path)
    print(f"Saved EMA model → {ema_path}")

    loss_curves_path = os.path.join(cfg.output_dir, "loss_curves.png")
    _plot_loss_curves(train_losses, train_steps, val_losses, val_steps, loss_curves_path)
    if run is not None:
        wandb_log(run, {"charts/loss_curves": wandb_image(loss_curves_path)}, step=step)

    if cfg.skip_eval:
        print("Skipping final ODE evaluation because --skip_eval is set.")
    else:
        evaluate(ema_model, c_val, y_val, leiden_val, y_mean, y_std, morph_cols, cfg, device, run)

    log_output_artifact(run, cfg)

    if run is not None:
        run.finish()


@torch.no_grad()
def _val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    FM,
    device: torch.device,
    amp_ctx,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for c_b, y1_b in val_loader:
        c_b = c_b.to(device, non_blocking=True)
        y1_b = y1_b.to(device, non_blocking=True)
        with amp_ctx:
            y0 = torch.randn_like(y1_b)
            t, yt, ut = FM.sample_location_and_conditional_flow(y0, y1_b)
            pred = model(yt, t, c_b)
            loss = nn.functional.mse_loss(pred, ut)
        total += loss.item() * len(c_b)
        count += len(c_b)
    return total / count


# ---------------------------------------------------------------------------
# Evaluation / Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    c_val: torch.Tensor,
    y_val: torch.Tensor,
    leiden_val: np.ndarray,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    morph_cols: list[str],
    cfg: Config,
    device: torch.device,
    run=None,
) -> None:
    model.eval()
    rng = np.random.default_rng(cfg.seed)
    unique_clusters = np.unique(leiden_val.astype(str))
    print(f"Validation evaluation over Leiden clusters: {', '.join(unique_clusters)}")

    y_real_parts = []
    y_gen_parts = []
    cluster_rows = []

    for cluster in unique_clusters:
        cluster_idx = np.flatnonzero(leiden_val.astype(str) == cluster)
        n = min(cfg.n_eval_samples_per_cluster, len(cluster_idx))
        if n == 0:
            continue
        chosen_idx = rng.choice(cluster_idx, size=n, replace=False)
        y_real, y_gen = _sample_morphology(
            model=model,
            c_eval=c_val[chosen_idx],
            y_real_norm=y_val[chosen_idx],
            y_mean=y_mean,
            y_std=y_std,
            cfg=cfg,
            device=device,
        )
        mean_w1 = _mean_w1(y_real, y_gen)
        mean_normalized_w1 = _mean_normalized_w1(y_real, y_gen)
        cluster_rows.append((cluster, len(cluster_idx), n, mean_w1, mean_normalized_w1))
        y_real_parts.append(y_real)
        y_gen_parts.append(y_gen)
        print(
            f"  leiden {cluster:>4s}: evaluated {n:5,d}/{len(cluster_idx):5,d} cells | "
            f"mean W1={mean_w1:.4f} | normalized W1={mean_normalized_w1:.4f}"
        )

    y_real_all = np.concatenate(y_real_parts, axis=0)
    y_gen_all = np.concatenate(y_gen_parts, axis=0)

    validation_mean_w1 = float(_mean_w1(y_real_all, y_gen_all))
    validation_mean_normalized_w1 = float(_mean_normalized_w1(y_real_all, y_gen_all))
    metrics_path = os.path.join(cfg.output_dir, "validation_cluster_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "split": "validation",
                "clusters": [
                    {
                        "leiden": cluster,
                        "n_available": int(n_available),
                        "n_evaluated": int(n_evaluated),
                        "mean_w1": float(mean_w1),
                        "mean_normalized_w1": float(mean_normalized_w1),
                    }
                    for cluster, n_available, n_evaluated, mean_w1, mean_normalized_w1 in cluster_rows
                ],
                "mean_w1": validation_mean_w1,
                "mean_normalized_w1": validation_mean_normalized_w1,
            },
            f,
            indent=2,
        )
    print(
        f"Validation combined: mean W1={validation_mean_w1:.4f}, "
        f"normalized W1={validation_mean_normalized_w1:.4f}"
    )
    print(f"Saved validation cluster metrics to {metrics_path}")
    wandb_log(
        run,
        {
            "validation/mean_w1": validation_mean_w1,
            "validation/mean_normalized_w1": validation_mean_normalized_w1,
            **{
                f"validation/cluster_mean_w1/{cluster}": float(mean_w1)
                for cluster, _, _, mean_w1, _ in cluster_rows
            },
            **{
                f"validation/cluster_mean_normalized_w1/{cluster}": float(mean_normalized_w1)
                for cluster, _, _, _, mean_normalized_w1 in cluster_rows
            },
        },
    )

    _print_marginal_stats(y_real_all, y_gen_all, morph_cols)
    marginals_path = os.path.join(cfg.output_dir, "validation_marginals.png")
    _plot_marginals(
        y_real_all,
        y_gen_all,
        morph_cols,
        marginals_path,
    )
    correlations_path = os.path.join(cfg.output_dir, "validation_correlations.png")
    _plot_correlations(
        y_real_all,
        y_gen_all,
        correlations_path,
    )
    if run is not None:
        wandb_log(
            run,
            {
                "charts/validation_marginals": wandb_image(marginals_path),
                "charts/validation_correlations": wandb_image(correlations_path),
            },
        )
    print(f"Validation eval plots saved to {cfg.output_dir}/")


@torch.no_grad()
def _sample_morphology(
    model: nn.Module,
    c_eval: torch.Tensor,
    y_real_norm: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(c_eval)
    c_eval = c_eval.to(device)

    print(f"Sampling {n} cells with dopri5 ODE solver ...")
    y0 = torch.randn(n, cfg.y_dim, device=device)

    def ode_fn(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_batch = t.expand(y.shape[0])
        return model(y, t_batch, c_eval)

    traj = torchdiffeq.odeint(
        ode_fn,
        y0,
        torch.tensor([0.0, 1.0], device=device),
        method="dopri5",
        rtol=cfg.ode_rtol,
        atol=cfg.ode_atol,
        options={"dtype": torch.float32},
    )
    y_gen_norm = traj[-1].cpu()

    y_mean = y_mean.cpu()
    y_std = y_std.cpu()

    # Inverse z-score, then inverse signed log1p → original CellProfiler scale
    y_real_log = y_real_norm * y_std + y_mean
    y_gen_log = y_gen_norm * y_std + y_mean
    y_real = (torch.sign(y_real_log) * torch.expm1(torch.abs(y_real_log))).numpy()
    y_gen = (torch.sign(y_gen_log) * torch.expm1(torch.abs(y_gen_log))).numpy()

    return y_real, y_gen


def _mean_w1(y_real: np.ndarray, y_gen: np.ndarray) -> float:
    return float(_w1_metrics(y_real, y_gen)["w1"].mean())


def _mean_normalized_w1(y_real: np.ndarray, y_gen: np.ndarray) -> float:
    return float(_w1_metrics(y_real, y_gen)["normalized_w1"].mean())


def _w1_metrics(y_real: np.ndarray, y_gen: np.ndarray, eps: float = 1e-8) -> dict[str, np.ndarray]:
    w1 = np.array([
        wasserstein_distance(y_real[:, i], y_gen[:, i])
        for i in range(y_real.shape[1])
    ])
    real_std = y_real.std(axis=0)
    normalized_w1 = w1 / np.maximum(real_std, eps)
    return {"w1": w1, "real_std": real_std, "normalized_w1": normalized_w1}


def _print_marginal_stats(
    y_real: np.ndarray, y_gen: np.ndarray, cols: list[str]
) -> None:
    metrics = _w1_metrics(y_real, y_gen)
    w1_list = metrics["w1"]
    normalized_w1_list = metrics["normalized_w1"]
    header = (
        f"{'Feature':<42} {'real_mean':>10} {'gen_mean':>10} "
        f"{'real_std':>10} {'gen_std':>10} {'W1':>10} {'norm_W1':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for i, col in enumerate(cols):
        rm, gm = y_real[:, i].mean(), y_gen[:, i].mean()
        rs, gs = y_real[:, i].std(), y_gen[:, i].std()
        w1 = w1_list[i]
        normalized_w1 = normalized_w1_list[i]
        print(
            f"{col:<42} {rm:>10.4f} {gm:>10.4f} {rs:>10.4f} "
            f"{gs:>10.4f} {w1:>10.4f} {normalized_w1:>10.4f}"
        )
    print(f"\nMean W1 distance: {np.mean(w1_list):.4f}")
    print(f"Mean normalized W1 distance: {np.mean(normalized_w1_list):.4f}")


def _plot_marginals(
    y_real: np.ndarray, y_gen: np.ndarray, cols: list[str], path: str
) -> None:
    ncols_g = 8
    nrows_g = math.ceil(len(cols) / ncols_g)
    fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(ncols_g * 3, nrows_g * 2.5))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        ax.hist(y_real[:, i], bins=50, alpha=0.5, density=True, color="steelblue", label="real")
        ax.hist(y_gen[:, i], bins=50, alpha=0.5, density=True, color="tomato", label="gen")
        ax.set_title(col, fontsize=6)
        ax.set_yticks([])
        ax.tick_params(labelsize=5)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.suptitle("Marginals: real (blue) vs generated (red)", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _plot_correlations(y_real: np.ndarray, y_gen: np.ndarray, path: str) -> None:
    corr_r = np.corrcoef(y_real.T)
    corr_g = np.corrcoef(y_gen.T)
    diff = np.abs(corr_r - corr_g)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im0 = axes[0].imshow(corr_r, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_title("Real correlations")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(corr_g, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1].set_title("Generated correlations")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, vmin=0, vmax=1, cmap="hot_r")
    axes[2].set_title("Absolute difference")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _plot_loss_curves(
    train_losses: list[float], train_steps: list[int],
    val_losses: list[float], val_steps: list[int],
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_steps, train_losses, label="train", alpha=0.8)
    if val_losses:
        ax.plot(val_steps, val_losses, label="val", marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss")
    ax.set_title("CFM Training Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CFM morphology model")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=["pca_concat", "pca_film", "full_film"],
    )
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--c_dim", type=int)
    parser.add_argument("--gene_encoder_hidden_dim", type=int)
    parser.add_argument("--n_res_blocks", type=int)
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--val_every", type=int)
    parser.add_argument("--artifact_checkpoint_step", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_group", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_artifact_name", type=str)
    args = parser.parse_args()

    cfg = Config()
    for key in [
        "data_path",
        "output_dir",
        "model_variant",
        "n_steps",
        "batch_size",
        "lr",
        "hidden_dim",
        "c_dim",
        "gene_encoder_hidden_dim",
        "n_res_blocks",
        "log_every",
        "val_every",
        "artifact_checkpoint_step",
        "seed",
        "device",
        "wandb_project",
        "wandb_group",
        "wandb_name",
        "wandb_mode",
        "wandb_artifact_name",
    ]:
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)
    if args.skip_eval:
        cfg.skip_eval = True
    if args.no_wandb:
        cfg.use_wandb = False

    train(cfg)
