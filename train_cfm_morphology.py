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

import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchdiffeq
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, TensorDataset

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    data_path: str = "data/whole_dataset.h5ad"
    output_dir: str = "outputs"

    # Architecture
    y_dim: int = 56
    c_dim: int = 50
    hidden_dim: int = 512
    n_res_blocks: int = 6
    time_emb_dim: int = 128

    # Training
    seed: int = 42
    batch_size: int = 1024
    n_steps: int = 50_000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    ema_decay: float = 0.9999
    sigma: float = 0.0

    # Logging / validation
    log_every: int = 100
    val_every: int = 1000

    # Eval
    n_eval_samples: int = 4096
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-5


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


class ResBlock(nn.Module):
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


class MorphologyVectorField(nn.Module):
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
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_res_blocks)])
        self.output_head = nn.Linear(hidden_dim, y_dim)

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_sin = sinusoidal_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_sin)
        x = torch.cat([y_t, t_emb, c], dim=-1)
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_head(x)


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
}


def get_morphology_columns(obs_columns: list[str]) -> list[str]:
    cols = [c for c in obs_columns if c[0].isupper()]
    cols = [c for c in cols if c not in _DROP_COLUMNS]
    cols = [c for c in cols if not c.startswith("SpatialMoment_")]
    assert len(cols) == 56, (
        f"Expected 56 morphological features, got {len(cols)}.\n"
        f"Columns: {cols}"
    )
    return cols


def load_data(cfg: Config):
    print(f"Loading {cfg.data_path} ...")
    adata = anndata.read_h5ad(cfg.data_path)

    c = torch.tensor(np.array(adata.obsm["X_pca"]), dtype=torch.float32)
    assert c.shape[1] == cfg.c_dim, f"Expected c.shape[1] == {cfg.c_dim}, got {c.shape[1]}"

    morph_cols = get_morphology_columns(list(adata.obs.columns))
    y = torch.tensor(adata.obs[morph_cols].values.astype(np.float32), dtype=torch.float32)
    assert y.shape[1] == cfg.y_dim, f"Expected y.shape[1] == {cfg.y_dim}, got {y.shape[1]}"

    print(f"Loaded: c={tuple(c.shape)}, y={tuple(y.shape)}")

    # Impute NaN morphological features with per-feature median
    nan_mask = torch.isnan(y)
    if nan_mask.any():
        n_nan_cells = nan_mask.any(dim=1).sum().item()
        n_nan_vals  = nan_mask.sum().item()
        print(f"Imputing {n_nan_vals} NaN values across {n_nan_cells} cells ({n_nan_cells/len(y)*100:.1f}%) with per-feature median")
        for j in range(y.shape[1]):
            col = y[:, j]
            median = col[~torch.isnan(col)].median()
            y[:, j] = torch.where(torch.isnan(col), median, col)
    assert not torch.isnan(y).any() and not torch.isnan(c).any(), "NaN values remain after imputation"

    # Raw CellProfiler features span many orders of magnitude and include negative values
    # (Hu moments, central moments). Signed log1p compresses the dynamic range while
    # preserving sign, making z-score standardization well-behaved.
    y = torch.sign(y) * torch.log1p(torch.abs(y))
    print(
        f"After signed log1p: min={y.min().item():.3f}, max={y.max().item():.3f}, "
        f"any NaN={torch.isnan(y).any().item()}, any inf={torch.isinf(y).any().item()}"
    )

    torch.manual_seed(cfg.seed)
    n = len(c)
    n_val = int(n * 0.1)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    c_train, c_val = c[train_idx], c[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0).clamp(min=1e-8)

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    abs_max = y_train.abs().max(dim=0).values
    top = torch.topk(abs_max, k=5)
    print("Top-5 features by |max| after standardization:")
    for i, idx in enumerate(top.indices.tolist()):
        print(f"  {morph_cols[idx]:30s} |max|={top.values[i].item():.2e}")

    return c_train, y_train, c_val, y_val, y_mean, y_std, morph_cols


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_lr_lambda(warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        return 1.0
    return lr_lambda


def train(cfg: Config) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    c_train, y_train, c_val, y_val, y_mean, y_std, morph_cols = load_data(cfg)
    torch.save(
        {"mean": y_mean, "std": y_std, "columns": morph_cols},
        os.path.join(cfg.output_dir, "y_stats.pt"),
    )

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

    model = MorphologyVectorField(
        y_dim=cfg.y_dim,
        c_dim=cfg.c_dim,
        hidden_dim=cfg.hidden_dim,
        n_res_blocks=cfg.n_res_blocks,
        time_emb_dim=cfg.time_emb_dim,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ema = EMA(model, decay=cfg.ema_decay)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=cfg.sigma)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(cfg.warmup_steps)
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
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

        if step % cfg.log_every == 0:
            avg_loss = running_loss / running_count
            lr_now = scheduler.get_last_lr()[0]
            ms_per_step = (time.time() - t0) / cfg.log_every * 1000
            print(
                f"step {step:6d}/{cfg.n_steps}"
                f" | loss={avg_loss:.5f}"
                f" | lr={lr_now:.2e}"
                f" | {ms_per_step:.1f}ms/step"
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
            model.train()

    # Save EMA weights
    ema_model = MorphologyVectorField(
        y_dim=cfg.y_dim,
        c_dim=cfg.c_dim,
        hidden_dim=cfg.hidden_dim,
        n_res_blocks=cfg.n_res_blocks,
        time_emb_dim=cfg.time_emb_dim,
    ).to(device)
    ema.apply_to(ema_model)
    ema_path = os.path.join(cfg.output_dir, "model_ema.pt")
    torch.save(ema_model.state_dict(), ema_path)
    print(f"Saved EMA model → {ema_path}")

    _plot_loss_curves(train_losses, train_steps, val_losses, val_steps,
                      os.path.join(cfg.output_dir, "loss_curves.png"))

    evaluate(ema_model, c_val, y_val, y_mean, y_std, morph_cols, cfg, device)


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
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    morph_cols: list[str],
    cfg: Config,
    device: torch.device,
) -> None:
    model.eval()

    n = min(cfg.n_eval_samples, len(c_val))
    c_eval = c_val[:n].to(device)
    y_real_norm = y_val[:n]

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

    _print_marginal_stats(y_real, y_gen, morph_cols)
    _plot_marginals(y_real, y_gen, morph_cols, os.path.join(cfg.output_dir, "marginals.png"))
    _plot_correlations(y_real, y_gen, os.path.join(cfg.output_dir, "correlations.png"))
    print(f"Eval plots saved to {cfg.output_dir}/")


def _print_marginal_stats(
    y_real: np.ndarray, y_gen: np.ndarray, cols: list[str]
) -> None:
    w1_list = []
    header = f"{'Feature':<42} {'real_mean':>10} {'gen_mean':>10} {'real_std':>10} {'gen_std':>10} {'W1':>10}"
    print("\n" + header)
    print("-" * len(header))
    for i, col in enumerate(cols):
        rm, gm = y_real[:, i].mean(), y_gen[:, i].mean()
        rs, gs = y_real[:, i].std(), y_gen[:, i].std()
        w1 = wasserstein_distance(y_real[:, i], y_gen[:, i])
        w1_list.append(w1)
        print(f"{col:<42} {rm:>10.4f} {gm:>10.4f} {rs:>10.4f} {gs:>10.4f} {w1:>10.4f}")
    print(f"\nMean W1 distance: {np.mean(w1_list):.4f}")


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
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    cfg = Config()
    for key in ["data_path", "output_dir", "n_steps", "batch_size", "lr", "hidden_dim", "seed"]:
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)

    train(cfg)
