"""
Training script for conditional flow matching on MNIST.

Usage:
  python main.py
  python main.py --epochs 20 --batch_size 128 --num_channels 64
  python main.py --no_wandb
"""

import argparse
import os
import time
from dataclasses import dataclass, asdict

import torch
import torchdiffeq
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from cfm_model import CFMUNet


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    sigma: float = 0.0
    epochs: int = 10
    batch_size: int = 64
    num_channels: int = 32
    num_res_blocks: int = 1
    cond_dim: int = 2
    cond_hidden_dim: int = 64
    lr: float = 1e-4
    solver: str = "dopri5"
    sample_freq: int = 5
    data_dir: str = "./data"
    save_dir: str = "./models/cond_mnist"
    wandb_project: str = "DL-Seminar"


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_channels", type=int)
    parser.add_argument("--num_res_blocks", type=int)
    parser.add_argument("--cond_dim", type=int)
    parser.add_argument("--cond_hidden_dim", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--solver", type=str)
    parser.add_argument("--sample_freq", type=int)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    for field in vars(cfg):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
    cfg._no_wandb = args.no_wandb
    return cfg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_class_embed(device: torch.device) -> torch.Tensor:
    angles = 2 * torch.pi * torch.arange(10) / 10
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).to(device)


@torch.no_grad()
def log_samples(model, class_embed_2d, device, epoch):
    model.eval()
    traj = torchdiffeq.odeint(
        lambda t, x: model(t, x, class_embed_2d[torch.arange(10, device=device).repeat_interleave(10)]),
        torch.randn(100, 1, 28, 28, device=device),
        torch.linspace(0, 1, 2, device=device),
        method="euler",
        options={"dtype": torch.float32},
    )
    grid = make_grid(traj[-1].clip(-1, 1), value_range=(-1, 1), nrow=10)
    wandb.log({"samples": wandb.Image(grid), "epoch": epoch})
    model.train()


def train(cfg: Config):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if not cfg._no_wandb:
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))
        assert wandb.run is not None
        run_name = wandb.run.name
    else:
        run_name = f"local_{int(time.time())}"

    run_dir = os.path.join(cfg.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    class_embed_2d = build_class_embed(device)

    model = CFMUNet(
        dim=(1, 28, 28),
        num_channels=cfg.num_channels,
        num_classes=10,
        num_res_blocks=cfg.num_res_blocks,
        class_cond=True,
        cond_dim=cfg.cond_dim,
        cond_hidden_dim=cfg.cond_hidden_dim,
    ).to(device)

    FM = ConditionalFlowMatcher(sigma=cfg.sigma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        for i, (x1, y_int) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            optimizer.zero_grad()
            x1 = x1.to(device)
            y = class_embed_2d[y_int]
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

            loss = torch.mean((model(t, xt, y) - ut) ** 2)
            loss.backward()
            optimizer.step()

            if not cfg._no_wandb:
                wandb.log({"loss": loss.item(), "epoch": epoch, "step": i})

        if not cfg._no_wandb and (epoch + 1) % cfg.sample_freq == 0 or (epoch +1) == cfg.epochs:
            log_samples(model, class_embed_2d, device, epoch)

    path = os.path.join(run_dir, "model.pt")
    torch.save(model, path)
    print(f"Saved model to {path}")

    if not cfg._no_wandb:
        wandb.finish()


def main():
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
