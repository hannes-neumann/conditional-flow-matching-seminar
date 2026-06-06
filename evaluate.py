"""
Metric-based evaluation for a conditional flow matching model trained on MNIST.

Metrics:
  - FID  (Fréchet Inception Distance)  — overall distributional quality
  - Classifier Accuracy                — how well generated images match their condition
    reported both overall and per-class

Usage:
  python evaluate.py --model models/cond_mnist/cond_mnist_9_... \
                     --data ../data \
                     --n_per_class 500 \
                     --solver euler

Dependencies (beyond the training notebook):
  pip install torchmetrics[image]
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torchdiffeq
import wandb
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from cfm_model import CFMUNet  # noqa: F401 — required so torch.load can unpickle CFMUNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_class_embed(device: torch.device) -> torch.Tensor:
    angles = 2 * torch.pi * torch.arange(10) / 10
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).to(device)


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    class_embed_2d: torch.Tensor,
    device: torch.device,
    n_per_class: int = 500,
    solver: str = "euler",
    batch_size: int = 256,
    euler_steps: int = 10, 
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (images, labels) where images are in [-1, 1]."""
    model.eval()
    all_imgs, all_labels = [], []

    for cls in tqdm(range(10), desc="Generating"):
        generated = []
        for start in range(0, n_per_class, batch_size):
            n = min(batch_size, n_per_class - start)
            y = class_embed_2d[cls].unsqueeze(0).expand(n, -1)
            x0 = torch.randn(n, 1, 28, 28, device=device)
            traj = torchdiffeq.odeint(
                lambda t, x: model(t, x, y),
                x0,
                torch.linspace(0, 1, euler_steps, device=device),
                method=solver,
                options={"dtype": torch.float32},
            )
            generated.append(traj[-1].clip(-1, 1).cpu())
        all_imgs.append(torch.cat(generated))
        all_labels.append(torch.full((n_per_class,), cls, dtype=torch.long))

    return torch.cat(all_imgs), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(
    generated: torch.Tensor,
    data_root: str,
    n_real: int = 1000,
) -> float:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    real_ds = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    real_loader = torch.utils.data.DataLoader(real_ds, batch_size=256, shuffle=True)

    fid = FrechetInceptionDistance(feature=2048, normalize=True)  # Mayber switch back to 192

    # Real images (FID expects 3-channel float in [0, 1])
    seen = 0
    for imgs, _ in real_loader:
        imgs = ((imgs + 1) / 2).repeat(1, 3, 1, 1).cpu()
        fid.update(imgs, real=True)
        seen += imgs.shape[0]
        if seen >= n_real:
            break

    gen = ((generated + 1) / 2).repeat(1, 3, 1, 1)
    for start in range(0, len(gen), 256):
        fid.update(gen[start : start + 256], real=False)

    return fid.compute().item()


# ---------------------------------------------------------------------------
# Classifier accuracy
# ---------------------------------------------------------------------------

class _Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(data_root: str, device: torch.device, epochs: int = 3) -> _Classifier:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    clf = _Classifier().to(device)
    opt = torch.optim.Adam(clf.parameters())
    loss_fn = nn.CrossEntropyLoss()

    clf.train()
    for epoch in range(epochs):
        for imgs, labels in tqdm(loader, desc=f"Classifier training {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss_fn(clf(imgs), labels).backward()
            opt.step()

    return clf


@torch.no_grad()
def compute_accuracy(
    clf: _Classifier,
    generated: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, float | dict[int, float]]:
    clf.eval()
    preds_all, labels_all = [], []

    for start in range(0, len(generated), 256):
        imgs = generated[start : start + 256].to(device)
        preds_all.append(clf(imgs).argmax(dim=1).cpu())
        labels_all.append(labels[start : start + 256])

    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    overall = (preds == labels).float().mean().item()

    per_class = {}
    for cls in range(10):
        mask = labels == cls
        per_class[cls] = (preds[mask] == labels[mask]).float().mean().item()

    return {"overall": overall, "per_class": per_class}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved model.pt file")
    parser.add_argument("--data", default="../data", help="MNIST data root")
    parser.add_argument("--n_per_class", type=int, default=500)
    parser.add_argument("--solver", default="euler", choices=["euler", "dopri5"])
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Resume wandb run from training if run_info.json exists in the model's directory
    run_dir = os.path.dirname(args.model)
    run_info_path = os.path.join(run_dir, "run_info.json")
    use_wandb = not args.no_wandb and os.path.exists(run_info_path)

    if use_wandb:
        with open(run_info_path) as f:
            run_info = json.load(f)
        wandb.init(
            project=run_info["wandb_project"],
            id=run_info.get("wandb_run_id"),
            resume="allow",
        )
        print(f"Logging to W&B run: {run_info.get('run_name')}")

    model = torch.load(args.model, weights_only=False, map_location=device)
    model.eval()

    class_embed_2d = build_class_embed(device)

    print(f"\nGenerating {args.n_per_class} samples per class ({args.n_per_class * 10} total)...")
    generated, labels = generate_samples(
        model, class_embed_2d, device,
        n_per_class=args.n_per_class,
        solver=args.solver,
    )

    print("\nComputing FID...")
    fid_score = compute_fid(generated, args.data)
    print(f"FID: {fid_score:.2f}")

    print("\nTraining reference classifier...")
    clf = train_classifier(args.data, device)

    acc = compute_accuracy(clf, generated, labels, device)
    print(f"\nClassifier Accuracy (overall): {acc['overall']:.3f}")
    print("Per-class accuracy:")
    for cls, a in acc["per_class"].items():
        print(f"  class {cls}: {a:.3f}")

    if use_wandb:
        wandb.log({
            "eval/fid": fid_score,
            "eval/accuracy": acc["overall"],
            **{f"eval/accuracy_class_{c}": a for c, a in acc["per_class"].items()},
        })
        wandb.finish()


if __name__ == "__main__":
    main()
