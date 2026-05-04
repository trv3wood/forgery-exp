from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torch import nn
from tqdm import tqdm

from .data import CLASS_NAMES, IMAGE_SIZE, build_loader
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple image forgery detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rao2016_simple"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, leave=False, desc="train" if is_train else "eval"):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return float(np.mean(losses)), float(accuracy_score(y_true, y_pred)), y_true, y_pred


def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "loss_accuracy.png", dpi=180)
    plt.close(fig)


def plot_confusion(y_true: list[int], y_pred: list[int], output_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_samples(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    max_samples: int = 8,
) -> None:
    model.eval()
    images, labels = next(iter(loader))
    images = images[:max_samples].to(device)
    labels = labels[:max_samples]

    with torch.no_grad():
        probs = torch.softmax(model(images), dim=1).detach().cpu()
    preds = probs.argmax(dim=1)

    images = images.detach().cpu()
    images = images * 0.5 + 0.5

    cols = min(4, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_arr = np.array(axes).reshape(-1)

    for ax in axes_arr:
        ax.axis("off")

    for idx, ax in enumerate(axes_arr[: len(images)]):
        img = images[idx].permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        true_name = CLASS_NAMES[int(labels[idx])]
        pred_name = CLASS_NAMES[int(preds[idx])]
        conf = float(probs[idx, preds[idx]])
        ax.set_title(f"T: {true_name}\nP: {pred_name} ({conf:.2f})", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "sample_predictions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        split: build_loader(
            data_dir=args.data_dir,
            split=split,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        for split in ["train", "val", "test"]
    }

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, loaders["val"], criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": args.image_size,
                    "val_acc": best_val_acc,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, test_true, test_pred = run_epoch(model, loaders["test"], criterion, device)

    metrics = {
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history,
        "class_names": CLASS_NAMES,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_history(history, args.output_dir)
    plot_confusion(test_true, test_pred, args.output_dir)
    plot_samples(model, loaders["test"], device, args.output_dir)

    print(f"best checkpoint: {best_path}")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
