"""Reproduce the original course project end to end, with every figure.

Runs the submitted pipeline -- FCMAE pre-training on 2,500 unlabelled images, then
supervised fine-tuning on 1,250 labelled ones -- and writes the qualitative figures
and the classification report. Unlike the original notebook this is fully seeded,
so the numbers are reproducible.

Usage:
    python -m src.reproduce --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import auc, classification_report, roc_curve  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402

from .data import (CachedImages, denormalize, load_dataset, make_splits,  # noqa: E402
                   normalize_batch)
from .engine import evaluate, make_loader, pretrain_ssl, set_seed, train_classifier  # noqa: E402
from .model import FCMAE, Classifier, ConvNeXtV2Encoder, count_parameters  # noqa: E402
from .theme import SUFFIX, THEMES, sequential_cmap, style_axes  # noqa: E402


def save(fig, figures_dir: Path, stem: str, theme_name: str, theme, dpi=200):
    """Charts render at 200 dpi; photographic grids at 140, which keeps them crisp
    on a README while cutting the committed file size by roughly half."""
    path = figures_dir / f"{stem}{SUFFIX[theme_name]}.png"
    fig.savefig(path, dpi=dpi, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def figure_dataset_overview(images, labels, class_names, figures_dir, theme_name, per_class=3):
    """Samples per class. The class counts are a single sentence, not a bar chart --
    five identical bars carry one number. What the images *do* show is the staining
    split: the first three classes sit on a peach field, the last two on blue-white.
    That is the pattern src/shortcut_check.py goes on to quantify."""
    theme = THEMES[theme_name]
    counts = labels.bincount().tolist()
    generator = torch.Generator().manual_seed(0)

    fig, axes = plt.subplots(per_class, len(class_names),
                             figsize=(2.05 * len(class_names), 2.05 * per_class + 0.9),
                             facecolor=theme["surface"])
    for col, name in enumerate(class_names):
        candidates = (labels == col).nonzero(as_tuple=True)[0]
        picks = candidates[torch.randperm(len(candidates), generator=generator)[:per_class]]
        for row in range(per_class):
            ax = axes[row, col]
            ax.imshow(images[picks[row]].permute(1, 2, 0).numpy())
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=10, color=theme["ink"], pad=8)

    fig.suptitle(f"{sum(counts):,} images, {counts[0]:,} per class -- perfectly balanced",
                 fontsize=12, color=theme["ink"], y=1.005)
    fig.text(0.5, 0.963, "note the two background populations: peach for the first three "
                         "classes, blue-white for the last two",
             ha="center", fontsize=9.5, color=theme["secondary"])
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    save(fig, figures_dir, "dataset_overview", theme_name, theme, dpi=140)


def figure_ssl_reconstruction(ssl_model, loader, device, figures_dir, theme_name, mask_ratio, n=8):
    theme = THEMES[theme_name]
    ssl_model.eval()
    with torch.no_grad():
        batch, _ = next(iter(loader))
        batch = normalize_batch(batch[:n].to(device))
        reconstruction, original, keep = ssl_model(batch, mask_ratio=mask_ratio)

    # NOTE: the masked input is shown as the model actually sees it -- zero in
    # NORMALISED space, which denormalises to the ImageNet mean colour, not black.
    # The original notebook multiplied the denormalised image by the mask, which
    # painted the masked pixels black and misrepresented the model's input.
    rows = [
        ("Original", torch.stack([denormalize(im) for im in original])),
        ("Masked input (60%)", torch.stack([denormalize(im) for im in original * keep])),
        ("Reconstruction", torch.stack([denormalize(im) for im in reconstruction])),
    ]

    fig, axes = plt.subplots(3, n, figsize=(1.55 * n, 5.1), facecolor=theme["surface"])
    for row, (title, tensor) in enumerate(rows):
        for col in range(n):
            ax = axes[row, col]
            ax.imshow(tensor[col].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            ax.axis("off")
            if col == 0:
                ax.text(-0.08, 0.5, title, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=10, color=theme["secondary"])
    fig.suptitle("FCMAE reconstruction on held-out images", fontsize=12,
                 color=theme["ink"], y=1.0)
    save(fig, figures_dir, "ssl_reconstruction", theme_name, theme, dpi=140)


def figure_training_curves(history, figures_dir, theme_name):
    theme = THEMES[theme_name]
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=theme["surface"])

    panels = [("Cross-entropy loss", "train_loss", "val_loss"),
              ("Accuracy (%)", "train_acc", "val_acc")]
    for ax, (ylabel, train_key, val_key) in zip(axes, panels):
        for slot, (key, name) in enumerate([(train_key, "Train"), (val_key, "Validation")]):
            ax.plot(epochs, history[key], color=theme["series"][slot], linewidth=2,
                    marker="o", markersize=7, markeredgecolor=theme["surface"],
                    markeredgewidth=2, label=name, zorder=3)
        style_axes(ax, theme)
        ax.set_xlabel("Epoch", fontsize=10, color=theme["secondary"])
        ax.set_ylabel(ylabel, fontsize=10, color=theme["secondary"])
        ax.legend(frameon=False, fontsize=9, labelcolor=theme["secondary"])

    axes[0].axvline(history["best_epoch"] + 1, color=theme["muted"], linewidth=1, zorder=1)
    axes[0].annotate("best (restored)", xy=(history["best_epoch"] + 1, max(history["train_loss"])),
                     xytext=(4, -2), textcoords="offset points", fontsize=8,
                     color=theme["muted"])
    fig.suptitle("Supervised fine-tuning, 1,250 labelled images", fontsize=12,
                 color=theme["ink"], x=0.008, ha="left")
    fig.tight_layout()
    save(fig, figures_dir, "training_curves", theme_name, theme)


def figure_confusion_matrix(matrix, class_names, figures_dir, theme_name):
    theme = THEMES[theme_name]
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6.8, 5.8), facecolor=theme["surface"])
    image = ax.imshow(matrix, cmap=sequential_cmap(theme), vmin=0)

    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted", fontsize=10, color=theme["secondary"])
    ax.set_ylabel("True", fontsize=10, color=theme["secondary"])
    ax.set_title("Confusion matrix, 750 held-out images", fontsize=11,
                 color=theme["ink"], loc="left", pad=10)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    # Value labels stay in ink tokens; flip to the surface colour on dark cells.
    threshold = matrix.max() * 0.6
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10,
                    color=theme["surface"] if matrix[i, j] > threshold else theme["ink"])

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=theme["muted"], labelsize=8, length=0)
    fig.tight_layout()
    save(fig, figures_dir, "confusion_matrix", theme_name, theme)


def figure_roc(y_true, y_score, class_names, figures_dir, theme_name):
    theme = THEMES[theme_name]
    binarized = label_binarize(y_true, classes=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor=theme["surface"])

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(binarized[:, i], y_score[:, i])
        ax.plot(fpr, tpr, color=theme["series"][i], linewidth=2, zorder=3,
                label=f"{name}  (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=theme["muted"], linewidth=1, zorder=1)

    style_axes(ax, theme)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate", fontsize=10, color=theme["secondary"])
    ax.set_ylabel("True positive rate", fontsize=10, color=theme["secondary"])
    ax.set_title("One-vs-rest ROC", fontsize=11, color=theme["ink"], loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=theme["secondary"])
    fig.tight_layout()
    save(fig, figures_dir, "roc_curves", theme_name, theme)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="atto")
    parser.add_argument("--ssl-epochs", type=int, default=15)
    parser.add_argument("--ft-epochs", type=int, default=50)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--metrics-dir", default="results/metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figures_dir, metrics_dir = Path(args.figures_dir), Path(args.metrics_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    images, labels, class_names = load_dataset(args.data_root)
    splits = make_splits(CachedImages(images, labels))
    print(f"device: {device}")
    print("split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    print("\n[1/4] dataset overview")
    for theme_name in THEMES:
        figure_dataset_overview(images, labels, class_names, figures_dir, theme_name)

    print(f"\n[2/4] FCMAE pre-training ({args.ssl_epochs} epochs, {len(splits['ssl'])} unlabelled)")
    set_seed(args.seed)
    ssl_model = FCMAE(ConvNeXtV2Encoder(variant=args.variant)).to(device)
    print(f"  encoder params: {count_parameters(ssl_model.encoder):,}")
    ssl_loader = make_loader(splits["ssl"], batch_size=args.batch_size, shuffle=True, seed=0)
    ssl_losses, ssl_best_epoch = pretrain_ssl(ssl_model, ssl_loader, device,
                                              epochs=args.ssl_epochs,
                                              mask_ratio=args.mask_ratio)
    test_loader = make_loader(splits["test"], batch_size=args.batch_size)
    for theme_name in THEMES:
        figure_ssl_reconstruction(ssl_model, test_loader, device, figures_dir,
                                  theme_name, args.mask_ratio)

    print(f"\n[3/4] supervised fine-tuning ({len(splits['train'])} labelled)")
    set_seed(args.seed + 1000)
    model = Classifier(ssl_model.encoder, num_classes=len(class_names)).to(device)
    history = train_classifier(
        model,
        make_loader(splits["train"], batch_size=args.batch_size, shuffle=True, seed=args.seed),
        make_loader(splits["val"], batch_size=args.batch_size),
        device, epochs=args.ft_epochs,
    )
    for theme_name in THEMES:
        figure_training_curves(history, figures_dir, theme_name)

    print("\n[4/4] evaluation on the held-out test set")
    metrics, y_true, y_pred, y_score = evaluate(model, test_loader, device, class_names)
    for theme_name in THEMES:
        figure_confusion_matrix(metrics["confusion_matrix"], class_names, figures_dir, theme_name)
        figure_roc(y_true, y_score, class_names, figures_dir, theme_name)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    (metrics_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (metrics_dir / "reproduce_metrics.json").write_text(
        json.dumps({"seed": args.seed, "ssl_epochs": args.ssl_epochs,
                    "ssl_losses": ssl_losses, "ssl_best_epoch": ssl_best_epoch,
                    "history": history, **metrics}, indent=2),
        encoding="utf-8")

    print("\n" + report)
    print(f"test accuracy {metrics['accuracy'] * 100:.2f}%  "
          f"macro-F1 {metrics['macro_f1']:.4f}  macro-AUC {metrics['macro_auc']:.4f}")
    torch.save({"model": model.state_dict(), "class_names": class_names, "args": vars(args)},
               Path("checkpoints") / f"reproduce_seed{args.seed}.pt")


if __name__ == "__main__":
    main()
