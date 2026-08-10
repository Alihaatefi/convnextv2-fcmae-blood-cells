"""How much of this task is solvable from colour alone?

Looking at sample images from the five classes, two visually distinct background
populations are apparent: basophil / erythroblast / monocyte sit on a peach-toned
field, while myeloblast / seg_neutrophil sit on a blue-white one. Those are also
exactly the classes the model finds easiest and hardest -- which raises the
possibility that a chunk of the reported accuracy comes from staining and
background colour rather than from cell morphology.

This script tests that directly, with no deep learning involved. It fits a
multinomial logistic regression on the same 1,250-image training split and
evaluates on the same 750-image test split, using deliberately morphology-free
features:

    mean-rgb       3 numbers: the average colour of the whole image
    corner-rgb     12 numbers: the average colour of the four 16x16 CORNERS,
                   which contain background only -- the cell is centred
    hist-rgb       48 numbers: a 16-bin per-channel colour histogram
    thumb-8x8      192 numbers: the image crushed to 8x8, destroying morphology

Chance is 20%. Whatever these features reach is a floor that the 93% deep model
should be measured against -- it is the part of the score that needs no
understanding of cells at all.

Usage:
    python -m src.shortcut_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CachedImages, load_dataset, make_splits


def mean_rgb(images: torch.Tensor) -> np.ndarray:
    return images.float().mean(dim=(2, 3)).numpy()


def corner_rgb(images: torch.Tensor, size: int = 16) -> np.ndarray:
    """Mean colour of the four corner patches -- background only, no cell."""
    corners = [
        images[:, :, :size, :size], images[:, :, :size, -size:],
        images[:, :, -size:, :size], images[:, :, -size:, -size:],
    ]
    return torch.cat([c.float().mean(dim=(2, 3)) for c in corners], dim=1).numpy()


def hist_rgb(images: torch.Tensor, bins: int = 16) -> np.ndarray:
    out = np.empty((len(images), 3 * bins), dtype=np.float32)
    edges = torch.linspace(0, 256, bins + 1)
    for i, image in enumerate(images):
        features = [torch.histogram(image[c].float(), bins=edges)[0] for c in range(3)]
        out[i] = torch.cat(features).numpy() / (image.shape[1] * image.shape[2])
    return out


def thumb(images: torch.Tensor, size: int = 8) -> np.ndarray:
    small = F.adaptive_avg_pool2d(images.float(), size)
    return small.flatten(1).numpy()


FEATURES = {
    "mean-rgb": (mean_rgb, "average colour of the whole image (3 dims)"),
    "corner-rgb": (corner_rgb, "average colour of the four corners -- background only (12 dims)"),
    "hist-rgb": (hist_rgb, "16-bin per-channel colour histogram (48 dims)"),
    "thumb-8x8": (thumb, "image crushed to 8x8 (192 dims)"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default="results/metrics/shortcut_check.json")
    args = parser.parse_args()

    images, labels, class_names = load_dataset(args.data_root)
    splits = make_splits(CachedImages(images, labels))
    train_idx = torch.as_tensor(splits["train"].indices)
    test_idx = torch.as_tensor(splits["test"].indices)

    print(f"train {len(train_idx)} / test {len(test_idx)} images, "
          f"{len(class_names)} classes (chance = {100 / len(class_names):.0f}%)\n")

    assert not (set(train_idx.tolist()) & set(test_idx.tolist())), "train/test overlap"

    results = {}
    y_train = labels[train_idx].numpy()
    y_test = labels[test_idx].numpy()

    for name, (extractor, description) in FEATURES.items():
        x_train = extractor(images[train_idx])
        x_test = extractor(images[test_idx])

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(x_train, y_train)
        accuracy = float(clf.score(x_test, y_test))
        results[name] = {"test_accuracy": accuracy, "n_features": x_train.shape[1],
                         "description": description}
        print(f"  {name:<12} {accuracy * 100:5.2f}%   {description}")

    # Permutation control: with the training labels shuffled the same pipeline must
    # collapse to chance. If it does not, the result above is a leak, not a finding.
    x_train = FEATURES["hist-rgb"][0](images[train_idx])
    x_test = FEATURES["hist-rgb"][0](images[test_idx])
    shuffled = np.random.default_rng(0).permutation(y_train)
    control = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    control.fit(x_train, shuffled)
    control_accuracy = float(control.score(x_test, y_test))
    results["_permutation_control"] = {
        "test_accuracy": control_accuracy,
        "description": "hist-rgb with shuffled training labels; must land at chance",
    }
    print(f"\n  {'permuted':<12} {control_accuracy * 100:5.2f}%   "
          f"sanity check -- chance is {100 / len(class_names):.0f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
