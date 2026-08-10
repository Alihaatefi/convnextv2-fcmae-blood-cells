"""Dataset download, in-memory caching, and the four-way split.

The dataset is `sumithsingh/blood-cell-images-for-cancer-detection` on Kaggle:
5,000 microscopy images of peripheral blood cells, 1,000 per class, five classes
(basophil, erythroblast, monocyte, myeloblast, seg_neutrophil).

Images are decoded once, resized to 128x128, and cached as a single uint8 tensor
(~245 MB) so that the 45-run experiment sweep never touches the JPEG decoder again.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_SIZE = 128
SPLIT_SEED = 42  # fixed: the data partition is identical across every run
SPLIT_FRACTIONS = {"ssl": 0.50, "train": 0.25, "val": 0.10}  # test gets the remaining 0.15


def find_image_root(start_path: str | Path) -> str:
    """Walk down until we find the directory whose children are the class folders."""
    for root, dirs, _ in os.walk(start_path):
        if dirs:
            first = os.path.join(root, dirs[0])
            if os.path.isdir(first) and any(
                f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(first)
            ):
                return root
    return str(start_path)


def resolve_data_root(data_root: str | None = None) -> str:
    """Return the directory containing the class folders, downloading from Kaggle if needed."""
    if data_root:
        return find_image_root(data_root)
    import kagglehub

    path = kagglehub.dataset_download("sumithsingh/blood-cell-images-for-cancer-detection")
    return find_image_root(path)


def build_cache(data_root: str, cache_path: Path, img_size: int = IMG_SIZE):
    """Decode every image once into a uint8 tensor and persist it."""
    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.PILToTensor()])
    folder = datasets.ImageFolder(root=data_root, transform=tf)
    images = torch.empty(len(folder), 3, img_size, img_size, dtype=torch.uint8)
    labels = torch.empty(len(folder), dtype=torch.long)
    for i in range(len(folder)):
        img, label = folder[i]
        images[i], labels[i] = img, label
    payload = {"images": images, "labels": labels, "classes": folder.classes}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def load_dataset(data_root: str | None = None, cache_dir: str = ".cache", img_size: int = IMG_SIZE):
    """Load (and build on first call) the in-memory image cache."""
    cache_path = Path(cache_dir) / f"blood_cells_{img_size}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, weights_only=False)
    else:
        payload = build_cache(resolve_data_root(data_root), cache_path, img_size)
    return payload["images"], payload["labels"], payload["classes"]


class CachedImages(Dataset):
    """Serves uint8 CHW tensors; normalization happens on the GPU in the training loop."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def normalize_batch(batch_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [N,3,H,W] on device -> normalized float32."""
    x = batch_uint8.float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Invert the ImageNet normalization, for visualization."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def make_splits(dataset: Dataset):
    """Four disjoint subsets: SSL pre-training / supervised train / val / test.

    Uses a fixed generator seed so the partition -- and therefore the test set --
    is byte-identical across every condition and every run seed.
    """
    total = len(dataset)
    n_ssl = int(SPLIT_FRACTIONS["ssl"] * total)
    n_train = int(SPLIT_FRACTIONS["train"] * total)
    n_val = int(SPLIT_FRACTIONS["val"] * total)
    n_test = total - n_ssl - n_train - n_val
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    ssl, train, val, test = random_split(dataset, [n_ssl, n_train, n_val, n_test], generator=generator)
    return {"ssl": ssl, "train": train, "val": val, "test": test}


def stratified_subset(subset: Subset, labels: torch.Tensor, fraction: float, seed: int) -> Subset:
    """Take a class-balanced fraction of a Subset.

    Plain random sampling at 1% (12 images over 5 classes) can drop a class entirely,
    which would make the low-label end of the sweep meaningless. This samples per class.
    """
    if fraction >= 1.0:
        return subset

    indices = torch.as_tensor(subset.indices)
    subset_labels = labels[indices]
    generator = torch.Generator().manual_seed(seed)

    chosen = []
    for class_id in subset_labels.unique().tolist():
        class_positions = (subset_labels == class_id).nonzero(as_tuple=True)[0]
        n_keep = max(1, int(round(fraction * len(class_positions))))
        perm = torch.randperm(len(class_positions), generator=generator)[:n_keep]
        chosen.append(class_positions[perm])

    chosen = torch.cat(chosen).sort().values
    return Subset(subset.dataset, indices[chosen].tolist())
