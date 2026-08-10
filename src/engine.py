"""Training loops, early stopping, and evaluation."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import normalize_batch


def set_seed(seed: int):
    """Seed every source of randomness we touch (the original notebook seeded only the split)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size=64, shuffle=False, seed=None):
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)
    # num_workers=0: the dataset is already resident in RAM, so workers would add
    # Windows process-spawn overhead for no I/O benefit.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


class EarlyStopping:
    """Stop when validation loss has not improved by `min_delta` for `patience` epochs."""

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None or val_loss <= self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model):
        if self.best_state is None:
            raise RuntimeError("EarlyStopping was never called; no weights to restore.")
        model.load_state_dict(self.best_state)
        return model


def pretrain_ssl(model, loader, device, epochs=15, lr=1e-3, weight_decay=0.05,
                 mask_ratio=0.6, verbose=True, progress=False):
    """FCMAE-style masked reconstruction pre-training.

    The weights left in `model` on return are the ones from the LOWEST-loss epoch,
    not the last. The optimizer here is the one the original project used -- constant
    AdamW, no warmup, no cosine decay -- and at 100 epochs it spikes late: in 2 of 3
    seeds the final epoch landed 10x above the minimum reached around epoch 95.
    Snapshotting the last epoch would then hand the downstream comparison a diverged
    encoder and confound "longer pre-training" with "unlucky final epoch".

    Returns (per-epoch mean losses, index of the selected epoch).
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    model.train()

    best_loss, best_state, best_epoch = float("inf"), None, 0
    losses = []
    for epoch in range(epochs):
        running = 0.0
        iterator = tqdm(loader, desc=f"SSL {epoch + 1}/{epochs}", leave=False, disable=not progress)
        for images, _ in iterator:
            images = normalize_batch(images.to(device, non_blocking=True))
            reconstruction, original, _ = model(images, mask_ratio=mask_ratio)
            loss = criterion(reconstruction, original)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        losses.append(running / len(loader))
        if losses[-1] < best_loss:
            best_loss, best_epoch = losses[-1], epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
            print(f"    SSL epoch {epoch + 1:>3}/{epochs}: recon MSE {losses[-1]:.4f}")

    model.load_state_dict(best_state)
    if verbose and best_epoch != epochs - 1:
        print(f"    selected epoch {best_epoch + 1} (MSE {best_loss:.4f}); "
              f"last epoch was {losses[-1]:.4f}")
    return losses, best_epoch


@torch.no_grad()
def _evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = normalize_batch(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / seen


def train_classifier(model, train_loader, val_loader, device, epochs=50, lr=5e-4,
                     weight_decay=1e-4, patience=5, min_delta=1e-3, verbose=True,
                     progress=False):
    """Supervised fine-tuning with early stopping on validation loss."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        if not any(p.requires_grad for p in model.encoder.parameters()):
            model.encoder.eval()  # keep the frozen backbone deterministic during linear probing

        total_loss, correct, seen = 0.0, 0, 0
        iterator = tqdm(train_loader, desc=f"FT {epoch + 1}/{epochs}", leave=False, disable=not progress)
        for images, labels in iterator:
            images = normalize_batch(images.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        val_loss, val_acc = _evaluate_loss(model, val_loader, criterion, device)
        history["train_loss"].append(total_loss / len(train_loader))
        history["train_acc"].append(100.0 * correct / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"  epoch {epoch + 1:>2}: train {history['train_loss'][-1]:.4f}/"
                  f"{history['train_acc'][-1]:.2f}%  val {val_loss:.4f}/{val_acc:.2f}%")

        stopper(val_loss, model, epoch)
        if stopper.early_stop:
            if verbose:
                print(f"  early stop at epoch {epoch + 1} (best was {stopper.best_epoch + 1})")
            break

    stopper.restore(model)
    history["best_epoch"] = stopper.best_epoch
    history["epochs_run"] = len(history["train_loss"])
    return history


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    """Final test-set evaluation. Returns a metrics dict plus raw predictions."""
    model.eval()
    y_true, y_pred, y_score = [], [], []
    for images, labels in loader:
        images = normalize_batch(images.to(device, non_blocking=True))
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        y_true.append(labels.numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_score.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_score = np.concatenate(y_score)

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True, zero_division=0
    )
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_auc": float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")),
        "per_class": {c: report[c] for c in class_names},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics, y_true, y_pred, y_score
