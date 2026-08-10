"""The controlled study the original course project was missing.

The submitted project reported a single number (94.8% test accuracy) for an
SSL-pre-trained encoder, with no baseline -- so it could not actually support its
claim that self-supervised pre-training helped. This script runs the controls:

  scratch         randomly initialised encoder, full supervised fine-tuning
  ssl-finetune    FCMAE-pre-trained encoder, full supervised fine-tuning
  scratch-linear  randomly initialised encoder FROZEN, linear head only
  ssl-linear      FCMAE-pre-trained encoder FROZEN, linear head only

`scratch` is the baseline for `ssl-finetune`; `scratch-linear` is the random-features
control that makes `ssl-linear` interpretable (a frozen random ConvNeXt is not a
trivial feature extractor, so the probe number means nothing on its own).

Two further axes:
  * label budget -- fine-tuning on 1%/5%/10%/25%/100% of the 1,250 labelled images,
    because label efficiency is where SSL is supposed to pay off;
  * pre-training length -- the submitted 15 epochs plus a 100-epoch run, to check
    whether any null result is just under-training.

The data split -- and hence the 750-image test set -- is identical everywhere;
only training stochasticity varies with the seed.

Usage:
    python -m src.experiments --seeds 0 1 2
    python -m src.experiments --seeds 0 --fractions 1.0 --ssl-epochs 15   # quick check
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import CachedImages, load_dataset, make_splits, stratified_subset
from .engine import evaluate, make_loader, pretrain_ssl, set_seed, train_classifier
from .model import FCMAE, Classifier, ConvNeXtV2Encoder, count_parameters

CONDITIONS = ("scratch", "scratch-linear", "ssl-finetune", "ssl-linear")
DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.00)
DEFAULT_SSL_EPOCHS = (15, 100)

# Frozen-backbone runs train only a 5-way linear head, so they take a larger step;
# 5e-4 is the value the original project used for full fine-tuning.
LEARNING_RATES = {"scratch": 5e-4, "ssl-finetune": 5e-4, "scratch-linear": 1e-3, "ssl-linear": 1e-3}

USES_SSL = {"ssl-finetune", "ssl-linear"}
IS_FROZEN = {"scratch-linear", "ssl-linear"}


def get_pretrained_encoder(seed, ssl_epochs, ssl_loader, device, args, ckpt_dir: Path):
    """Pre-train the FCMAE encoder for this (seed, ssl_epochs), or load it if already done."""
    path = ckpt_dir / f"fcmae_{args.variant}_e{ssl_epochs}_seed{seed}.pt"
    encoder = ConvNeXtV2Encoder(variant=args.variant)

    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(payload["encoder"])
        print(f"[seed {seed}] loaded cached SSL encoder ({ssl_epochs} epochs)")
        return encoder, payload["ssl_losses"]

    set_seed(seed)
    encoder = ConvNeXtV2Encoder(variant=args.variant)
    ssl_model = FCMAE(encoder).to(device)
    print(f"[seed {seed}] SSL pre-training {ssl_epochs} epochs on "
          f"{len(ssl_loader.dataset)} unlabelled images", flush=True)
    started = time.time()
    losses, best_epoch = pretrain_ssl(ssl_model, ssl_loader, device, epochs=ssl_epochs,
                                      mask_ratio=args.mask_ratio, progress=args.progress)
    print(f"    done in {time.time() - started:.0f}s "
          f"(MSE {losses[0]:.4f} -> {losses[best_epoch]:.4f} @ epoch {best_epoch + 1})")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": ssl_model.encoder.state_dict(), "ssl_losses": losses,
                "ssl_best_epoch": best_epoch, "ssl_epochs": ssl_epochs, "seed": seed}, path)
    return ssl_model.encoder.cpu(), losses


def run_one(condition, fraction, seed, encoder_state, splits, labels, class_names, device, args):
    """Fine-tune and evaluate a single cell of the sweep."""
    set_seed(seed + 1000)  # decouple fine-tuning randomness from pre-training randomness

    encoder = ConvNeXtV2Encoder(variant=args.variant)
    if condition in USES_SSL:
        encoder.load_state_dict(encoder_state)

    model = Classifier(encoder, num_classes=len(class_names), dropout_rate=args.dropout).to(device)
    if condition in IS_FROZEN:
        model.freeze_encoder()

    train_subset = stratified_subset(splits["train"], labels, fraction, seed)
    train_loader = make_loader(train_subset, batch_size=args.batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(splits["val"], batch_size=args.batch_size)
    test_loader = make_loader(splits["test"], batch_size=args.batch_size)

    started = time.time()
    history = train_classifier(model, train_loader, val_loader, device,
                               epochs=args.ft_epochs, lr=LEARNING_RATES[condition],
                               patience=args.patience, verbose=args.verbose,
                               progress=args.progress)
    metrics, *_ = evaluate(model, test_loader, device, class_names)

    return {
        "condition": condition,
        "fraction": fraction,
        "seed": seed,
        "n_train": len(train_subset),
        "trainable_params": count_parameters(model),
        "test_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_auc": metrics["macro_auc"],
        "best_epoch": history["best_epoch"],
        "epochs_run": history["epochs_run"],
        "val_acc_at_best": history["val_acc"][history["best_epoch"]],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "runtime_s": round(time.time() - started, 1),
    }


def build_plan(args):
    """Enumerate every (condition, ssl_epochs, fraction, seed) cell to run.

    ssl_epochs is None for the two conditions that never touch the pre-trained
    encoder, so they are not needlessly repeated once per pre-training length.
    """
    plan = []
    for seed in args.seeds:
        for condition in args.conditions:
            epoch_values = args.ssl_epochs if condition in USES_SSL else [None]
            for ssl_epochs in epoch_values:
                for fraction in args.fractions:
                    plan.append((condition, ssl_epochs, fraction, seed))
    # Group by (seed, ssl_epochs) so each encoder is pre-trained/loaded once.
    plan.sort(key=lambda c: (c[3], c[1] is not None, c[1] or 0, c[0], c[2]))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=None, help="Local dataset dir; downloads from Kaggle if omitted")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--ssl-epochs", type=int, nargs="+", default=list(DEFAULT_SSL_EPOCHS))
    parser.add_argument("--variant", default="atto", help="Encoder size (atto is what the project used)")
    parser.add_argument("--ft-epochs", type=int, default=50)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out", default="results/metrics/sweep.json")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch fine-tuning progress")
    parser.add_argument("--progress", action="store_true", help="Show tqdm bars")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    images, labels, class_names = load_dataset(args.data_root)
    dataset = CachedImages(images, labels)
    splits = make_splits(dataset)
    print(f"classes: {class_names}")
    print("split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["condition"], r["ssl_epochs"], r["fraction"], r["seed"]) for r in records}

    ckpt_dir = Path(args.ckpt_dir)
    ssl_loader = make_loader(splits["ssl"], batch_size=args.batch_size, shuffle=True, seed=0)

    plan = build_plan(args)
    print(f"planned runs: {len(plan)}\n")

    encoder_cache_key, encoder_state = None, None
    started_all = time.time()

    for index, (condition, ssl_epochs, fraction, seed) in enumerate(plan, start=1):
        key = (condition, ssl_epochs, fraction, seed)
        label = (f"[{index}/{len(plan)}] {condition:<14} "
                 f"ssl_ep={str(ssl_epochs):<4} frac={fraction:<5} seed={seed}")
        if key in done:
            print(f"{label}  (cached)")
            continue

        if condition in USES_SSL and encoder_cache_key != (seed, ssl_epochs):
            encoder, ssl_losses = get_pretrained_encoder(seed, ssl_epochs, ssl_loader,
                                                         device, args, ckpt_dir)
            encoder_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            encoder_cache_key = (seed, ssl_epochs)
            (out_path.parent / f"ssl_losses_e{ssl_epochs}_seed{seed}.json").write_text(
                json.dumps(ssl_losses, indent=2))

        print(label, flush=True)
        record = run_one(condition, fraction, seed, encoder_state,
                         splits, labels, class_names, device, args)
        record["ssl_epochs"] = ssl_epochs
        print(f"    -> acc {record['test_accuracy'] * 100:5.2f}%  "
              f"macroF1 {record['macro_f1']:.4f}  AUC {record['macro_auc']:.4f}  "
              f"n_train={record['n_train']:<4} ({record['runtime_s']}s)")

        records.append(record)
        out_path.write_text(json.dumps(records, indent=2))

    print(f"\ndone: {len(records)} records in {(time.time() - started_all) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
