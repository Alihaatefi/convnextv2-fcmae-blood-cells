"""Turn the raw sweep records into the label-efficiency figure and summary tables.

Produces:
    results/figures/label_efficiency.png       (light)
    results/figures/label_efficiency_dark.png  (dark, for GitHub's dark theme)
    results/metrics/summary.md                 markdown table, mean +/- std over seeds
    results/metrics/summary.csv                the same numbers, machine-readable

Usage:
    python -m src.analyze
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .theme import THEMES, style_axes  # noqa: E402

# (condition, ssl_epochs) -> (panel, series slot, display name)
SERIES = {
    ("scratch", None):         (0, 0, "From scratch"),
    ("ssl-finetune", 15):      (0, 1, "SSL 15 ep"),
    ("ssl-finetune", 100):     (0, 2, "SSL 100 ep"),
    ("scratch-linear", None):  (1, 0, "Random features"),
    ("ssl-linear", 15):        (1, 1, "SSL 15 ep"),
    ("ssl-linear", 100):       (1, 2, "SSL 100 ep"),
}
PANEL_TITLES = ["Full fine-tuning", "Linear probe (frozen encoder)"]


def load_records(path: Path):
    records = json.loads(path.read_text())
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["condition"], r.get("ssl_epochs"), r["fraction"])].append(r)
    return records, grouped


def aggregate(grouped):
    """(condition, ssl_epochs, fraction) -> dict of mean/std/n for each metric."""
    out = {}
    for key, runs in grouped.items():
        row = {"n_train": runs[0]["n_train"], "n_seeds": len(runs)}
        for metric in ("test_accuracy", "macro_f1", "macro_auc"):
            values = np.array([r[metric] for r in runs], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out[key] = row
    return out


# Direct labels are applied SELECTIVELY -- the series that carries each panel's
# point. In the left panel all three endpoints sit within ~1.2 accuracy points, so
# labelling all of them would stack them off the top of the axes; the legend plus
# the table view in results/metrics/summary.md carry the rest of the identity.
DIRECT_LABEL = {0: {"From scratch"}, 1: {"Random features", "SSL 100 ep"}}


def _place_labels(ax, endpoints, theme, panel_index):
    wanted = DIRECT_LABEL.get(panel_index, set())
    for x, y, text in endpoints:
        if text in wanted:
            ax.annotate(text, xy=(x, y), xytext=(9, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=9, color=theme["secondary"])


def make_figure(summary, out_path: Path, theme_name: str):
    theme = THEMES[theme_name]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True,
                             facecolor=theme["surface"])

    fractions = sorted({k[2] for k in summary})
    panels = defaultdict(list)
    for (condition, ssl_epochs, fraction), row in summary.items():
        spec = SERIES.get((condition, ssl_epochs))
        if spec:
            panels[spec[0]].append((spec[1], spec[2], fraction, row))

    for panel_index, ax in enumerate(axes):
        ax.set_facecolor(theme["surface"])
        endpoints = []

        by_series = defaultdict(list)
        for slot, name, fraction, row in panels[panel_index]:
            by_series[(slot, name)].append((fraction, row))
        if not by_series:  # a partial sweep may leave a panel empty
            ax.set_visible(False)
            continue

        for (slot, name), points in sorted(by_series.items()):
            points.sort()
            xs = np.array([p[1]["n_train"] for p in points], dtype=float)
            means = np.array([p[1]["test_accuracy_mean"] for p in points]) * 100
            stds = np.array([p[1]["test_accuracy_std"] for p in points]) * 100
            color = theme["series"][slot]

            ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.13, linewidth=0)
            ax.plot(xs, means, color=color, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=theme["surface"], markeredgewidth=2,
                    label=name, zorder=3, solid_capstyle="round")
            endpoints.append((xs[-1], means[-1], name))

        ax.set_xscale("log")
        ax.set_xticks([p[1]["n_train"] for p in sorted(next(iter(by_series.values())))])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Labelled training images (log scale)", fontsize=10, color=theme["secondary"])
        ax.set_title(PANEL_TITLES[panel_index], fontsize=11, color=theme["ink"],
                     pad=10, loc="left")

        style_axes(ax, theme)
        ax.legend(frameon=False, fontsize=9, loc="lower right",
                  labelcolor=theme["secondary"])
        _place_labels(ax, endpoints, theme, panel_index)

    axes[0].set_ylabel("Test accuracy (%)", fontsize=10, color=theme["secondary"])
    fig.subplots_adjust(right=0.87)
    fig.suptitle("Does FCMAE pre-training help? Mean of 3 seeds, shaded band = ±1 SD",
                 fontsize=12, color=theme["ink"], x=0.008, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0, 0.94, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return fractions


def write_tables(summary, metrics_dir: Path):
    rows = []
    for (condition, ssl_epochs, fraction), row in sorted(
        summary.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2])
    ):
        rows.append({
            "condition": condition,
            "ssl_epochs": ssl_epochs if ssl_epochs is not None else "",
            "label_fraction": fraction,
            "n_train": row["n_train"],
            "n_seeds": row["n_seeds"],
            "test_acc_mean": round(row["test_accuracy_mean"] * 100, 2),
            "test_acc_std": round(row["test_accuracy_std"] * 100, 2),
            "macro_f1_mean": round(row["macro_f1_mean"], 4),
            "macro_f1_std": round(row["macro_f1_std"], 4),
            "macro_auc_mean": round(row["macro_auc_mean"], 4),
        })

    csv_path = metrics_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    lines = ["| Condition | SSL epochs | Labels | n | Test acc (%) | Macro-F1 |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        labels = f"{r['label_fraction']:.0%}"
        lines.append(
            f"| {r['condition']} | {r['ssl_epochs'] or '--'} | {labels} | {r['n_train']} | "
            f"{r['test_acc_mean']:.2f} ± {r['test_acc_std']:.2f} | "
            f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} |"
        )
    md_path = metrics_dir / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")
    return rows


def print_headline(summary):
    """The comparison the original project was missing, at full label budget."""
    print("\n=== Headline comparison (100% labels, 1250 images, 3 seeds) ===")
    for key in (("scratch", None, 1.0), ("ssl-finetune", 15, 1.0), ("ssl-finetune", 100, 1.0),
                ("scratch-linear", None, 1.0), ("ssl-linear", 15, 1.0), ("ssl-linear", 100, 1.0)):
        row = summary.get(key)
        if row:
            name = f"{key[0]}" + (f" ({key[1]} ep)" if key[1] else "")
            print(f"  {name:<26} {row['test_accuracy_mean'] * 100:5.2f} "
                  f"± {row['test_accuracy_std'] * 100:.2f} %")

    print("\n=== Low-label regime (1% labels, 13 images) ===")
    for key in (("scratch", None, 0.01), ("ssl-finetune", 15, 0.01), ("ssl-finetune", 100, 0.01)):
        row = summary.get(key)
        if row:
            name = f"{key[0]}" + (f" ({key[1]} ep)" if key[1] else "")
            print(f"  {name:<26} {row['test_accuracy_mean'] * 100:5.2f} "
                  f"± {row['test_accuracy_std'] * 100:.2f} %")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="results/metrics/sweep.json")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--metrics-dir", default="results/metrics")
    args = parser.parse_args()

    records, grouped = load_records(Path(args.sweep))
    summary = aggregate(grouped)
    print(f"loaded {len(records)} runs -> {len(summary)} (condition, ssl_epochs, fraction) cells")

    figures_dir = Path(args.figures_dir)
    make_figure(summary, figures_dir / "label_efficiency.png", "light")
    make_figure(summary, figures_dir / "label_efficiency_dark.png", "dark")
    write_tables(summary, Path(args.metrics_dir))
    print_headline(summary)


if __name__ == "__main__":
    main()
