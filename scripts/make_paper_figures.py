#!/usr/bin/env python3
"""Paper figures from analyze_cv_refinement.py outputs.

Figure: change in pooled out-of-fold accuracy (percentage points) relative to
the fixed baseline, for refinement gated by different uncertainty signals.
"""
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posthoc_core import OUTPUT_DIR

# Categorical slots 1-2 of the reference palette (validated, CVD-safe pair);
# hatching gives a second encoding for grayscale print.
COLORS = {"tabm": "#2a78d6", "resnet": "#eb6834"}
HATCH = {"tabm": "", "resnet": "///"}
LABELS = {"tabm": "TabM", "resnet": "ResNet"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"


def gate_ablation(df: pd.DataFrame, rows: list[tuple[str, str]], out: Path, backbones: list[str]) -> None:
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": TEXT})
    fig, ax = plt.subplots(figsize=(4.7, 0.36 * len(rows) + 0.7))
    h = 0.36
    y = np.arange(len(rows))[::-1]
    for i, bb in enumerate(backbones):
        sub = df[df["backbone"] == bb].set_index("method")
        base = sub.loc["baseline", "pooled_accuracy"]
        vals = np.array([100 * (sub.loc[m, "pooled_accuracy"] - base) for m, _ in rows])
        off = (0.5 - i) * h
        ax.barh(y + off, vals, height=h - 0.04, color=COLORS[bb], hatch=HATCH[bb], edgecolor="white", linewidth=0.6,
                label=LABELS[bb], zorder=3)
        for yy, v in zip(y + off, vals):
            ax.text(v + (0.06 if v >= 0 else -0.06), yy, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=6.5, color=TEXT)
    ax.axvline(0, color=TEXT, linewidth=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in rows])
    ax.set_xlabel("Change in pooled accuracy vs. fixed baseline (pp)")
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    lo = min(ax.get_xlim()[0], -0.5) - 0.4
    hi = max(ax.get_xlim()[1], 0.5) + 0.4
    ax.set_xlim(lo, hi)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


DEFERRAL_SERIES = [
    ("confidence", "Confidence", "#2a78d6", "o", "-"),
    ("dir_opp_pearson", "SHAP-Dirichlet evidence", "#eb6834", "s", "--"),
    ("conf+dir", "Confidence + SHAP-Dirichlet", "#1baf7a", "^", "-."),
]


def deferral_figure(rel_dir: Path, out: Path) -> None:
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.75), sharey=True)
    for ax, (bb, name) in zip(axes, [("tabm", "TabM"), ("resnet", "ResNet")]):
        d = pd.read_csv(rel_dir / f"{bb}_deferral.csv")
        for key, label, color, marker, ls in DEFERRAL_SERIES:
            s = d[d["score"] == key].sort_values("target_coverage")
            ax.plot(100 * s["target_coverage"], 100 * s["auto_accuracy"], color=color, marker=marker, ls=ls, lw=1.5, ms=4,
                    mec="white", mew=0.6, label=label, zorder=3)
        base = 100 * d[(d["score"] == "confidence") & (d["target_coverage"] == 1.0)]["auto_accuracy"].iloc[0]
        ax.axhline(base, color=MUTED, lw=0.8, ls=":", zorder=2)
        ax.set_title(name, fontsize=8, color=TEXT)
        ax.set_xlim(102, 48)
        ax.set_xticks([100, 90, 80, 70, 60, 50])
        ax.grid(color=GRID, lw=0.5, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("Auto-handled subjects (%)")
    axes[0].set_ylabel("Accuracy on handled\nsubjects (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.53, 1.07))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", type=str, default=str(OUTPUT_DIR / "cv_analysis" / "seed42"))
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "6a4e845051244b4623abab2e" / "img"))
    p.add_argument("--suffix", type=str, default="")
    p.add_argument("--reliability-dir", type=str, default=str(OUTPUT_DIR / "cv_analysis" / "reliability_seeds42-43"))
    args = p.parse_args()
    if Path(args.reliability_dir).exists():
        deferral_figure(Path(args.reliability_dir), Path(args.out_dir) / f"deferral{args.suffix}.pdf")
    df = pd.read_csv(Path(args.analysis_dir) / "all_methods.csv")
    rows = [
        ("gate_none_valselect", "No gate"),
        ("gate_random_valselect_mean20", "Random gate"),
        ("gate_confidence_valselect", "Confidence gate"),
        ("gate_mc_dropout_valselect", "MC-dropout gate"),
        ("gate_ros_valselect", "ROS gate"),
        ("gate_ris_valselect", "RIS gate"),
    ]
    gate_ablation(df, rows, Path(args.out_dir) / f"gate_ablation{args.suffix}.pdf", ["tabm", "resnet"])


if __name__ == "__main__":
    main()
