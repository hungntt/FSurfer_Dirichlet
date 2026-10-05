#!/usr/bin/env python3
"""LaTeX tables and summary numbers for the paper, generated from the analysis CSVs.

Every number in the results section is produced here, so the paper can be
regenerated after any rerun. Metrics of a repetition are computed on the 504
pooled out-of-fold test predictions; the tables report the mean over the
available CV repetitions (split seeds).
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from posthoc_core import OUTPUT_DIR

BB = [("tabm", "TabM"), ("resnet", "ResNet")]
MAIN_METRICS = [("accuracy", "Acc."), ("balanced_accuracy", "BAcc"), ("f1", "F1"), ("macro_f1", "MF1"),
                ("precision", "Prec."), ("average_precision", "AP"), ("roc_auc", "AUC")]
PRIMARY = "proposed_valselect_soft"


def load_methods(seeds):
    out = []
    for s in seeds:
        p = OUTPUT_DIR / "cv_analysis" / f"seed{s}" / "all_methods.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["seed"] = s
            out.append(d)
    return pd.concat(out, ignore_index=True)


def mean_over_seeds(df, bb, method, col):
    v = df[(df.backbone == bb) & (df.method == method)][col].astype(float)
    return float(v.mean()), len(v)


def fmt(v, digits=2):
    return f"{v:.{digits}f}"


def signed(v, digits=2):
    return f"{v:+.{digits}f}"


def main_table(df) -> str:
    lines = []
    for bb, name in BB:
        base = [100 * mean_over_seeds(df, bb, "baseline", f"pooled_{m}")[0] for m, _ in MAIN_METRICS]
        ref = [100 * mean_over_seeds(df, bb, PRIMARY, f"pooled_{m}")[0] for m, _ in MAIN_METRICS]
        lines.append(r"\rowcolor{baselineblue}")
        lines.append(f"{name} baseline & " + " & ".join(fmt(v) for v in base) + r" \\")
        lines.append(r"\rowcolor{refinedgreen}")
        lines.append(f"{name} refined & " + " & ".join(fmt(v) for v in ref) + r" \\")
        lines.append(rf"$\Delta$ {name} & " + " & ".join(signed(r - b) for r, b in zip(ref, base)) + r" \\")
        lines.append(r"\hline")
    return "\n".join(lines)


ABLATION_ROWS = [
    ("Correction", [("proposed_valselect_hard", "Hard flip"), ("proposed_valselect_all", "Hard or soft"), (PRIMARY, "Soft fusion (ours)")]),
    ("Gate", [("soft_gate_none", "None"), ("soft_gate_random_mean20", "Random"), ("soft_gate_confidence", "Confidence"),
              ("soft_gate_mc_dropout", "MC dropout"), (PRIMARY, "RIS/ROS (ours)")]),
    ("Evidence", [("soft_space_feature", "Feature space"), (PRIMARY, "SHAP space (ours)")]),
]


def counts(df, bb, method):
    sub = df[(df.backbone == bb) & (df.method == method)]
    fx, br = float(sub["fixed"].sum()), float(sub["broken"].sum())
    if abs(fx - round(fx)) < 1e-9 and abs(br - round(br)) < 1e-9:
        return f"{int(round(fx))}/{int(round(br))}"
    return f"{fx:.1f}/{br:.1f}"


def ablation_table(df) -> str:
    lines = []
    for group, rows in ABLATION_ROWS:
        first = True
        for method, label in rows:
            cells = []
            for bb, _ in BB:
                d, _ = mean_over_seeds(df, bb, method, "pooled_accuracy")
                b, _ = mean_over_seeds(df, bb, "baseline", "pooled_accuracy")
                cells += [signed(100 * (d - b)), counts(df, bb, method)]
            g = rf"\multirow{{{len(rows)}}}{{*}}{{{group}}}" if first else ""
            first = False
            lines.append(f"{g} & {label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    return "\n".join(lines)


def variants_table(df) -> str:
    lines = []
    for gate in ("ris", "ros"):
        for metric in ("pearson", "cosine", "euclidean"):
            for corr in ("soft", "hard"):
                m = f"fixed_{gate}_{metric}_{corr}"
                cells = []
                for bb, _ in BB:
                    d, _ = mean_over_seeds(df, bb, m, "pooled_accuracy")
                    b, _ = mean_over_seeds(df, bb, "baseline", "pooled_accuracy")
                    imp = df[(df.backbone == bb) & (df.method == m)]["folds_improved"].sum()
                    wor = df[(df.backbone == bb) & (df.method == m)]["folds_worse"].sum()
                    cells += [signed(100 * (d - b)), f"{int(imp)}/{int(wor)}"]
                lines.append(f"{gate.upper()} & {metric.capitalize()} & {corr} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def sensitivity_table(df) -> str:
    rows = [("soft_topk_3", "$K=3$"), (PRIMARY, "$K=5$ (default)"), ("soft_topk_7", "$K=7$"), ("soft_topk_10", "$K=10$"),
            ("soft_topk_15", "$K=15$"), ("soft_bank_train", "Training bank"), ("soft_bank_train_val", "Training + validation bank"),
            ("soft_gate_ris", "RIS gate only"), ("soft_gate_ros", "ROS gate only"),
            ("threshold_tuning", "Decision-threshold tuning")]
    lines = []
    for m, label in rows:
        cells = []
        for bb, _ in BB:
            d, _ = mean_over_seeds(df, bb, m, "pooled_accuracy")
            b, _ = mean_over_seeds(df, bb, "baseline", "pooled_accuracy")
            cells.append(signed(100 * (d - b)))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def sci(v: float) -> str:
    m, e = f"{v:.0e}".split("e")
    e = int(e)
    return f"$10^{{{e}}}$" if m == "1" else f"${m}{{\\times}}10^{{{e}}}$"


def per_fold_table(seed) -> str:
    lines = []
    for bb, name in BB:
        for k in range(1, 6):
            meta = json.loads((OUTPUT_DIR / "cv_artifacts" / f"seed{seed}" / bb / f"fold{k}" / "meta.json").read_text())
            p = meta["params"]
            arch = f"[{', '.join(map(str, p['layer_sizes']))}]" if bb == "tabm" else f"{p['num_blocks']}$\\times$[{', '.join(map(str, p['layer_sizes']))}]"
            t = meta["metrics"]["test"]
            lines.append(f"{name} & {k} & {meta['best_candidate_id']} & {arch} & {p['dropout']} & {sci(p['lr'])} & {sci(p['weight_decay'])} & "
                         f"{100 * t['accuracy']:.2f} & {100 * t['f1']:.2f} & {100 * t['roc_auc']:.2f}" + r" \\")
        lines.append(r"\hline")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42,43")
    p.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR / "paper_tables"))
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_methods(seeds)
    (out / "main_table.tex").write_text(main_table(df))
    (out / "ablation_table.tex").write_text(ablation_table(df))
    (out / "variants_table.tex").write_text(variants_table(df))
    (out / "sensitivity_table.tex").write_text(sensitivity_table(df))
    for s in seeds:
        (out / f"per_fold_seed{s}.tex").write_text(per_fold_table(s))
    # Key numbers for the text.
    cols = ["seed", "backbone", "method", "pooled_accuracy", "mean_accuracy", "std_accuracy", "fixed", "broken", "n_changed", "mcnemar_p",
            "delta_acc_ci_lo", "delta_acc_ci_hi", "folds_improved", "folds_worse"]
    keep = df[df.method.isin(["baseline", PRIMARY, "proposed_valselect_hard", "gate_none_valselect", "gate_ris_valselect", "gate_ros_valselect",
                              "threshold_tuning", "bank_train_val_valselect"])][cols]
    keep.to_csv(out / "key_numbers.csv", index=False)
    print(keep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for f in sorted(out.glob("*.tex")):
        print(f"\n%% {f.name}\n{f.read_text()}")


if __name__ == "__main__":
    main()


ERROR_SCORES = [("confidence", "Low confidence"), ("mc_dropout", "MC-dropout variance"), ("ris", "RIS"), ("ros", "ROS"),
                ("dir_opp_pearson", "SHAP-Dirichlet, Pearson"), ("dir_opp_cosine", "SHAP-Dirichlet, cosine"),
                ("dir_opp_euclidean", "SHAP-Dirichlet, Euclidean"), ("conf+dir", "Confidence + SHAP-Dirichlet"),
                ("lr_conf_dir", "Logistic combination")]


def error_table(rel_dir: Path) -> str:
    dfs = {bb: pd.read_csv(rel_dir / f"{bb}_error_detection.csv").set_index("score") for bb, _ in BB}
    lines = []
    for key, label in ERROR_SCORES:
        cells = []
        for bb, _ in BB:
            r = dfs[bb].loc[key]
            cells.append(f"{r['auroc']:.3f} ({r['auroc_ci_lo']:.3f}--{r['auroc_ci_hi']:.3f})")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def repetition_table(df, seeds) -> str:
    lines = []
    for bb, name in BB:
        for s in seeds:
            sub = df[(df.backbone == bb) & (df.seed == s)].set_index("method")
            b, r = sub.loc["baseline"], sub.loc[PRIMARY]
            lines.append(f"{name} & {s} & {100 * b['pooled_accuracy']:.2f} & {100 * r['pooled_accuracy']:.2f} & "
                         f"{signed(100 * (r['pooled_accuracy'] - b['pooled_accuracy']))} & {int(r['fixed'])}/{int(r['broken'])} & "
                         f"{int(r['folds_improved'])}/{int(r['folds_worse'])} & {r['mcnemar_p']:.3f}" + r" \\")
        lines.append(r"\hline")
    return "\n".join(lines)
