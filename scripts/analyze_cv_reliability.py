#!/usr/bin/env python3
"""Error detection and expert-review deferral from the post-hoc signals.

For every backbone, outer fold, and split seed, each score is computed for the
validation and test subjects of the fold. Scores are mapped to their percentile
within the validation distribution of the same fold, so out-of-fold test scores
can be pooled across folds without using test information.

Scores (larger = more likely to be a baseline error):
  confidence      1 - |2p - 1|
  mc_dropout      std of p_MDD over Monte Carlo dropout passes
  ris, ros        local explanation instability (label-preserving max)
  dir_opp_<m>     Dirichlet support for the class opposite to the baseline label,
                  from SHAP-space Top-K retrieval with metric m
  conf+dir        mean of the confidence and dir_opp_pearson percentiles
  lr_conf_dir     logistic regression on (confidence, dir_opp_pearson), fit on validation
  lr_all          logistic regression on all signals above, fit on validation

Outputs per backbone: pooled AUROC/AUPRC with bootstrap CIs, paired bootstrap
differences against confidence, and deferral curves (accuracy of the
automatically handled subjects at a target coverage).
"""
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from analyze_cv_refinement import evidence, finite_rank_scores, gate_scores, load_folds
from posthoc_core import OUTPUT_DIR, THRESHOLD

BASE_SCORES = ["confidence", "mc_dropout", "ris", "ros", "dir_opp_pearson", "dir_opp_cosine", "dir_opp_euclidean"]
DERIVED = ["conf+dir", "lr_conf_dir", "lr_all"]
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def raw_score(f, name, split):
    key = "validation" if split == "val" else "test"
    if name.startswith("dir_opp_"):
        ev = evidence(f, split, name.replace("dir_opp_", ""))
        p = f.a[f"p_{key}"]
        d = ev["dir_p_MDD"].to_numpy()
        return np.where(p >= THRESHOLD, 1.0 - d, d)
    return finite_rank_scores(gate_scores(f, name, split))


def to_val_percentile(sv, st):
    srt = np.sort(sv)
    return (np.searchsorted(srt, sv, side="right") / len(sv), np.searchsorted(srt, st, side="right") / len(sv))


def fold_scores(f) -> tuple[dict, dict]:
    """Validation-percentile scores for validation and test subjects of one fold."""
    val, test = {}, {}
    has_mc = f.meta.get("mc_available", True)
    for n in BASE_SCORES:
        if n == "mc_dropout" and not has_mc:
            continue
        val[n], test[n] = to_val_percentile(raw_score(f, n, "val"), raw_score(f, n, "test"))
    val["conf+dir"] = 0.5 * (val["confidence"] + val["dir_opp_pearson"])
    test["conf+dir"] = 0.5 * (test["confidence"] + test["dir_opp_pearson"])
    err_v = ((f.a["p_validation"] >= THRESHOLD).astype(int) != f.yv).astype(int)
    for name, feats in (("lr_conf_dir", ["confidence", "dir_opp_pearson"]),
                        ("lr_all", ["confidence", "mc_dropout", "ris", "ros", "dir_opp_pearson"] if has_mc else ["confidence", "ris", "ros", "dir_opp_pearson"])):
        Xv = np.column_stack([val[c] for c in feats])
        Xt = np.column_stack([test[c] for c in feats])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xv, err_v)
        sv, st = lr.predict_proba(Xv)[:, 1], lr.predict_proba(Xt)[:, 1]
        val[name], test[name] = to_val_percentile(sv, st)
    return val, test


def pooled(folds_by_seed):
    """Concatenate test scores, errors, and fold ids over seeds and folds."""
    S, E, C, G = {}, [], [], []
    for seed, folds in folds_by_seed.items():
        for f in folds:
            _, test = fold_scores(f)
            for k, v in test.items():
                S.setdefault(k, []).append(v)
            pred = (f.a["p_test"] >= THRESHOLD).astype(int)
            E.append((pred != f.yt).astype(int))
            C.append((pred == f.yt).astype(int))
            G.append(np.full(len(f.yt), f"{seed}_{f.fold}"))
    return {k: np.concatenate(v) for k, v in S.items()}, np.concatenate(E), np.concatenate(C), np.concatenate(G)


def bootstrap_auc(err, s, ref=None, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(err)
    a, d = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if err[i].min() == err[i].max():
            continue
        a.append(roc_auc_score(err[i], s[i]))
        if ref is not None:
            d.append(a[-1] - roc_auc_score(err[i], ref[i]))
    out = {"auroc_ci_lo": np.percentile(a, 2.5), "auroc_ci_hi": np.percentile(a, 97.5)}
    if ref is not None:
        d = np.asarray(d)
        out.update({"delta_vs_conf": float(np.mean(d)), "delta_ci_lo": np.percentile(d, 2.5), "delta_ci_hi": np.percentile(d, 97.5),
                    "p_delta_le_0": float((d <= 0).mean())})
    return out


def deferral(folds_by_seed, names):
    rows = []
    for name in names:
        for cov in COVERAGES:
            kept_c, kept_n, total = 0, 0, 0
            for seed, folds in folds_by_seed.items():
                for f in folds:
                    val, test = fold_scores(f)
                    tau = np.percentile(val[name], 100 * cov) if cov < 1 else np.inf
                    keep = test[name] <= tau if cov < 1 else np.ones(len(f.yt), dtype=bool)
                    pred = (f.a["p_test"] >= THRESHOLD).astype(int)
                    kept_c += int((pred[keep] == f.yt[keep]).sum())
                    kept_n += int(keep.sum())
                    total += len(f.yt)
            rows.append({"score": name, "target_coverage": cov, "coverage": kept_n / total, "auto_accuracy": kept_c / max(kept_n, 1),
                         "n_auto": kept_n, "n_total": total})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--backbones", type=str, default="tabm,resnet")
    p.add_argument("--output-root", type=str, default=str(OUTPUT_DIR / "cv_analysis"))
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.output_root) / ("reliability_seeds" + "-".join(map(str, seeds)))
    out_dir.mkdir(parents=True, exist_ok=True)
    for bb in args.backbones.split(","):
        fbs = {s: load_folds(OUTPUT_DIR / "cv_artifacts" / f"seed{s}", [bb])[bb] for s in seeds}
        fbs = {s: v for s, v in fbs.items() if v}
        S, err, _, groups = pooled(fbs)
        rows = []
        for name in [k for k in BASE_SCORES + DERIVED if k in S]:
            r = {"backbone": bb, "score": name, "auroc": roc_auc_score(err, S[name]), "auprc": average_precision_score(err, S[name]),
                 "base_error_rate": err.mean(), "n": len(err), "n_folds": len(np.unique(groups))}
            r.update(bootstrap_auc(err, S[name], None if name == "confidence" else S["confidence"]))
            per = [roc_auc_score(err[groups == g], S[name][groups == g]) for g in np.unique(groups) if err[groups == g].min() != err[groups == g].max()]
            r["fold_auroc_mean"], r["fold_auroc_std"] = float(np.mean(per)), float(np.std(per, ddof=1))
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{bb}_error_detection.csv", index=False)
        dc = deferral(fbs, [k for k in ["confidence", "mc_dropout", "ris", "ros", "dir_opp_pearson", "conf+dir", "lr_conf_dir", "lr_all"] if k in S])
        dc.insert(0, "backbone", bb)
        dc.to_csv(out_dir / f"{bb}_deferral.csv", index=False)
        print(f"== {bb} (seeds {list(fbs)}; n={len(err)})")
        print(df[[c for c in ["score", "auroc", "auroc_ci_lo", "auroc_ci_hi", "fold_auroc_mean", "fold_auroc_std", "delta_vs_conf", "delta_ci_lo", "delta_ci_hi", "p_delta_le_0"] if c in df.columns]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(dc.pivot(index="target_coverage", columns="score", values="auto_accuracy").round(4).to_string())


if __name__ == "__main__":
    main()
