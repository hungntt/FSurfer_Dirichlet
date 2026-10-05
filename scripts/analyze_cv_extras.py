#!/usr/bin/env python3
"""Additional analyses of the SHAP-Dirichlet layer on the saved CV artifacts.

1. Per-subject export of the primary refinement (validation-selected RIS/ROS
   gate, similarity, percentile, and lambda with soft fusion).
2. Calibration (Brier score, ECE, NLL) of baseline and refined probabilities.
3. Selective prediction: risk-coverage curves and AURC with confidence-based
   abstention, for the baseline and for refine-then-defer.
4. Retrieval quality of SHAP space versus raw feature space: label precision
   of the top-K references and accuracy/AUROC of the Dirichlet estimate.
5. Global SHAP importance and its stability across folds and backbones.

Every quantity uses out-of-fold test subjects; all rule parameters come from
the validation sets.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from analyze_cv_refinement import METRICS, evidence, gate_mask, gate_scores, load_folds, select_and_apply
from posthoc_core import OUTPUT_DIR, THRESHOLD, load_table

EPS = 1e-12


BANK = "val"


def primary_details(f):
    """Apply the primary rule to a fold and return per-subject test details."""
    rule, pred, score = select_and_apply(f, gates=["ris", "ros"], metrics=METRICS, corrections=["soft"], bank=BANK)
    pt = f.a["p_test"]
    if rule.correction == "none":
        gated = np.zeros(len(pt), dtype=bool)
        dir_p = np.full(len(pt), np.nan)
    else:
        sv = gate_scores(f, rule.gate, "val")
        st = gate_scores(f, rule.gate, "test")
        _, gated, _ = gate_mask(sv, st, rule.percentile, rule.gate)
        dir_p = evidence(f, "test", rule.metric, bank=BANK)["dir_p_MDD"].to_numpy()
    ev_p = evidence(f, "test", "pearson", bank=BANK)["dir_p_MDD"].to_numpy()
    return rule, pd.DataFrame({
        "backbone": f.backbone, "fold": f.fold, "seed": f.meta["split_seed"],
        "index": f.a["test_idx"], "y": f.yt, "p_base": pt, "p_ref": score,
        "pred_base": (pt >= THRESHOLD).astype(int), "pred_ref": pred.astype(int),
        "gated": gated, "dir_p_selected": dir_p, "dir_p_pearson": ev_p,
        "rule_gate": rule.gate, "rule_metric": rule.metric, "rule_percentile": rule.percentile, "rule_lambda": rule.lam,
    })


def calibration(y, p, n_bins=10):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    brier = float(np.mean((p - y) ** 2))
    nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return {"brier": brier, "ece": float(ece), "nll": nll}


def risk_coverage(y, p, coverages):
    """Selective accuracy when keeping the most confident subjects (confidence |2p-1|)."""
    conf = np.abs(2 * np.asarray(p) - 1)
    order = np.argsort(-conf, kind="mergesort")
    correct = ((np.asarray(p) >= 0.5).astype(int) == np.asarray(y))[order]
    n = len(y)
    acc = [float(correct[: max(1, int(round(c * n)))].mean()) for c in coverages]
    risks = 1 - np.cumsum(correct) / np.arange(1, n + 1)
    return acc, float(risks.mean())


def retrieval_quality(folds, top_k=5):
    """Label precision of top-K references and Dirichlet classifier quality per space and metric."""
    rows = []
    for space in ("shap", "feature"):
        for metric in METRICS:
            prec, accs, aucs, agree = [], [], [], []
            for f in folds:
                ev = evidence(f, "test", metric, top_k=top_k, space=space)
                d = ev["dir_p_MDD"].to_numpy()
                yt = f.yt
                frac = np.where(yt == 1, ev["n_MDD_topk"].to_numpy(), top_k - ev["n_MDD_topk"].to_numpy()) / top_k
                prec.append(frac)
                accs.append(((d >= 0.5).astype(int) == yt))
                aucs.append(roc_auc_score(yt, d))
                agree.append(((d >= 0.5).astype(int) == (f.a["p_test"] >= THRESHOLD).astype(int)))
            prec, accs, agree = map(np.concatenate, (prec, accs, agree))
            rows.append({"space": space, "metric": metric, "label_precision_at_k": float(prec.mean()),
                         "dirichlet_accuracy": float(accs.mean()), "dirichlet_auroc_mean_fold": float(np.mean(aucs)),
                         "agreement_with_model": float(agree.mean()), "n": len(prec)})
    return pd.DataFrame(rows)


def shap_importance(folds, feature_cols):
    imp, signed_mdd, per_fold = [], [], []
    for f in folds:
        S = f.a["shap_test"]
        imp.append(np.abs(S))
        per_fold.append(np.abs(S).mean(0))
    imp = np.vstack(imp)
    mean_abs = imp.mean(0)
    per_fold = np.vstack(per_fold)
    order = np.argsort(-mean_abs)
    top = order[:15]
    jac = []
    tops = [set(np.argsort(-r)[:15]) for r in per_fold]
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            jac.append(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))
    signed = np.vstack([f.a["shap_test"] for f in folds]).mean(0)
    df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs, "mean_signed_shap": signed})
    return df.sort_values("mean_abs_shap", ascending=False), float(np.mean(jac))


def calibration_ci(subj, n_boot=2000, seed=0):
    """Paired bootstrap of the change in Brier/ECE/NLL, averaged over repetitions."""
    rng = np.random.default_rng(seed)
    rows = []
    for bb in subj.backbone.unique():
        x = subj[subj.backbone == bb]
        seeds = sorted(x.seed.unique())
        point = {k: np.mean([calibration(x[x.seed == s].y, x[x.seed == s].p_ref)[k] - calibration(x[x.seed == s].y, x[x.seed == s].p_base)[k]
                             for s in seeds]) for k in ("brier", "ece", "nll")}
        boots = {k: [] for k in point}
        for _ in range(n_boot):
            parts = []
            for s in seeds:
                xs = x[x.seed == s]
                i = rng.integers(0, len(xs), len(xs))
                cb, cr = calibration(xs.y.values[i], xs.p_base.values[i]), calibration(xs.y.values[i], xs.p_ref.values[i])
                parts.append({k: cr[k] - cb[k] for k in cb})
            for k in boots:
                boots[k].append(np.mean([q[k] for q in parts]))
        for k in point:
            b = np.asarray(boots[k])
            rows.append({"backbone": bb, "metric": k, "delta": float(point[k]), "ci_lo": float(np.percentile(b, 2.5)),
                         "ci_hi": float(np.percentile(b, 97.5)), "p_ge_0": float((b >= 0).mean()), "n_seeds": len(seeds)})
    return pd.DataFrame(rows)


def gated_and_terciles(subj):
    g_rows, t_rows = [], []
    for bb in subj.backbone.unique():
        x = subj[subj.backbone == bb].copy()
        g = x[x.gated]
        u = x[~x.gated]
        g_rows.append({"backbone": bb, "gated_fraction": float(x.gated.mean()), "n_gated": int(len(g)),
                       "gated_acc_base": float((g.pred_base == g.y).mean()), "gated_acc_ref": float((g.pred_ref == g.y).mean()),
                       "ungated_acc": float((u.pred_base == u.y).mean())})
        x["conf"] = np.abs(2 * x.p_base - 1)
        parts = []
        for s in x.seed.unique():
            xs = x[x.seed == s].copy()
            xs["tercile"] = pd.qcut(xs.conf, 3, labels=["low", "mid", "high"])
            parts.append(xs)
        x = pd.concat(parts)
        for t, xt in x.groupby("tercile", observed=True):
            t_rows.append({"backbone": bb, "tercile": t, "acc_base": float((xt.pred_base == xt.y).mean()),
                           "acc_ref": float((xt.pred_ref == xt.y).mean()), "gated_fraction": float(xt.gated.mean()),
                           "changed_fraction": float((xt.pred_ref != xt.pred_base).mean()), "n": len(xt)})
    return pd.DataFrame(g_rows), pd.DataFrame(t_rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=str, default="42,43")
    p.add_argument("--backbones", type=str, default="tabm,resnet")
    p.add_argument("--output-root", type=str, default=str(OUTPUT_DIR / "cv_analysis" / "extras"))
    p.add_argument("--bank", type=str, default="val", choices=["val", "train", "train_val"])
    args = p.parse_args()
    global BANK
    BANK = args.bank
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    _, _, _, feature_cols = load_table()
    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    subj, calib, rc_rows, ret_rows, imp_rows, stab_rows = [], [], [], [], [], []
    for bb in args.backbones.split(","):
        all_folds = []
        for s in seeds:
            folds = load_folds(OUTPUT_DIR / "cv_artifacts" / f"seed{s}", [bb])[bb]
            if len(folds) < 5:
                print(f"skip {bb} seed {s}: {len(folds)} folds")
                continue
            all_folds += folds
            parts = [primary_details(f)[1] for f in folds]
            d = pd.concat(parts, ignore_index=True)
            subj.append(d)
            for kind, col in (("baseline", "p_base"), ("refined", "p_ref")):
                calib.append({"backbone": bb, "seed": s, "model": kind, **calibration(d["y"], d[col])})
                acc, aurc = risk_coverage(d["y"].to_numpy(), d[col].to_numpy(), coverages)
                rc_rows.append({"backbone": bb, "seed": s, "model": kind, "aurc": aurc, **{f"acc@{c:.1f}": a for c, a in zip(coverages, acc)}})
            r = retrieval_quality(folds) if bb in ("tabm", "resnet") or True else None
            r.insert(0, "seed", s)
            r.insert(0, "backbone", bb)
            ret_rows.append(r)
        if all_folds:
            imp, jac = shap_importance(all_folds, feature_cols)
            imp.insert(0, "backbone", bb)
            imp_rows.append(imp)
            stab_rows.append({"backbone": bb, "top15_jaccard_across_folds": jac, "n_folds": len(all_folds)})
    subj = pd.concat(subj, ignore_index=True)
    subj.to_csv(out / "per_subject_primary.csv", index=False)
    pd.DataFrame(calib).to_csv(out / "calibration.csv", index=False)
    pd.DataFrame(rc_rows).to_csv(out / "risk_coverage.csv", index=False)
    pd.concat(ret_rows).to_csv(out / "retrieval_quality.csv", index=False)
    pd.concat(imp_rows).to_csv(out / "shap_importance.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(out / "shap_importance_stability.csv", index=False)

    calibration_ci(subj).to_csv(out / "calibration_ci.csv", index=False)
    g, t = gated_and_terciles(subj)
    g.to_csv(out / "gated_subset.csv", index=False)
    t.to_csv(out / "terciles.csv", index=False)
    print(pd.read_csv(out / "calibration_ci.csv").to_string(index=False))
    print(g.to_string(index=False))
    print(t.to_string(index=False))

    pd.set_option("display.width", 250)
    fmt = lambda v: f"{v:.4f}"
    print(pd.DataFrame(calib).groupby(["backbone", "model"]).mean(numeric_only=True).to_string(float_format=fmt))
    print(pd.DataFrame(rc_rows).groupby(["backbone", "model"]).mean(numeric_only=True).to_string(float_format=fmt))
    print(pd.concat(ret_rows).groupby(["backbone", "space", "metric"]).mean(numeric_only=True).to_string(float_format=fmt))
    for bb in args.backbones.split(","):
        im = pd.concat(imp_rows)
        print(im[im.backbone == bb].head(15).to_string(index=False, float_format=fmt))
    print(pd.DataFrame(stab_rows).to_string(index=False))
    a = pd.concat(imp_rows)
    if {"tabm", "resnet"} <= set(a.backbone):
        from scipy.stats import spearmanr
        t = a[a.backbone == "tabm"].set_index("feature")["mean_abs_shap"]
        r = a[a.backbone == "resnet"].set_index("feature")["mean_abs_shap"].reindex(t.index)
        print("Spearman TabM vs ResNet importance:", spearmanr(t, r))
    g = subj.groupby(["backbone"]).agg(gated=("gated", "mean"))
    print(g)


if __name__ == "__main__":
    main()
