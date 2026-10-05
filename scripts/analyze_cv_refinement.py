#!/usr/bin/env python3
"""Validation-selected SHAP-Dirichlet refinement across all outer folds.

Reads the artifacts written by run_cv_posthoc_artifacts.py and, for every
backbone and outer fold:

1. builds SHAP-space Top-K Dirichlet evidence with the validation set as the
   reference bank (a validation query never retrieves itself),
2. builds instability gates (RIS, ROS, and comparison gates),
3. selects every rule parameter on the validation set only,
4. applies the selected rule once to the outer test fold.

Out-of-fold test predictions are pooled over the five folds, so every subject
is evaluated exactly once. The script reports pooled metrics, per-fold
mean +- std, exact McNemar tests, paired bootstrap intervals, error-detection
AUROC, and deferral (human review) curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from posthoc_core import OUTPUT_DIR, THRESHOLD, compute_metrics, dirichlet_evidence, load_table, pairwise_distance

METRICS = ["cosine", "euclidean", "pearson"]
PERCENTILES = [50, 60, 70, 80, 90]
LAMBDAS = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]
OPPOSE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
MIN_OPPOSE_COUNTS = [0, 3]
MIN_FLAGS_HARD = 5
REPORT_METRICS = ["accuracy", "balanced_accuracy", "f1", "macro_f1", "precision", "average_precision", "roc_auc",
                  "recall_sensitivity", "specificity", "mcc"]


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------


@dataclass
class Fold:
    backbone: str
    fold: int
    meta: dict
    a: dict
    X: np.ndarray
    cache: dict = field(default_factory=dict)

    @property
    def yv(self):
        return self.a["y"][self.a["val_idx"]]

    @property
    def yt(self):
        return self.a["y"][self.a["test_idx"]]

    @property
    def yr(self):
        return self.a["y"][self.a["train_idx"]]


def load_folds(root: Path, backbones: list[str]) -> dict[str, list[Fold]]:
    X, _, _, _ = load_table()
    Xa = X.to_numpy(dtype=float)
    out = {}
    for bb in backbones:
        folds = []
        for k in range(1, 6):
            d = root / bb / f"fold{k}"
            if not (d / "artifacts.npz").exists():
                continue
            a = dict(np.load(d / "artifacts.npz"))
            meta = json.loads((d / "meta.json").read_text())
            folds.append(Fold(bb, k, meta, a, Xa))
        out[bb] = folds
    return out


# ---------------------------------------------------------------------------
# Scores: instability, comparison gates, and Dirichlet evidence.
# ---------------------------------------------------------------------------


def instability(f: Fold, split: str, kind: str, setting: int = 0, agg: str = "label_preserving") -> np.ndarray:
    """Local RIS/ROS score; larger means less stable.

    label_preserving: max over perturbations that keep the predicted label
    (the definition in the paper). A subject whose label flips under every
    perturbation receives +inf.
    """
    key = "validation" if split == "val" else "test"
    vals = f.a[f"pert{setting}_{key}_{kind}"]
    if agg == "max_all":
        return vals.max(axis=1)
    if agg == "mean_all":
        return vals.mean(axis=1)
    p0 = f.a[f"p_{key}"]
    keep = (f.a[f"pert{setting}_{key}_p"] >= THRESHOLD) == (p0[:, None] >= THRESHOLD)
    masked = np.where(keep, vals, -np.inf)
    out = masked.max(axis=1)
    out[~keep.any(axis=1)] = np.inf
    return out


def gate_scores(f: Fold, gate: str, split: str, setting: int = 0, agg: str = "label_preserving") -> np.ndarray:
    key = "validation" if split == "val" else "test"
    p = f.a[f"p_{key}"]
    if gate == "ris":
        return instability(f, split, "ris", setting, agg)
    if gate == "ros":
        return instability(f, split, "ros", setting, agg)
    if gate == "confidence":
        return 1.0 - np.abs(2.0 * p - 1.0)
    if gate == "mc_dropout":
        return f.a[f"mc_{key}"].std(axis=0)
    if gate == "label_flip_rate":
        pp = f.a[f"pert{setting}_{key}_p"]
        return ((pp >= THRESHOLD) != (p[:, None] >= THRESHOLD)).mean(axis=1) + 1e-9 * (1.0 - np.abs(2 * p - 1))
    if gate == "none":
        return np.ones_like(p)
    raise ValueError(gate)


def evidence(f: Fold, split: str, metric: str, top_k: int = 5, space: str = "shap", bank: str = "val",
             beta: float = 1.0) -> pd.DataFrame:
    """Dirichlet evidence for validation or test queries."""
    ck = ("ev", split, metric, top_k, space, bank, beta)
    if ck in f.cache:
        return f.cache[ck]
    a = f.a
    if space == "shap":
        Q = a["shap_validation"] if split == "val" else a["shap_test"]
        banks = {"val": a["shap_validation"], "train": a["shap_train"]}
    else:  # standardized raw features, statistics from the training split
        mu = f.X[a["train_idx"]].mean(0)
        sd = f.X[a["train_idx"]].std(0) + 1e-12
        Z = (f.X - mu) / sd
        Q = Z[a["val_idx"]] if split == "val" else Z[a["test_idx"]]
        banks = {"val": Z[a["val_idx"]], "train": Z[a["train_idx"]]}
    if bank == "val":
        R, ry = banks["val"], f.yv
        excl = np.eye(len(Q), len(R), dtype=bool) if split == "val" else None
    elif bank == "train":
        R, ry = banks["train"], f.yr
        excl = None
    elif bank == "train_val":
        R = np.vstack([banks["train"], banks["val"]])
        ry = np.concatenate([f.yr, f.yv])
        excl = None
        if split == "val":
            excl = np.zeros((len(Q), len(R)), dtype=bool)
            excl[:, len(banks["train"]):] = np.eye(len(Q), dtype=bool)
    else:
        raise ValueError(bank)
    D = pairwise_distance(Q, R, metric)
    ev = dirichlet_evidence(D, ry, top_k=top_k, beta=beta, exclude_mask=excl)
    f.cache[ck] = ev
    return ev


# ---------------------------------------------------------------------------
# Rules.
# ---------------------------------------------------------------------------


def apply_soft(p, dir_p, gated, lam):
    out = p.copy()
    out[gated] = lam * p[gated] + (1.0 - lam) * dir_p[gated]
    return out


def apply_hard(p, dir_p, gated, n_mdd_topk, top_k, th, mc):
    pred = (p >= THRESHOLD).astype(int)
    opp_w = np.where(pred == 1, 1.0 - dir_p, dir_p)
    opp_n = np.where(pred == 1, top_k - n_mdd_topk, n_mdd_topk)
    flag = gated & (opp_w > 0.5) & (opp_w >= th) & (opp_n >= mc)
    new_pred = pred.copy()
    new_pred[flag] = 1 - new_pred[flag]
    score = p.copy()
    score[flag] = new_pred[flag].astype(float)
    return new_pred, score, flag


@dataclass
class Rule:
    gate: str
    metric: str
    correction: str  # soft | hard | none
    percentile: float = np.nan
    lam: float = np.nan
    th: float = np.nan
    mc: int = 0
    top_k: int = 5
    space: str = "shap"
    bank: str = "val"
    setting: int = 0
    agg: str = "label_preserving"
    val_acc: float = np.nan
    val_changes: int = 0

    def describe(self) -> dict:
        return {k: getattr(self, k) for k in ("gate", "metric", "correction", "percentile", "lam", "th", "mc", "top_k",
                                              "space", "bank", "setting", "agg", "val_acc", "val_changes")}


def gate_mask(scores_v, scores_t, percentile, gate):
    if gate == "none":
        return np.ones_like(scores_v, dtype=bool), np.ones_like(scores_t, dtype=bool), -np.inf
    finite = scores_v[np.isfinite(scores_v)]
    tau = np.percentile(finite, percentile) if finite.size else np.inf
    return scores_v >= tau, scores_t >= tau, tau


def candidate_rules(f: Fold, gates, metrics, corrections, top_k=5, space="shap", bank="val", setting=0,
                    agg="label_preserving", percentiles=PERCENTILES, rng=None):
    """Yield (rule, val_pred, val_score, test_pred, test_score) for all rule parameters."""
    pv, pt = f.a["p_validation"], f.a["p_test"]
    for gate in gates:
        if gate == "random":
            sv, st = rng.random(len(pv)), rng.random(len(pt))
        else:
            sv = gate_scores(f, gate, "val", setting, agg)
            st = gate_scores(f, gate, "test", setting, agg)
        pcts = [np.nan] if gate == "none" else percentiles
        for metric in metrics:
            ev_v = evidence(f, "val", metric, top_k, space, bank)
            ev_t = evidence(f, "test", metric, top_k, space, bank)
            for pct in pcts:
                gv, gt, _ = gate_mask(sv, st, pct, gate)
                if "soft" in corrections:
                    for lam in LAMBDAS:
                        if lam >= 1.0:
                            continue
                        sv_ = apply_soft(pv, ev_v["dir_p_MDD"].to_numpy(), gv, lam)
                        st_ = apply_soft(pt, ev_t["dir_p_MDD"].to_numpy(), gt, lam)
                        yield (Rule(gate, metric, "soft", pct, lam, top_k=top_k, space=space, bank=bank, setting=setting, agg=agg),
                               (sv_ >= THRESHOLD).astype(int), sv_, (st_ >= THRESHOLD).astype(int), st_)
                if "hard" in corrections:
                    for th, mc in product(OPPOSE_THRESHOLDS, MIN_OPPOSE_COUNTS):
                        yv_, sv_, flv = apply_hard(pv, ev_v["dir_p_MDD"].to_numpy(), gv, ev_v["n_MDD_topk"].to_numpy(), top_k, th, mc)
                        if flv.sum() < MIN_FLAGS_HARD:
                            continue
                        yt_, st_, _ = apply_hard(pt, ev_t["dir_p_MDD"].to_numpy(), gt, ev_t["n_MDD_topk"].to_numpy(), top_k, th, mc)
                        yield (Rule(gate, metric, "hard", pct, th=th, mc=mc, top_k=top_k, space=space, bank=bank, setting=setting, agg=agg),
                               yv_, sv_, yt_, st_)


def select_and_apply(f: Fold, **kw):
    """Select the best rule on validation and return its test predictions.

    Objective: validation accuracy. Ties are broken toward the most
    conservative rule: fewer changed validation predictions, then a higher
    gate percentile, then a larger lambda (closer to the baseline). If no rule
    improves validation accuracy, the baseline is kept.
    """
    pv, pt = f.a["p_validation"], f.a["p_test"]
    base_v = (pv >= THRESHOLD).astype(int)
    base_acc = float((base_v == f.yv).mean())
    best = None
    best_key = None
    for rule, yv_, sv_, yt_, st_ in candidate_rules(f, **kw):
        acc = float((yv_ == f.yv).mean())
        changes = int((yv_ != base_v).sum())
        key = (acc, -changes, np.nan_to_num(rule.percentile, nan=100.0), np.nan_to_num(rule.lam, nan=1.0), np.nan_to_num(rule.th, nan=1.0))
        if best_key is None or key > best_key:
            rule.val_acc, rule.val_changes = acc, changes
            best, best_key = (rule, yt_, st_), key
    if best is None or best[0].val_acc <= base_acc + 1e-12:
        r = Rule(gate="none", metric="none", correction="none", val_acc=base_acc)
        return r, (pt >= THRESHOLD).astype(int), pt.copy()
    return best


# ---------------------------------------------------------------------------
# Statistics.
# ---------------------------------------------------------------------------


def mcnemar_exact(y, pred_a, pred_b) -> dict:
    from scipy.stats import binomtest
    ca, cb = pred_a == y, pred_b == y
    b = int((ca & ~cb).sum())  # broken by b
    c = int((~ca & cb).sum())  # fixed by b
    p = binomtest(c, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return {"fixed": c, "broken": b, "mcnemar_p": float(p)}


def bootstrap_delta(y, pred_a, pred_b, n_boot=5000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    d = (pred_b == y).astype(float) - (pred_a == y).astype(float)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = d[idx].mean(axis=1)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def method_report(name: str, folds: list[Fold], results: list[tuple]) -> dict:
    """results[i] = (rule, test_pred, test_score) for fold i."""
    y = np.concatenate([f.yt for f in folds])
    base_pred = np.concatenate([(f.a["p_test"] >= THRESHOLD).astype(int) for f in folds])
    pred = np.concatenate([r[1] for r in results])
    score = np.concatenate([r[2] for r in results])
    pooled = compute_metrics(y, pred, score)
    per_fold = [compute_metrics(f.yt, r[1], r[2]) for f, r in zip(folds, results)]
    row = {"method": name}
    for m in REPORT_METRICS:
        row[f"pooled_{m}"] = pooled[m]
        vals = np.array([pf[m] for pf in per_fold])
        row[f"mean_{m}"] = float(vals.mean())
        row[f"std_{m}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    row.update(mcnemar_exact(y, base_pred, pred))
    lo, hi = bootstrap_delta(y, base_pred, pred)
    row["delta_acc_ci_lo"], row["delta_acc_ci_hi"] = lo, hi
    row["n_changed"] = int((pred != base_pred).sum())
    row["folds_improved"] = int(sum(pf["accuracy"] > compute_metrics(f.yt, (f.a["p_test"] >= THRESHOLD).astype(int), f.a["p_test"])["accuracy"] + 1e-12 for f, pf in zip(folds, per_fold)))
    row["folds_worse"] = int(sum(pf["accuracy"] < compute_metrics(f.yt, (f.a["p_test"] >= THRESHOLD).astype(int), f.a["p_test"])["accuracy"] - 1e-12 for f, pf in zip(folds, per_fold)))
    row["selected_rules"] = json.dumps([r[0].describe() for r in results], default=float)
    for i, pf in enumerate(per_fold, 1):
        row[f"fold{i}_accuracy"] = pf["accuracy"]
    return row


def baseline_results(folds):
    return [(Rule("none", "none", "none"), (f.a["p_test"] >= THRESHOLD).astype(int), f.a["p_test"].copy()) for f in folds]


def threshold_tuning(folds):
    out = []
    for f in folds:
        grid = np.round(np.arange(0.30, 0.701, 0.01), 2)
        accs = [((f.a["p_validation"] >= t).astype(int) == f.yv).mean() for t in grid]
        best = max(range(len(grid)), key=lambda i: (accs[i], -abs(grid[i] - 0.5)))
        t = float(grid[best])
        r = Rule("none", "none", "threshold", val_acc=float(accs[best]))
        r.th = t
        out.append((r, (f.a["p_test"] >= t).astype(int), f.a["p_test"].copy()))
    return out


# ---------------------------------------------------------------------------
# Error detection and deferral.
# ---------------------------------------------------------------------------


def finite_rank_scores(s):
    s = np.asarray(s, dtype=float).copy()
    fin = s[np.isfinite(s)]
    big = (fin.max() if fin.size else 1.0)
    s[np.isposinf(s)] = big * 1.05 + 1.0
    return s


def error_detection(folds, setting=0, agg="label_preserving", metric="pearson") -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score, average_precision_score
    rows = []
    y = np.concatenate([f.yt for f in folds])
    pred = np.concatenate([(f.a["p_test"] >= THRESHOLD).astype(int) for f in folds])
    err = (pred != y).astype(int)
    scores = {}
    for g in ("ris", "ros", "confidence", "mc_dropout", "label_flip_rate"):
        # Rank within each fold (validation-referenced percentile) so folds are comparable.
        parts = []
        for f in folds:
            sv = finite_rank_scores(gate_scores(f, g, "val", setting, agg))
            st = finite_rank_scores(gate_scores(f, g, "test", setting, agg))
            parts.append(np.searchsorted(np.sort(sv), st, side="right") / len(sv))
        scores[g] = np.concatenate(parts)
    opp = []
    for f in folds:
        ev = evidence(f, "test", metric)
        pt = f.a["p_test"]
        opp.append(np.where(pt >= THRESHOLD, 1.0 - ev["dir_p_MDD"].to_numpy(), ev["dir_p_MDD"].to_numpy()))
    scores["dirichlet_opposition"] = np.concatenate(opp)
    for name, s in scores.items():
        rows.append({"score": name, "auroc_error": roc_auc_score(err, s), "auprc_error": average_precision_score(err, s),
                     "base_error_rate": err.mean(), "n": len(err)})
    return pd.DataFrame(rows)


def deferral_curve(folds, score_name, setting=0, agg="label_preserving", coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """Accuracy of the automatically handled subjects when the most uncertain ones are deferred.

    The deferral threshold is the validation percentile of the score in each
    fold, so no test information is used.
    """
    rows = []
    for cov in coverages:
        kept_correct, kept_n = 0, 0
        for f in folds:
            sv = finite_rank_scores(gate_scores(f, score_name, "val", setting, agg))
            st = finite_rank_scores(gate_scores(f, score_name, "test", setting, agg))
            tau = np.percentile(sv, 100 * cov) if cov < 1 else np.inf
            keep = st < tau if cov < 1 else np.ones_like(st, dtype=bool)
            pred = (f.a["p_test"] >= THRESHOLD).astype(int)
            kept_correct += int((pred[keep] == f.yt[keep]).sum())
            kept_n += int(keep.sum())
        total = sum(len(f.yt) for f in folds)
        rows.append({"score": score_name, "target_coverage": cov, "realized_coverage": kept_n / total,
                     "auto_accuracy": kept_correct / max(kept_n, 1), "n_auto": kept_n})
    return rows


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def run_backbone(bb: str, folds: list[Fold], out_dir: Path, full: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    has_mc = all(f.meta.get("mc_available", True) for f in folds)
    gate_list = ["none", "confidence", "mc_dropout", "label_flip_rate", "ris", "ros"] if has_mc else ["none", "confidence", "label_flip_rate", "ris", "ros"]
    soft_gate_list = ["none", "confidence", "mc_dropout", "ris", "ros"] if has_mc else ["none", "confidence", "ris", "ros"]

    def add(name, results):
        rows.append(method_report(name, folds, results))

    add("baseline", baseline_results(folds))
    add("threshold_tuning", threshold_tuning(folds))

    # Proposed: all design choices selected jointly on validation.
    add("proposed_valselect_all", [select_and_apply(f, gates=["ris", "ros"], metrics=METRICS, corrections=["soft", "hard"]) for f in folds])
    add("proposed_valselect_soft", [select_and_apply(f, gates=["ris", "ros"], metrics=METRICS, corrections=["soft"]) for f in folds])
    add("proposed_valselect_hard", [select_and_apply(f, gates=["ris", "ros"], metrics=METRICS, corrections=["hard"]) for f in folds])

    # Controlled ablations of the proposed soft-fusion design: one factor changes at a time.
    for gate in soft_gate_list:
        add(f"soft_gate_{gate}", [select_and_apply(f, gates=[gate], metrics=METRICS, corrections=["soft"]) for f in folds])
    rand_soft = []
    for r in range(20):
        rr = np.random.default_rng(2000 + r)
        rand_soft.append(method_report("tmp", folds, [select_and_apply(f, gates=["random"], metrics=METRICS, corrections=["soft"], rng=rr) for f in folds]))
    rs_df = pd.DataFrame(rand_soft)
    rs_row = {"method": "soft_gate_random_mean20"}
    for c in rs_df.columns:
        if c.startswith(("pooled_", "mean_", "std_", "fixed", "broken", "n_changed", "folds_")):
            rs_row[c] = float(rs_df[c].mean())
    rows.append(rs_row)
    add("soft_space_feature", [select_and_apply(f, gates=["ris", "ros"], metrics=METRICS, corrections=["soft"], space="feature") for f in folds])
    for bank in ["train", "train_val"]:
        add(f"soft_bank_{bank}", [select_and_apply(f, gates=["ris", "ros"], metrics=METRICS, corrections=["soft"], bank=bank) for f in folds])
    for k in [3, 7, 10, 15]:
        add(f"soft_topk_{k}", [select_and_apply(f, gates=["ris", "ros"], metrics=METRICS, corrections=["soft"], top_k=k) for f in folds])

    if not full:
        df = pd.DataFrame(rows)
        df.insert(0, "backbone", bb)
        df.to_csv(out_dir / f"{bb}_methods.csv", index=False)
        return df

    # Fixed design variants (percentile and lambda/thresholds still selected on validation).
    for gate, metric, corr in product(["ris", "ros"], METRICS, ["soft", "hard"]):
        add(f"fixed_{gate}_{metric}_{corr}", [select_and_apply(f, gates=[gate], metrics=[metric], corrections=[corr]) for f in folds])

    # Gate ablation (metric and correction selected on validation).
    for gate in gate_list:
        add(f"gate_{gate}_valselect", [select_and_apply(f, gates=[gate], metrics=METRICS, corrections=["soft", "hard"]) for f in folds])
    rand_runs = []
    for r in range(20):
        rr = np.random.default_rng(1000 + r)
        rand_runs.append(method_report("tmp", folds, [select_and_apply(f, gates=["random"], metrics=METRICS, corrections=["soft", "hard"], rng=rr) for f in folds]))
    rr_df = pd.DataFrame(rand_runs)
    rand_row = {"method": "gate_random_valselect_mean20"}
    for c in rr_df.columns:
        if c.startswith(("pooled_", "mean_", "std_", "fixed", "broken", "n_changed", "folds_")):
            rand_row[c] = float(rr_df[c].mean())
    rand_row["pooled_accuracy_std_over_draws"] = float(rr_df["pooled_accuracy"].std(ddof=1))
    rows.append(rand_row)

    # Evidence-space ablation.
    for gate in ["ris", "ros"]:
        add(f"space_feature_{gate}_valselect", [select_and_apply(f, gates=[gate], metrics=METRICS, corrections=["soft", "hard"], space="feature") for f in folds])
    add("space_feature_risros_valselect", [select_and_apply(f, gates=["ris", "ros"], metrics=METRICS, corrections=["soft", "hard"], space="feature") for f in folds])

    # Reference bank ablation.
    for bank in ["train", "train_val"]:
        add(f"bank_{bank}_valselect", [select_and_apply(f, gates=["ris", "ros"], metrics=METRICS, corrections=["soft", "hard"], bank=bank) for f in folds])

    # Top-K ablation.
    for k in [3, 7, 10, 15]:
        add(f"topk_{k}_valselect", [select_and_apply(f, gates=["ris", "ros"], metrics=METRICS, corrections=["soft", "hard"], top_k=k) for f in folds])

    # Perturbation and aggregation ablation.
    set_ids = [d["id"] for d in folds[0].meta["perturbation_settings"]]
    for s in set_ids:
        for agg in ["label_preserving", "max_all"]:
            if s == 0 and agg == "label_preserving":
                continue
            add(f"pert{s}_{agg}_valselect", [select_and_apply(f, gates=["ris", "ros"], metrics=METRICS, corrections=["soft", "hard"], setting=s, agg=agg) for f in folds])

    df = pd.DataFrame(rows)
    df.insert(0, "backbone", bb)
    df.to_csv(out_dir / f"{bb}_methods.csv", index=False)

    ed = error_detection(folds)
    ed.insert(0, "backbone", bb)
    ed.to_csv(out_dir / f"{bb}_error_detection.csv", index=False)

    dc = []
    for s in ("ris", "ros", "confidence", "mc_dropout", "label_flip_rate"):
        dc.extend(deferral_curve(folds, s))
    dc = pd.DataFrame(dc)
    dc.insert(0, "backbone", bb)
    dc.to_csv(out_dir / f"{bb}_deferral.csv", index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts-root", type=str, default=str(OUTPUT_DIR / "cv_artifacts" / "seed42"))
    p.add_argument("--output-root", type=str, default=str(OUTPUT_DIR / "cv_analysis" / "seed42"))
    p.add_argument("--backbones", type=str, default="tabm,resnet")
    p.add_argument("--light", action="store_true", help="Only the primary method and the controlled ablations.")
    args = p.parse_args()
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = load_folds(Path(args.artifacts_root), args.backbones.split(","))
    allrows = []
    for bb, fl in folds.items():
        if not fl:
            continue
        print(f"{bb}: {len(fl)} folds", flush=True)
        df = run_backbone(bb, fl, out_dir, full=not args.light)
        allrows.append(df)
        cols = ["method", "pooled_accuracy", "mean_accuracy", "std_accuracy", "pooled_balanced_accuracy", "pooled_macro_f1",
                "pooled_roc_auc", "fixed", "broken", "mcnemar_p", "folds_improved", "folds_worse"]
        print(df[[c for c in cols if c in df.columns]].to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    if allrows:
        new = pd.concat(allrows)
        path = out_dir / "all_methods.csv"
        if path.exists():
            old = pd.read_csv(path)
            old = old[~old.backbone.isin(new.backbone.unique())]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(path, index=False)


if __name__ == "__main__":
    main()
