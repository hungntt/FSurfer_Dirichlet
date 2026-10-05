#!/usr/bin/env python3
"""Tables and in-text numbers for the ACM version of the paper.

Writes LaTeX table bodies to <out>/tables/*.tex and every number cited in the
text as a macro in <out>/acm_numbers.tex, so the paper can be rebuilt after any
rerun without hand-copying numbers.

Statistics (pre-specified):
- per-repetition exact McNemar tests (from analyze_cv_refinement.py),
- a sign test over CV repetitions of the pooled accuracy change,
- the corrected resampled t-test of Nadeau and Bengio over all fold-level
  accuracy changes, with test/train ratio 1/4 for 5-fold outer CV.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from posthoc_core import OUTPUT_DIR

PRIMARY = "proposed_valselect_soft"
NAMES = {"tabm": "TabM", "resnet": "ResNet", "logreg": "Logistic regression", "svm": "RBF-SVM", "rf": "Random forest"}
SHORT = {"tabm": "TabM", "resnet": "ResNet", "logreg": "LogReg", "svm": "SVM", "rf": "RF"}
MACRO = {"tabm": "TabM", "resnet": "ResNet", "logreg": "LogReg", "svm": "SVM", "rf": "RF"}


def sgn(v, d=2):
    s = f"{v:+.{d}f}"
    return s.replace("-", "$-$")


def num(v, d=2):
    return f"{v:.{d}f}".replace("-", "$-$")


def load_methods(seeds):
    out = []
    for s in seeds:
        p = OUTPUT_DIR / "cv_analysis" / f"seed{s}" / "all_methods.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["seed"] = s
            out.append(d)
    return pd.concat(out, ignore_index=True)


def fold_deltas(df, bb, method):
    base = df[(df.backbone == bb) & (df.method == "baseline")].set_index("seed")
    m = df[(df.backbone == bb) & (df.method == method)].set_index("seed")
    seeds = sorted(set(base.index) & set(m.index))
    cols = [f"fold{i}_accuracy" for i in range(1, 6)]
    d = np.concatenate([(m.loc[s, cols].astype(float).values - base.loc[s, cols].astype(float).values) for s in seeds])
    rep = np.array([m.loc[s, "pooled_accuracy"] - base.loc[s, "pooled_accuracy"] for s in seeds])
    return d, rep, seeds


def corrected_t(d, ratio=0.25):
    J = len(d)
    var = d.var(ddof=1)
    if var <= 0:
        return np.nan, 1.0
    t = d.mean() / np.sqrt((1.0 / J + ratio) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return float(t), float(p)


def sign_test(rep):
    pos, neg = int((rep > 1e-12).sum()), int((rep < -1e-12).sum())
    p = stats.binomtest(pos, pos + neg, 0.5, alternative="two-sided").pvalue if pos + neg else 1.0
    return pos, neg, float(p)


BANKED = "soft_bank_train_val"


def main_table(df, calib, calib_bank, backbones):
    lines = []
    cols = ["pooled_accuracy", "pooled_balanced_accuracy", "pooled_macro_f1", "pooled_roc_auc"]
    for bb in backbones:
        sub = df[df.backbone == bb]
        mean = lambda meth, col: 100 * sub[sub.method == meth][col].astype(float).mean()
        b = [mean("baseline", c) for c in cols]
        nseed = sub[sub.method == "baseline"].seed.nunique()
        cb = calib[calib.backbone == bb]
        cbb = calib_bank[calib_bank.backbone == bb]
        lines.append(rf"\multirow{{3}}{{*}}{{{NAMES[bb]}}} & \multirow{{3}}{{*}}{{{nseed}}} & Baseline & " + " & ".join(num(v) for v in b)
                     + f" & {cb[cb.model == 'baseline'].brier.mean():.4f} & {num(100 * cb[cb.model == 'baseline'].ece.mean())} & & \\\\")
        for meth, label, cal in ((PRIMARY, "Layer, val.\\ bank", cb), (BANKED, "Layer, train+val.\\ bank", cbb)):
            r = [mean(meth, c) for c in cols]
            d, rep, _ = fold_deltas(df, bb, meth)
            t, p = corrected_t(d)
            pos, neg, _ = sign_test(rep)
            cell = lambda rv, bv: f"{num(rv)} ({sgn(rv - bv)})"
            bold = (lambda x: f"\\textbf{{{x}}}") if meth == BANKED else (lambda x: x)
            lines.append(f" & & {label} & " + " & ".join(cell(rv, bv) for rv, bv in zip(r, b))
                         + f" & {cal[cal.model == 'refined'].brier.mean():.4f} & {num(100 * cal[cal.model == 'refined'].ece.mean())} & {pos}/{len(rep)} & {p:.2f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return "\n".join(lines)


ABL = [
    ("Correction", [("proposed_valselect_hard", "Hard flip"), ("proposed_valselect_all", "Hard or soft (validated)"), (PRIMARY, "Soft fusion (ours)")]),
    ("Gate", [("soft_gate_none", "None"), ("soft_gate_random_mean20", "Random"), ("soft_gate_confidence", "Low confidence"),
              ("soft_gate_mc_dropout", "MC-dropout variance"), ("soft_gate_ris", "RIS only"), ("soft_gate_ros", "ROS only"), (PRIMARY, "RIS/ROS, validated (ours)")]),
    ("Evidence", [("soft_space_feature", "Raw-feature retrieval"), ("soft_bank_train", "SHAP, training bank"),
                  ("soft_bank_train_val", "SHAP, training+validation bank"), (PRIMARY, "SHAP, validation bank (ours)")]),
    ("Top-$K$", [("soft_topk_3", "$K=3$"), (PRIMARY, "$K=5$ (ours)"), ("soft_topk_7", "$K=7$"), ("soft_topk_10", "$K=10$"), ("soft_topk_15", "$K=15$")]),
]


def ablation_table(df, backbones, groups=ABL):
    lines = []
    for g, rows in groups:
        for i, (m, label) in enumerate(rows):
            cells = []
            for bb in backbones:
                sub = df[df.backbone == bb]
                if m not in set(sub.method):
                    cells += ["--", "--"]
                    continue
                d = 100 * (sub[sub.method == m].set_index("seed").pooled_accuracy - sub[sub.method == "baseline"].set_index("seed").pooled_accuracy).dropna()
                fx, br = sub[sub.method == m].fixed.sum(), sub[sub.method == m].broken.sum()
                pos = int((d > 1e-9).sum())
                cells += [sgn(d.mean()), f"{pos}/{len(d)}"]
            lab = rf"\multirow{{{len(rows)}}}{{*}}{{{g}}}" if i == 0 else ""
            lines.append(f"{lab} & {label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return "\n".join(lines)


def calib_table(cci, backbones):
    lines = []
    for bb in backbones:
        c = cci[cci.backbone == bb].set_index("metric")
        cells = []
        for k, scale in (("brier", 1000), ("ece", 100), ("nll", 1000)):
            r = c.loc[k]
            star = r"$^{*}$" if r.ci_hi < 0 else ""
            cells.append(f"{sgn(scale * r.delta, 1)}{star} [{num(scale * r.ci_lo, 1)}, {num(scale * r.ci_hi, 1)}]")
        lines.append(f"{NAMES[bb]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def retrieval_table(ret, backbones):
    lines = []
    r = ret.groupby(["backbone", "space", "metric"]).mean(numeric_only=True).reset_index()
    for space, sname in (("feature", "Raw features"), ("shap", "SHAP")):
        for metric in ("cosine", "pearson", "euclidean"):
            cells = []
            for bb in backbones:
                x = r[(r.backbone == bb) & (r.space == space) & (r.metric == metric)]
                if x.empty:
                    cells += ["--"] * 3
                    continue
                x = x.iloc[0]
                cells += [num(100 * x.label_precision_at_k, 1), num(100 * x.dirichlet_accuracy, 1), num(x.dirichlet_auroc_mean_fold, 3)]
            lines.append(f"{sname} & {metric.capitalize()} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return "\n".join(lines)


ERR = [("confidence", "Low confidence"), ("mc_dropout", "MC-dropout variance"), ("ris", "RIS"), ("ros", "ROS"),
       ("dir_opp_pearson", "SHAP-Dirichlet (Pearson)"), ("dir_opp_euclidean", "SHAP-Dirichlet (Euclidean)"),
       ("conf+dir", "Confidence + SHAP-Dirichlet")]


def error_table(rel_dir, backbones):
    dfs = {bb: pd.read_csv(rel_dir / f"{bb}_error_detection.csv").set_index("score") for bb in backbones if (rel_dir / f"{bb}_error_detection.csv").exists()}
    lines = []
    for key, label in ERR:
        cells = []
        for bb in backbones:
            if bb not in dfs or key not in dfs[bb].index:
                cells.append("--")
                continue
            r = dfs[bb].loc[key]
            cells.append(f"{r.auroc:.3f} ({r.auroc_ci_lo:.2f}--{r.auroc_ci_hi:.2f})")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def repetition_table(df, backbones):
    lines = []
    for bb in backbones:
        for sd in sorted(df[df.backbone == bb].seed.unique()):
            sub = df[(df.backbone == bb) & (df.seed == sd)].set_index("method")
            if PRIMARY not in sub.index or BANKED not in sub.index:
                continue
            b, r, k = sub.loc["baseline"], sub.loc[PRIMARY], sub.loc[BANKED]
            cells = [f"{SHORT[bb]}", f"{sd}", num(100 * b.pooled_accuracy)]
            for x in (r, k):
                cells += [sgn(100 * (x.pooled_accuracy - b.pooled_accuracy)), f"{int(x.fixed)}/{int(x.broken)}", f"{x.mcnemar_p:.2f}"]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return "\n".join(lines)


def per_fold_configs(seeds, backbones):
    """Distribution of the selected hyperparameters over all outer folds and repetitions."""
    lines = []
    for bb in backbones:
        rows = []
        for s in seeds:
            for k in range(1, 6):
                p = OUTPUT_DIR / "cv_artifacts" / f"seed{s}" / bb / f"fold{k}" / "meta.json"
                if p.exists():
                    m = json.loads(p.read_text())
                    rows.append({"id": m["best_candidate_id"], "acc": m["metrics"]["test"]["accuracy"], "params": m["params"]})
        if not rows:
            continue
        accs = np.array([r["acc"] for r in rows])
        ids = pd.Series([r["id"] for r in rows]).value_counts()
        lines.append(f"{NAMES[bb]} & {len(rows)} & {len(ids)} & {num(100 * accs.mean())} $\\pm$ {num(100 * accs.std(ddof=1))} & "
                     f"{num(100 * accs.min())}--{num(100 * accs.max())}" + r" \\")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--classic-seeds", type=str, default="42,43")
    ap.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "6a4e845051244b4623abab2e"))
    args = ap.parse_args()
    deep_seeds = [int(s) for s in args.deep_seeds.split(",")]
    classic_seeds = [int(s) for s in args.classic_seeds.split(",")]
    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df = load_methods(sorted(set(deep_seeds) | set(classic_seeds)))
    df = df[((df.backbone.isin(["tabm", "resnet"])) & df.seed.isin(deep_seeds)) | ((~df.backbone.isin(["tabm", "resnet"])) & df.seed.isin(classic_seeds))]
    deep = [b for b in ("tabm", "resnet") if b in set(df.backbone)]
    classic = [b for b in ("logreg", "svm", "rf") if b in set(df.backbone)]
    allbb = deep + classic
    ex = OUTPUT_DIR / "cv_analysis" / "extras"
    calib = pd.read_csv(ex / "calibration.csv")
    cci = pd.read_csv(OUTPUT_DIR / "cv_analysis" / "extras_bank" / "calibration_ci.csv")
    cci_val = pd.read_csv(ex / "calibration_ci.csv")
    ret = pd.read_csv(ex / "retrieval_quality.csv")
    exb = OUTPUT_DIR / "cv_analysis" / "extras_bank"
    gated = pd.read_csv(exb / "gated_subset.csv").set_index("backbone")
    terc = pd.read_csv(exb / "terciles.csv")
    calib_bank_all = pd.read_csv(exb / "calibration.csv")
    rel_dir = OUTPUT_DIR / "cv_analysis" / ("reliability_seeds" + "-".join(map(str, deep_seeds)))

    T = out / "tables"
    calib_bank = pd.read_csv(OUTPUT_DIR / "cv_analysis" / "extras_bank" / "calibration.csv")
    (T / "acm_main.tex").write_text(main_table(df, calib, calib_bank, allbb))
    (T / "acm_ablation.tex").write_text(ablation_table(df, deep))
    classic_groups = [ABL[0], ("Gate", [r for r in ABL[1][1] if r[0] != "soft_gate_mc_dropout"]), ("Evidence", [ABL[2][1][0], ABL[2][1][-1]])]
    (T / "acm_ablation_classic.tex").write_text(ablation_table(df, classic, classic_groups))
    (T / "acm_calibration.tex").write_text(calib_table(cci, [b for b in allbb if b in set(cci.backbone)]) + "\n\\bottomrule")
    (T / "acm_retrieval.tex").write_text(retrieval_table(ret, deep))
    if rel_dir.exists():
        (T / "acm_error.tex").write_text(error_table(rel_dir, deep) + "\n\\bottomrule")
    (T / "acm_repetitions.tex").write_text(repetition_table(df, allbb))
    (T / "acm_configs.tex").write_text(per_fold_configs(deep_seeds, deep) + "\n" + per_fold_configs(classic_seeds, classic) + "\n\\bottomrule")

    # Macros for the text.
    M = {}
    allrep = []
    for bb in allbb:
        sub = df[df.backbone == bb]
        b = 100 * sub[sub.method == "baseline"].pooled_accuracy.mean()
        r = 100 * sub[sub.method == PRIMARY].pooled_accuracy.mean()
        d, rep, seeds = fold_deltas(df, bb, PRIMARY)
        t, p = corrected_t(d)
        pos, neg, sp = sign_test(rep)
        fixed, broken = sub[sub.method == PRIMARY].fixed.sum(), sub[sub.method == PRIMARY].broken.sum()
        nch = sub[sub.method == PRIMARY].n_changed.astype(float)
        k = MACRO[bb]
        M.update({f"{k}Base": num(b), f"{k}Ref": num(r), f"{k}Delta": sgn(r - b), f"{k}T": num(t), f"{k}P": f"{p:.3f}",
                  f"{k}RepPos": str(pos), f"{k}RepN": str(len(rep)), f"{k}SignP": f"{sp:.3f}", f"{k}Fixed": str(int(fixed)),
                  f"{k}Broken": str(int(broken)), f"{k}MaxChanged": num(100 * nch.max() / 504, 1), f"{k}NSeeds": str(len(seeds)),
                  f"{k}RepMin": sgn(100 * rep.min()), f"{k}RepMax": sgn(100 * rep.max())})
        mp = sub[sub.method == PRIMARY].mcnemar_p
        M[f"{k}McMin"] = f"{mp.min():.3f}"
        if bb in deep:
            allrep.append(rep)
        if bb in gated.index:
            g = gated.loc[bb]
            M.update({f"{k}GatedFrac": num(100 * g.gated_fraction, 1), f"{k}GatedBase": num(100 * g.gated_acc_base, 1),
                      f"{k}GatedRef": num(100 * g.gated_acc_ref, 1), f"{k}UngatedAcc": num(100 * g.ungated_acc, 1)})
        th = terc[(terc.backbone == bb) & (terc.tercile == "high")]
        if not th.empty and "changed_fraction" in th:
            M[f"{k}HighChanged"] = num(100 * th.changed_fraction.iloc[0], 1)
        tt = terc[(terc.backbone == bb) & (terc.tercile == "low")]
        if not tt.empty:
            M.update({f"{k}LowBase": num(100 * tt.acc_base.iloc[0], 1), f"{k}LowRef": num(100 * tt.acc_ref.iloc[0], 1),
                      f"{k}LowDelta": sgn(100 * (tt.acc_ref.iloc[0] - tt.acc_base.iloc[0]), 1)})
        c = cci[cci.backbone == bb].set_index("metric") if bb in set(cci.backbone) else None
        if c is not None:
            M.update({f"{k}DBrier": sgn(1000 * c.loc["brier"].delta, 1), f"{k}DEce": sgn(100 * c.loc["ece"].delta, 1),
                      f"{k}DNll": sgn(1000 * c.loc["nll"].delta, 1)})
            cb = calib[calib.backbone == bb]
            M.update({f"{k}EceBase": num(100 * cb[cb.model == "baseline"].ece.mean(), 1), f"{k}EceRef": num(100 * cb[cb.model == "refined"].ece.mean(), 1)})
    for bb in allbb:
        seeds_bb = deep_seeds if bb in ("tabm", "resnet") else classic_seeds
        ps = [np.load(OUTPUT_DIR / "cv_artifacts" / f"seed{s_}" / bb / f"fold{k_}" / "artifacts.npz")["p_test"]
              for s_ in seeds_bb for k_ in range(1, 6) if (OUTPUT_DIR / "cv_artifacts" / f"seed{s_}" / bb / f"fold{k_}" / "artifacts.npz").exists()]
        if ps:
            c = np.abs(2 * np.concatenate(ps) - 1)
            M[f"{MACRO[bb]}NearHalf"] = num(100 * np.mean(c < 0.1), 0)
    bank_rep_all, bank_rep_deep, bank_conf = [], [], []
    for bb in allbb:
        sub = df[df.backbone == bb]
        k = MACRO[bb]
        b = 100 * sub[sub.method == "baseline"].pooled_accuracy.mean()
        r = 100 * sub[sub.method == BANKED].pooled_accuracy.mean()
        d, rep, seeds = fold_deltas(df, bb, BANKED)
        t, p = corrected_t(d)
        pos, neg, sp = sign_test(rep)
        M.update({f"{k}BRef": num(r), f"{k}BDelta": sgn(r - b), f"{k}BP": f"{p:.2f}", f"{k}BRepPos": str(pos), f"{k}BRepN": str(len(rep)),
                  f"{k}BFixed": str(int(sub[sub.method == BANKED].fixed.sum())), f"{k}BBroken": str(int(sub[sub.method == BANKED].broken.sum())),
                  f"{k}BMaxChanged": num(100 * sub[sub.method == BANKED].n_changed.astype(float).max() / 504, 1)})
        bank_rep_all.append(rep)
        if bb in deep:
            bank_rep_deep.append(rep)
            seeds_sorted = sorted(sub[sub.method == BANKED].seed.unique())
            conf = [s_ for s_ in seeds_sorted if s_ >= 44]
            x = sub[sub.method == BANKED].set_index("seed").pooled_accuracy - sub[sub.method == "baseline"].set_index("seed").pooled_accuracy
            M[f"{k}BConfirm"] = ", ".join(sgn(100 * x.loc[s_]) for s_ in conf)
            M[f"{k}BDev"] = ", ".join(sgn(100 * x.loc[s_]) for s_ in seeds_sorted if s_ < 44)
        cb = calib_bank_all[calib_bank_all.backbone == bb]
        M.update({f"{k}BEceRef": num(100 * cb[cb.model == "refined"].ece.mean(), 1)})
        if bb in set(cci_val.backbone):
            cv = cci_val[cci_val.backbone == bb].set_index("metric")
            M.update({f"{k}VDEce": sgn(100 * cv.loc["ece"].delta, 1), f"{k}VDBrier": sgn(1000 * cv.loc["brier"].delta, 1), f"{k}VDNll": sgn(1000 * cv.loc["nll"].delta, 1)})
    bd = np.concatenate(bank_rep_deep)
    pos, neg, sp = sign_test(bd)
    M.update({"BDeepRepPos": str(pos), "BDeepRepN": str(len(bd)), "BDeepSignP": f"{sp:.3f}"})
    ba = np.concatenate(bank_rep_all)
    pos, neg, sp = sign_test(ba)
    M.update({"BAllRepPos": str(pos), "BAllRepN": str(len(ba)), "BAllSignP": f"{sp:.3f}"})
    bases = {bb: 100 * df[(df.backbone == bb) & (df.method == "baseline")].pooled_accuracy.mean() for bb in allbb}
    lo, hi = min(bases, key=bases.get), max(bases, key=bases.get)
    TXT = {"tabm": "TabM", "resnet": "ResNet", "logreg": "logistic regression", "svm": "RBF-SVM", "rf": "random forest"}
    M.update({"MinBase": num(bases[lo]), "MaxBase": num(bases[hi]), "MinBaseName": TXT[lo], "MaxBaseName": TXT[hi]})
    if allrep:
        allrep = np.concatenate(allrep)
        pos, neg, sp = sign_test(allrep)
        M.update({"DeepRepPos": str(pos), "DeepRepN": str(len(allrep)), "DeepSignP": f"{sp:.3f}"})
    # Gate and correction ablations for the text.
    for bb in deep:
        sub = df[df.backbone == bb]
        base = sub[sub.method == "baseline"].set_index("seed").pooled_accuracy
        for m, key in (("soft_gate_none", "NoGate"), ("soft_gate_confidence", "ConfGate"), ("soft_gate_mc_dropout", "McGate"),
                       ("soft_gate_random_mean20", "RandGate"), ("proposed_valselect_hard", "Hard"), ("soft_space_feature", "Feat"),
                       ("soft_bank_train_val", "BankTV"), ("threshold_tuning", "Thresh")):
            if m in set(sub.method):
                d = 100 * (sub[sub.method == m].set_index("seed").pooled_accuracy - base).dropna()
                M[f"{MACRO[bb]}{key}"] = sgn(d.mean())
    # Reliability numbers.
    if rel_dir.exists():
        for bb in deep:
            e = pd.read_csv(rel_dir / f"{bb}_error_detection.csv").set_index("score")
            dfr = pd.read_csv(rel_dir / f"{bb}_deferral.csv")
            k = MACRO[bb]
            M.update({f"{k}AucConf": f"{e.loc['confidence'].auroc:.2f}", f"{k}AucDir": f"{e.loc['dir_opp_pearson'].auroc:.2f}",
                      f"{k}AucDirLo": f"{e.loc['dir_opp_pearson'].auroc_ci_lo:.2f}", f"{k}AucDirHi": f"{e.loc['dir_opp_pearson'].auroc_ci_hi:.2f}",
                      f"{k}AucComb": f"{e.loc['conf+dir'].auroc:.2f}", f"{k}AucRis": f"{e.loc['ris'].auroc:.2f}", f"{k}AucRos": f"{e.loc['ros'].auroc:.2f}",
                      f"{k}AucCombGain": f"{e.loc['conf+dir'].delta_vs_conf:+.3f}", f"{k}AucCombGainLo": f"{e.loc['conf+dir'].delta_ci_lo:+.3f}",
                      f"{k}AucCombGainHi": f"{e.loc['conf+dir'].delta_ci_hi:+.3f}"})
            for cov, tag in ((0.7, "Seventy"), (0.5, "Half")):
                for sc, st in (("confidence", "Conf"), ("conf+dir", "Comb")):
                    v = dfr[(dfr.score == sc) & (np.isclose(dfr.target_coverage, cov))].auto_accuracy.iloc[0]
                    M[f"{k}Def{tag}{st}"] = num(100 * v)
    ret_m = ret.groupby(["backbone", "space", "metric"]).mean(numeric_only=True)
    for bb in deep:
        k = MACRO[bb]
        best_f = ret_m.loc[(bb, "feature")].dirichlet_accuracy.max()
        M[f"{k}RetShapEuc"] = num(100 * ret_m.loc[(bb, "shap", "euclidean")].dirichlet_accuracy, 1)
        M[f"{k}RetFeatBest"] = num(100 * best_f, 1)
        M[f"{k}PrecShapEuc"] = num(100 * ret_m.loc[(bb, "shap", "euclidean")].label_precision_at_k, 1)
        M[f"{k}PrecFeatBest"] = num(100 * ret_m.loc[(bb, "feature")].label_precision_at_k.max(), 1)
    stab = pd.read_csv(ex / "shap_importance_stability.csv")
    imp = pd.read_csv(ex / "shap_importance.csv")
    if {"tabm", "resnet"} <= set(imp.backbone):
        t = imp[imp.backbone == "tabm"].set_index("feature").mean_abs_shap
        r = imp[imp.backbone == "resnet"].set_index("feature").mean_abs_shap.reindex(t.index)
        M["ImpSpearman"] = f"{stats.spearmanr(t, r).statistic:.2f}"
    lines = [f"\\newcommand{{\\N{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (out / "acm_numbers.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for f in sorted(T.glob("acm_*.tex")):
        print(f"\n%% {f.name}\n{f.read_text()}")


if __name__ == "__main__":
    main()
