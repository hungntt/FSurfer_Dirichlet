#!/usr/bin/env python3
"""Figures for the ACM version of the paper.

All figures are built from the analysis outputs in outputs/cv_analysis and
the saved artifacts, using the categorical palette (slots 1-3) with markers or
hatching as a second encoding for grayscale print.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posthoc_core import OUTPUT_DIR

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
NAMES = {"tabm": "TabM", "resnet": "ResNet", "logreg": "Logistic reg.", "svm": "RBF-SVM", "rf": "Random forest"}


def style():
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 8, "legend.fontsize": 6.8,
                         "pdf.fonttype": 42})


def clean(ax, grid_axis="y"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5, zorder=0)


def methods_table(seeds):
    rows = []
    for s in seeds:
        p = OUTPUT_DIR / "cv_analysis" / f"seed{s}" / "all_methods.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["seed"] = s
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def fig_repetitions(df, backbones, out):
    """Accuracy change per CV repetition (dots) and mean (large marker) for both reference banks."""
    style()
    variants = [("proposed_valselect_soft", "Validation bank", BLUE, "o"), ("soft_bank_train_val", "Training+validation bank", ORANGE, "s")]
    fig, ax = plt.subplots(figsize=(3.35, 0.5 * len(backbones) + 0.6))
    y = np.arange(len(backbones))[::-1]
    for yy, bb in zip(y, backbones):
        sub = df[df.backbone == bb]
        base = sub[sub.method == "baseline"].set_index("seed")["pooled_accuracy"]
        for j, (m, label, color, mk) in enumerate(variants):
            ref = sub[sub.method == m].set_index("seed")["pooled_accuracy"]
            d = 100 * (ref - base.reindex(ref.index)).dropna()
            off = 0.14 if j == 0 else -0.14
            ax.scatter(d.values, np.full(len(d), yy + off), s=11, color=color, marker=mk, alpha=0.55, lw=0, zorder=3)
            ax.scatter([d.mean()], [yy + off], s=34, color=color, marker=mk, edgecolor="white", lw=0.6, zorder=4,
                       label=label if yy == y[0] else None)
    ax.axvline(0, color=TEXT, lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES[b] for b in backbones])
    ax.set_xlabel("Accuracy change vs. fixed baseline (pp)")
    clean(ax, "x")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_terciles(subj, backbones, out):
    """Accuracy by baseline-confidence tercile, before and after refinement."""
    style()
    fig, axes = plt.subplots(1, len(backbones), figsize=(3.35, 1.55), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, bb in zip(axes, backbones):
        x = subj[subj.backbone == bb].copy()
        x["conf"] = np.abs(2 * x.p_base - 1)
        x["t"] = pd.qcut(x.conf, 3, labels=["Low", "Mid", "High"])
        g = pd.DataFrame({"b": (x.pred_base == x.y).groupby(x.t, observed=True).mean(), "r": (x.pred_ref == x.y).groupby(x.t, observed=True).mean()})
        pos = np.arange(3)
        ax.bar(pos - 0.19, 100 * g.b, 0.36, color=BLUE, label="Baseline", zorder=3)
        ax.bar(pos + 0.19, 100 * g.r, 0.36, color=ORANGE, hatch="///", edgecolor="white", lw=0.4, label="Refined", zorder=3)
        for p_, b_, r_ in zip(pos, g.b, g.r):
            ax.text(p_ + 0.19, 100 * r_ + 0.8, f"{100 * (r_ - b_):+.1f}", ha="center", fontsize=6, color=TEXT)
        ax.set_xticks(pos)
        ax.set_xticklabels(g.index)
        ax.set_title(NAMES[bb])
        ax.set_ylim(45, 85)
        clean(ax)
        ax.set_xlabel("Baseline confidence")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def reliability_curve(y, p, n_bins=8):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 10:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)


def fig_calibration(subj, backbones, out):
    style()
    fig, axes = plt.subplots(1, len(backbones), figsize=(3.35, 1.75), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, bb in zip(axes, backbones):
        x = subj[subj.backbone == bb]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":", zorder=2)
        for col, label, color, mk, ls in (("p_base", "Baseline", BLUE, "o", "-"), ("p_ref", "Refined", ORANGE, "s", "--")):
            xs, ys, _ = reliability_curve(x.y.values, x[col].values)
            ax.plot(xs, ys, color=color, marker=mk, ms=3.5, lw=1.4, ls=ls, mec="white", mew=0.5, label=label, zorder=3)
        ax.set_title(NAMES[bb])
        ax.set_xlabel("Predicted $p_{\\mathrm{MDD}}$")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        clean(ax, "both")
    axes[0].set_ylabel("Observed MDD rate")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_deferral(rel_dir, backbones, out):
    style()
    series = [("confidence", "Confidence", BLUE, "o", "-"), ("dir_opp_pearson", "SHAP-Dirichlet", ORANGE, "s", "--"),
              ("conf+dir", "Confidence + SHAP-Dirichlet", AQUA, "^", "-.")]
    fig, axes = plt.subplots(1, len(backbones), figsize=(3.35, 1.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, bb in zip(axes, backbones):
        d = pd.read_csv(rel_dir / f"{bb}_deferral.csv")
        for key, label, color, mk, ls in series:
            s = d[d.score == key].sort_values("target_coverage")
            ax.plot(100 * s.target_coverage, 100 * s.auto_accuracy, color=color, marker=mk, ls=ls, lw=1.3, ms=3.5,
                    mec="white", mew=0.5, label=label, zorder=3)
        base = 100 * d[(d.score == "confidence") & (d.target_coverage == 1.0)].auto_accuracy.iloc[0]
        ax.axhline(base, color=MUTED, lw=0.8, ls=":", zorder=2)
        ax.set_xlim(103, 47)
        ax.set_xticks([100, 80, 60])
        ax.set_title(NAMES[bb])
        ax.set_xlabel("Auto-handled (%)")
        clean(ax, "both")
    axes[0].set_ylabel("Accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.54, 1.08), columnspacing=0.8, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def pretty_feature(f):
    hemi, rest = (f.split(".", 1) + [""])[:2]
    region, kind = (rest.rsplit(".", 1) + [""])[:2] if "." in rest else (rest, "")
    if f.startswith("ICV"):
        return "Intracranial volume"
    kind = {"thickness": "thickness", "surface_area": "area", "volume": "volume"}.get(kind, kind)
    names = {"caudalanteriorcingulate": "caudal ACC", "rostralanteriorcingulate": "rostral ACC", "caudalmiddlefrontal": "caudal MFG",
             "rostralmiddlefrontal": "rostral MFG", "lateral-ventricle": "lateral ventricle", "accumbens-area": "accumbens",
             "inferiorparietal": "inferior parietal", "superiorparietal": "superior parietal", "lateralorbitofrontal": "lateral OFC",
             "medialorbitofrontal": "medial OFC", "parsorbitalis": "pars orbitalis", "parstriangularis": "pars triangularis",
             "parsopercularis": "pars opercularis", "pericalcarine": "pericalcarine", "mean-cortical-thickness": "mean cortex",
             "frontalpole": "frontal pole", "posteriorcingulate": "posterior cingulate", "isthmuscingulate": "isthmus cingulate",
             "superiorfrontal": "superior frontal", "superiortemporal": "superior temporal", "middletemporal": "middle temporal",
             "inferiortemporal": "inferior temporal", "transversetemporal": "transverse temporal", "lateraloccipital": "lateral occipital",
             "parahippocampal": "parahippocampal", "supramarginal": "supramarginal", "bankssts": "banks STS", "temporalpole": "temporal pole",
             "Cerebellum-Cortex": "cerebellar cortex"}
    return f"{hemi.upper()} {names.get(region, region)} {kind}".strip()


def fig_importance(imp, out, top=12):
    style()
    t = imp[imp.backbone == "tabm"].set_index("feature")["mean_abs_shap"]
    r = imp[imp.backbone == "resnet"].set_index("feature")["mean_abs_shap"]
    avg = ((t / t.sum()) + (r / r.sum())).sort_values(ascending=False)
    feats = list(avg.index[:top])[::-1]
    fig, ax = plt.subplots(figsize=(3.35, 0.19 * top + 0.55))
    y = np.arange(len(feats))
    ax.barh(y + 0.19, 1000 * t.reindex(feats), 0.36, color=BLUE, label="TabM", zorder=3)
    ax.barh(y - 0.19, 1000 * r.reindex(feats), 0.36, color=ORANGE, hatch="///", edgecolor="white", lw=0.4, label="ResNet", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_feature(f) for f in feats])
    ax.set_xlabel("Mean $|\\phi|$ on test subjects ($\\times 10^{-3}$)")
    clean(ax, "x")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_embedding(seed, backbone, out):
    """t-SNE of standardized features and of SHAP vectors for the pooled test subjects of one repetition."""
    from sklearn.manifold import TSNE
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from posthoc_core import load_table
    style()
    X, y_all, _, _ = load_table()
    Xa = X.to_numpy(float)
    S, Z, Y = [], [], []
    for k in range(1, 6):
        a = np.load(OUTPUT_DIR / "cv_artifacts" / f"seed{seed}" / backbone / f"fold{k}" / "artifacts.npz")
        tr, te = a["train_idx"], a["test_idx"]
        mu, sd = Xa[tr].mean(0), Xa[tr].std(0) + 1e-12
        Z.append((Xa[te] - mu) / sd)
        S.append(a["shap_test"])
        Y.append(y_all[te])
    S, Z, Y = np.vstack(S), np.vstack(Z), np.concatenate(Y)
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.75))
    scores = {}
    for ax, (M, title) in zip(axes, ((Z, "Raw features"), (S / (S.std() + 1e-12), "SHAP explanations"))):
        emb = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(M)
        acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), M, Y, cv=5).mean()
        scores[title] = acc
        for lab, color, mk, name in ((0, BLUE, "o", "HC"), (1, ORANGE, "^", "MDD")):
            m = Y == lab
            ax.scatter(emb[m, 0], emb[m, 1], s=4, color=color, marker=mk, alpha=0.75, lw=0, label=name)
        ax.set_title(f"{title}\n(5-NN accuracy {100 * acc:.1f}%)", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    axes[0].legend(frameon=False, loc="lower left", markerscale=2, handletextpad=0.1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return scores


def fig_case_study(subj, seed, backbone, out, bank="val"):
    """One corrected test subject: its SHAP profile and the retrieved references."""
    from posthoc_core import load_table, pairwise_distance
    style()
    _, _, _, feature_cols = load_table()
    c = subj[(subj.backbone == backbone) & (subj.seed == seed) & subj.gated & (subj.pred_base != subj.y) & (subj.pred_ref == subj.y)]
    c = c.assign(shift=(c.p_ref - c.p_base).abs()).sort_values("shift", ascending=False)
    row = c.iloc[0]
    a = np.load(OUTPUT_DIR / "cv_artifacts" / f"seed{seed}" / backbone / f"fold{int(row.fold)}" / "artifacts.npz")
    i = int(np.where(a["test_idx"] == int(row["index"]))[0][0])
    q = a["shap_test"][i]
    if bank == "train_val":
        R = np.vstack([a["shap_train"], a["shap_validation"]])
        yr = np.concatenate([a["y"][a["train_idx"]], a["y"][a["val_idx"]]])
    else:
        R, yr = a["shap_validation"], a["y"][a["val_idx"]]
    d = pairwise_distance(q[None, :], R, row.rule_metric)[0]
    top = np.argsort(d, kind="mergesort")[:5]
    tau = np.median(d[top])
    w = np.exp(-d[top] / tau)
    e_mdd, e_hc = w[yr[top] == 1].sum(), w[yr[top] == 0].sum()
    p_dir = (e_mdd + 1) / (e_mdd + e_hc + 2)
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.7), gridspec_kw={"width_ratios": [1.35, 1]})
    k = np.argsort(-np.abs(q))[:7][::-1]
    axes[0].barh(np.arange(len(k)), 1000 * q[k], color=[ORANGE if v > 0 else BLUE for v in q[k]], zorder=3)
    axes[0].set_yticks(np.arange(len(k)))
    axes[0].set_yticklabels([pretty_feature(feature_cols[j]) for j in k], fontsize=6)
    axes[0].axvline(0, color=TEXT, lw=0.7)
    axes[0].set_xlabel("$\\phi_q$ ($\\times 10^{-3}$)")
    axes[0].set_title("Query explanation", fontsize=7)
    clean(axes[0], "x")
    cols = [ORANGE if l == 1 else BLUE for l in yr[top]]
    axes[1].bar(np.arange(5), w, color=cols, zorder=3, hatch=["///" if l == 1 else "" for l in yr[top]], edgecolor="white", lw=0.4)
    axes[1].set_xticks(np.arange(5))
    axes[1].set_xticklabels(["MDD" if l == 1 else "HC" for l in yr[top]], fontsize=6, rotation=55)
    axes[1].set_ylabel("Weight $w_{qr}$")
    axes[1].set_title("Top-5 references", fontsize=7)
    axes[1].set_ylim(0, 1.05)
    clean(axes[1])
    true = "MDD" if int(row.y) == 1 else "HC"
    fig.text(0.5, -0.04, f"True: {true}.  $p_q$={row.p_base:.2f}, $p^{{\\mathrm{{Dir}}}}_1$={p_dir:.2f}, "
             f"$\\lambda$={row.rule_lambda:.2f}, $p^{{\\mathrm{{soft}}}}_q$={row.p_ref:.2f}", ha="center", fontsize=6.5, color=TEXT)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {"subject_index": int(row["index"]), "fold": int(row.fold), "true": true, "p_base": float(row.p_base), "p_dir": float(p_dir),
            "p_dir_saved": float(row.dir_p_selected), "lambda": float(row.rule_lambda), "p_ref": float(row.p_ref), "metric": row.rule_metric,
            "gate": row.rule_gate, "ref_labels": yr[top].tolist(), "weights": w.round(3).tolist()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "6a4e845051244b4623abab2e" / "img" / "acm"))
    p.add_argument("--extras-dir", type=str, default=str(OUTPUT_DIR / "cv_analysis" / "extras"))
    p.add_argument("--bank-extras-dir", type=str, default=str(OUTPUT_DIR / "cv_analysis" / "extras_bank"))
    p.add_argument("--reliability-dir", type=str, default=None)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = methods_table(seeds)
    bbs = [b for b in ["tabm", "resnet", "logreg", "svm", "rf"] if b in set(df.backbone)]
    fig_repetitions(df, bbs, out / "repetitions.pdf")
    ex = Path(args.extras_dir)
    subj = pd.read_csv(Path(args.bank_extras_dir) / "per_subject_primary.csv")
    deep = [b for b in ["tabm", "resnet"] if b in set(subj.backbone)]
    subj = subj[subj.backbone.isin(deep)]
    fig_terciles(subj, deep, out / "terciles.pdf")
    fig_calibration(subj, deep, out / "calibration.pdf")
    fig_importance(pd.read_csv(ex / "shap_importance.csv"), out / "importance.pdf")
    rel = Path(args.reliability_dir) if args.reliability_dir else OUTPUT_DIR / "cv_analysis" / ("reliability_seeds" + "-".join(map(str, seeds)))
    if rel.exists():
        fig_deferral(rel, deep, out / "deferral.pdf")
    cs = fig_case_study(subj, seeds[0], "tabm", out / "case_study.pdf", bank="train_val")
    (out / "case_study.json").write_text(json.dumps(cs, indent=2))
    print("case study", cs)
    sc = fig_embedding(seeds[0], "tabm", out / "embedding.pdf")
    (out / "embedding_knn.json").write_text(json.dumps(sc, indent=2))
    print("figures written to", out, sc)


if __name__ == "__main__":
    main()
