#!/usr/bin/env python3
"""Compute all post-hoc refinement inputs for one backbone and one outer fold.

The script refits the GridSearch-selected model on the final train/validation
split, then saves everything the refinement analysis needs:

- p_MDD for train/validation/test and Monte Carlo dropout passes,
- fixed-p_MDD KernelSHAP vectors for train/validation/test subjects,
- for each perturbation setting, perturbed p_MDD, perturbed SHAP vectors, and
  the per-perturbation RIS/ROS objectives of every validation/test subject.

The refinement rules themselves are selected and evaluated in
analyze_cv_refinement.py, which only reads these artifacts.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

from posthoc_core import (
    OUTPUT_DIR,
    SEED,
    THRESHOLD,
    EPS,
    BatchedKernelSHAP,
    make_predictor,
    compute_metrics,
    final_split,
    load_table,
    logit_prob_pair,
    patch_environment,
    perturb,
    ris_objective,
    ros_objective,
    train_model,
)

PERTURBATION_SETTINGS = [
    ("relative_uniform", 0.02),
    ("relative_uniform", 0.05),
    ("std_gaussian", 0.10),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=["tabm", "resnet", "logreg", "svm", "rf"], required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--split-seed", type=int, default=SEED)
    p.add_argument("--gridsearch-root", type=str, default=str(OUTPUT_DIR / "cv_gridsearch"))
    p.add_argument("--output-root", type=str, default=str(OUTPUT_DIR / "cv_artifacts"))
    p.add_argument("--n-perturbations", type=int, default=10)
    p.add_argument("--n-background", type=int, default=20)
    p.add_argument("--shap-nsamples", type=int, default=800)
    p.add_argument("--mc-passes", type=int, default=30)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--perturbation-settings", type=str, default=None,
                   help="Comma-separated setting ids. Default: all settings for split seed 42, only setting 0 otherwise.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    torch.set_num_threads(args.torch_threads)
    patch_environment()
    (OUTPUT_DIR / "_tmp_ckpt").mkdir(parents=True, exist_ok=True)

    gs_dir = Path(args.gridsearch_root) / f"seed{args.split_seed}" / args.backbone / f"fold{args.fold}"
    best = json.loads((gs_dir / "best.json").read_text())
    params = best["best_params"]
    out_dir = Path(args.output_root) / f"seed{args.split_seed}" / args.backbone / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X, y, subjects, feature_cols = load_table()
    tr, va, te = final_split(X, y, args.fold, args.split_seed)
    train_seed = args.split_seed + 10000 + args.fold
    model = train_model(args.backbone, params, X.iloc[tr], y[tr], X.iloc[va], y[va], seed=train_seed)
    fp = make_predictor(args.backbone, model, feature_cols)

    Xa = X.to_numpy(dtype=float)
    idx = {"train": tr, "validation": va, "test": te}
    p = {k: fp.p_mdd(Xa[v]) for k, v in idx.items()}
    metrics = {k: compute_metrics(y[v], (p[k] >= THRESHOLD).astype(int), p[k]) for k, v in idx.items()}
    mc = {k: fp.p_mdd_mc_dropout(Xa[idx[k]], n_passes=args.mc_passes, seed=train_seed) for k in ("validation", "test")}

    explainer = BatchedKernelSHAP(fp.p_mdd, Xa[tr], n_background=args.n_background, nsamples=args.shap_nsamples, seed=0,
                                  f_tensor=fp.p_mdd_tensor)
    shap_check = explainer.check(Xa[te[:2]], atol=1e-4)
    print(f"[{args.backbone} fold{args.fold}] batched KernelSHAP matches shap (max diff {shap_check:.1e})", flush=True)
    shap_vals = {k: explainer.explain(Xa[v]) for k, v in idx.items()}
    print(f"[{args.backbone} fold{args.fold}] SHAP banks done, {(time.time() - t0) / 60:.1f} min", flush=True)

    arrays = {
        "train_idx": tr, "val_idx": va, "test_idx": te,
        "y": y,
        **{f"p_{k}": v for k, v in p.items()},
        **{f"mc_{k}": v for k, v in mc.items()},
        **{f"shap_{k}": v for k, v in shap_vals.items()},
        "shap_expected_value": np.asarray([explainer.expected_value]),
    }

    train_std = Xa[tr].std(axis=0, ddof=0)
    train_std = np.where(np.isfinite(train_std) & (train_std > EPS), train_std, 1.0)
    if args.perturbation_settings is None:
        setting_ids = list(range(len(PERTURBATION_SETTINGS))) if args.split_seed == SEED else [0]
    else:
        setting_ids = [int(v) for v in args.perturbation_settings.split(",")]
    for s_id, (mode, scale) in enumerate(PERTURBATION_SETTINGS):
        if s_id not in setting_ids:
            continue
        for split in ("validation", "test"):
            rng = np.random.default_rng(args.split_seed * 1000 + args.fold * 100 + s_id * 10 + (0 if split == "validation" else 1))
            n = len(idx[split])
            M = args.n_perturbations
            ris = np.zeros((n, M))
            ros = np.zeros((n, M))
            p_pert = np.zeros((n, M))
            shap_pert = np.zeros((n, M, Xa.shape[1]), dtype=np.float32)
            for i, row in enumerate(idx[split]):
                x = Xa[row]
                xs = np.stack([perturb(x, rng, mode, scale, train_std) for _ in range(M)])
                proba_x = fp.predict_proba(x.reshape(1, -1))
                proba_xs = fp.predict_proba(xs)
                e_x = shap_vals[split][i]
                e_xs = explainer.explain(xs)
                ris[i] = ris_objective(x, xs, e_x, e_xs)
                ros[i] = ros_objective(logit_prob_pair(proba_x), logit_prob_pair(proba_xs), e_x, e_xs)
                p_pert[i] = proba_xs[:, 1]
                shap_pert[i] = e_xs.astype(np.float32)
            arrays[f"pert{s_id}_{split}_ris"] = ris
            arrays[f"pert{s_id}_{split}_ros"] = ros
            arrays[f"pert{s_id}_{split}_p"] = p_pert
            arrays[f"pert{s_id}_{split}_shap"] = shap_pert
            print(f"[{args.backbone} fold{args.fold}] perturbation {mode}/{scale} {split} done, {(time.time() - t0) / 60:.1f} min", flush=True)

    np.savez_compressed(out_dir / "artifacts.npz", **arrays)
    meta = {
        "backbone": args.backbone,
        "fold": args.fold,
        "split_seed": args.split_seed,
        "train_seed": train_seed,
        "params": params,
        "best_candidate_id": best["best_candidate_id"],
        "gridsearch_final_test_metrics": best["final_refit_metrics"]["test"],
        "metrics": {k: {m: float(v) for m, v in d.items()} for k, d in metrics.items()},
        "perturbation_settings": [{"id": i, "mode": m, "scale": s} for i, (m, s) in enumerate(PERTURBATION_SETTINGS) if i in setting_ids],
        "n_perturbations": args.n_perturbations,
        "n_background": args.n_background,
        "shap_nsamples": args.shap_nsamples,
        "mc_passes": args.mc_passes,
        "mc_available": bool(getattr(fp, "has_mc_dropout", True)),
        "shap_impl": "batched_kernel_shap_exact",
        "feature_cols": feature_cols,
        "subjects": subjects.tolist(),
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{args.backbone} fold{args.fold}] done: test acc {metrics['test']['accuracy']:.4f}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
