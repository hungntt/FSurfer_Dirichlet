#!/usr/bin/env python3
"""Nested GridSearch for one backbone and one outer fold.

For every candidate configuration, a model is trained on each inner training
split (early stopping on the inner validation split) and scored by F1 on the
inner validation split. The candidate with the highest mean inner F1 is
selected (ties keep the first candidate in grid order). The final model is then
refit on the first inner split and evaluated once on the outer test fold.

Results are written incrementally, so an interrupted run resumes where it
stopped.
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
    make_predictor,
    compute_metrics,
    final_split,
    grid_candidates,
    inner_splits,
    load_table,
    outer_split,
    patch_environment,
    train_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=["tabm", "resnet", "logreg", "svm", "rf"], required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--split-seed", type=int, default=SEED)
    p.add_argument("--output-root", type=str, default=str(OUTPUT_DIR / "cv_gridsearch"))
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--cand-start", type=int, default=1, help="First candidate id (1-based) handled by this process.")
    p.add_argument("--cand-end", type=int, default=10**6, help="Last candidate id (inclusive) handled by this process.")
    p.add_argument("--finalize", action="store_true", help="Merge all result parts, select the best candidate, and refit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    torch.set_num_threads(args.torch_threads)
    patch_environment()

    out_dir = Path(args.output_root) / f"seed{args.split_seed}" / args.backbone / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "_tmp_ckpt").mkdir(parents=True, exist_ok=True)
    inner_csv = out_dir / f"inner_results_c{args.cand_start:02d}-{args.cand_end:02d}.csv"

    X, y, subjects, feature_cols = load_table()
    outer_tr, test_idx = outer_split(X, y, args.fold, args.split_seed)
    splits = inner_splits(X, y, outer_tr, args.fold, args.split_seed)
    cands = grid_candidates(args.backbone)
    train_seed = args.split_seed + 10000 + args.fold

    t0 = time.time()
    if args.finalize:
        parts = [pd.read_csv(f) for f in sorted(out_dir.glob("inner_results*.csv"))]
        res = pd.concat(parts).drop_duplicates(["candidate_id", "inner_fold"])
        missing = len(cands) * len(splits) - len(res)
        if missing:
            raise RuntimeError(f"{missing} candidate/inner-fold results are missing in {out_dir}")
        finalize(args, out_dir, res, cands, X, y, feature_cols, train_seed)
        return

    done = pd.read_csv(inner_csv) if inner_csv.exists() else pd.DataFrame()
    done_keys = set(zip(done["candidate_id"], done["inner_fold"])) if not done.empty else set()
    rows = done.to_dict("records") if not done.empty else []
    for cid, params in enumerate(cands, start=1):
        if not (args.cand_start <= cid <= args.cand_end):
            continue
        for j, (tr, va) in enumerate(splits, start=1):
            if (cid, j) in done_keys:
                continue
            model = train_model(args.backbone, params, X.iloc[tr], y[tr], X.iloc[va], y[va], seed=train_seed)
            fp = make_predictor(args.backbone, model, feature_cols)
            p = fp.p_mdd(X.iloc[va])
            m = compute_metrics(y[va], (p >= THRESHOLD).astype(int), p)
            rows.append({"candidate_id": cid, "inner_fold": j, "epochs": model.n_epochs_trained_,
                         "params": json.dumps({k: v for k, v in params.items() if k in ("layer_sizes", "dropout", "lr", "weight_decay", "num_blocks", "C", "gamma", "max_depth", "min_samples_leaf", "max_features")}),
                         **m})
            pd.DataFrame(rows).to_csv(inner_csv, index=False)
            del model, fp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[{args.backbone} fold{args.fold}] candidate {cid}/{len(cands)} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[{args.backbone} fold{args.fold}] candidates {args.cand_start}-{args.cand_end} finished", flush=True)


def finalize(args, out_dir, res, cands, X, y, feature_cols, train_seed) -> None:
    import torch
    res.sort_values(["candidate_id", "inner_fold"]).to_csv(out_dir / "inner_results_all.csv", index=False)
    agg = res.groupby("candidate_id").mean(numeric_only=True).drop(columns=["inner_fold"]).reset_index()
    agg = agg.sort_values(["f1", "candidate_id"], ascending=[False, True], kind="mergesort")
    agg.to_csv(out_dir / "inner_summary.csv", index=False)
    best_id = int(agg.iloc[0]["candidate_id"])
    best_params = cands[best_id - 1]

    tr, va, te = final_split(X, y, args.fold, args.split_seed)
    model = train_model(args.backbone, best_params, X.iloc[tr], y[tr], X.iloc[va], y[va], seed=train_seed)
    fp = make_predictor(args.backbone, model, feature_cols)
    split_metrics = {}
    for name, idx in (("train", tr), ("validation", va), ("test", te)):
        p = fp.p_mdd(X.iloc[idx])
        split_metrics[name] = compute_metrics(y[idx], (p >= THRESHOLD).astype(int), p)
    summary = {
        "backbone": args.backbone,
        "fold": args.fold,
        "split_seed": args.split_seed,
        "train_seed": train_seed,
        "best_candidate_id": best_id,
        "best_params": best_params,
        "best_inner_mean": {k: float(v) for k, v in agg.iloc[0].items() if k != "candidate_id"},
        "final_refit_metrics": {k: {m: float(v) for m, v in d.items()} for k, d in split_metrics.items()},
        "split_sizes": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
    }
    (out_dir / "best.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"best_candidate_id": best_id, "test_acc": split_metrics["test"]["accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
