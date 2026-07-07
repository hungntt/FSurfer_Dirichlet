#!/usr/bin/env python3
"""Threshold-0.5 Direction B + Direction C analysis.

This script uses the official TabM decision rule from report.txt:
    pred_label = 1 if p_MDD >= 0.5 else 0.

It runs, in one file:
  Direction B: fixed-p_MDD SHAP-space Top-K similarity + Dirichlet.
  Direction C: Direction-A RIS/ROS local-instability gate + Direction-B Dirichlet evidence.

It does NOT recompute Direction A. It reads completed Direction A outputs from
--direction-a-root. It writes all new Direction B/C outputs under --output-root.

Default output_root is intentionally new and will not overwrite prior results.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd

try:
    from run_tabm_directions_abc_pmdd_corrected_diagnostics import (
        PROJECT_DIR,
        FEATURES_CSV,
        LABELS_CSV,
        SPLITS_CSV,
        CKPT_PATH,
        SELECTED_REFERENCES_CSV,
        ID_COL,
        parse_float_list,
        parse_int_list,
        parse_str_list,
        patch_sklearn_for_pretab,
        load_table,
        load_split_indices,
        load_selected_references,
        load_tabm,
        make_predict_function,
        build_fixed_pmdd_shap_explainer,
        compute_or_load_shap_bank,
        compute_metrics,
        evaluate_split_shap_similarity,
        evaluate_hard_flip_grid,
        select_hard_flip_rule,
        apply_hard_flip_rule,
        evaluate_soft_fusion,
        select_lambda,
        apply_lambda_to_split,
        attach_instability,
        direction_c_hard_flip_grid,
        direction_c_soft_fusion_grid,
        select_best_grid_row,
        apply_direction_c_hard_flip,
        apply_direction_c_soft_fusion,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import run_tabm_directions_abc_pmdd_corrected_diagnostics.py. "
        "Please place this script in the same folder as that file. "
        f"Original error: {exc}"
    )

DEFAULT_DIRECTION_A_ROOT = PROJECT_DIR / "outputs/tabm_pmdd_directionABC_split_corrected_diagnostics"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs/tabm_pmdd_directionBC_threshold05_currentA_diagnostics"


# ---------------------------------------------------------------------------
# Argument parsing and robust I/O.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run official-threshold-0.5 Direction B and Direction C in one script."
    )
    p.add_argument("--features-csv", type=str, default=str(FEATURES_CSV))
    p.add_argument("--labels-csv", type=str, default=str(LABELS_CSV))
    p.add_argument("--splits-csv", type=str, default=str(SPLITS_CSV))
    p.add_argument("--ckpt-path", type=str, default=str(CKPT_PATH))
    p.add_argument("--selected-references-csv", type=str, default=str(SELECTED_REFERENCES_CSV))
    p.add_argument("--direction-a-root", type=str, default=str(DEFAULT_DIRECTION_A_ROOT))
    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))

    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha-prior", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--tau-mode", type=str, default="topk_median", choices=["topk_median", "all20_median", "topk_mean", "all20_mean", "fixed", "none"])
    p.add_argument("--tau-value", type=float, default=None)
    p.add_argument("--similarity-metrics", type=str, default="cosine,euclidean,pearson")
    p.add_argument("--local-instability-metrics", type=str, default="ris,ros")
    p.add_argument("--instability-percentiles", type=str, default="70,80,90")
    p.add_argument("--oppose-weight-thresholds", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75")
    p.add_argument("--min-oppose-counts", type=str, default="0,3")
    p.add_argument("--min-flags-for-rule", type=int, default=5)
    p.add_argument("--lambda-grid", type=str, default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00")
    p.add_argument("--fusion-objective", type=str, choices=["accuracy", "balanced_accuracy", "macro_f1", "mcc"], default="accuracy")

    p.add_argument("--include-self-reference", action="store_true")
    p.add_argument("--nan-as-high-instability", action="store_true")

    p.add_argument("--reuse-shap-cache", action="store_true")
    p.add_argument("--reuse-direction-b", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-per-query-similarity", action="store_true")
    p.add_argument("--reuse-per-query-similarity", action="store_true")

    p.add_argument("--wait-for-direction-a", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wait-poll-seconds", type=float, default=300.0)
    p.add_argument("--wait-timeout-minutes", type=float, default=0.0, help="0 means wait indefinitely")

    p.add_argument("--max-validation-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    return p.parse_args()


def read_required(path: Path, description: str, args: argparse.Namespace) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if not args.wait_for_direction_a:
        raise FileNotFoundError(f"Missing {description}: {path}")
    start_wait = time.time()
    timeout_seconds = None if args.wait_timeout_minutes <= 0 else args.wait_timeout_minutes * 60.0
    print(f"Waiting for {description}: {path}", flush=True)
    while not path.exists():
        elapsed = time.time() - start_wait
        if timeout_seconds is not None and elapsed >= timeout_seconds:
            raise TimeoutError(f"Timed out after {args.wait_timeout_minutes:.2f} minutes waiting for {description}: {path}")
        time.sleep(max(float(args.wait_poll_seconds), 1.0))
        waited_min = (time.time() - start_wait) / 60.0
        print(f"Still waiting for {description} ({waited_min:.1f} min): {path}", flush=True)
    print(f"Found {description}: {path}", flush=True)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Threshold-0.5 prediction handling.
# ---------------------------------------------------------------------------


def add_reference_indices(selected_ref_df: pd.DataFrame, subjects: np.ndarray) -> pd.DataFrame:
    refs = selected_ref_df.copy()
    if "index" in refs.columns:
        refs["index"] = refs["index"].astype(int)
        return refs
    if ID_COL not in refs.columns:
        raise ValueError(
            f"Selected reference CSV must contain either 'index' or '{ID_COL}'. Available columns: {list(refs.columns)}"
        )
    subject_to_index = {str(s): int(i) for i, s in enumerate(subjects)}
    missing = sorted(set(refs[ID_COL].astype(str)) - set(subject_to_index))
    if missing:
        raise ValueError(f"Reference subjects not found in main feature table: {missing[:20]}")
    refs["index"] = refs[ID_COL].astype(str).map(subject_to_index).astype(int)
    return refs


def make_threshold05_lookup(indices, X, y, subjects, feature_cols, predict_function) -> dict:
    proba = np.asarray(predict_function(X.iloc[indices][feature_cols]), dtype=float)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"Expected predict_function to return shape (n,2), got {proba.shape}")
    lookup = {}
    for row_i, idx in enumerate(indices):
        idx = int(idx)
        subj = str(subjects[idx])
        p_hc = float(proba[row_i, 0])
        p_mdd = float(proba[row_i, 1])
        pred = int(p_mdd >= 0.5)
        lookup[subj] = {
            "tabm_p_HC": p_hc,
            "tabm_p_MDD": p_mdd,
            "tabm_pred_label": pred,
            "tabm_pred_confidence": float(p_mdd if pred == 1 else 1.0 - p_mdd),
            "true_label": int(y[idx]),
        }
    return lookup


def overwrite_shap_df_predictions_threshold05(shap_df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    out = shap_df.copy()
    missing = sorted(set(out[ID_COL].astype(str)) - set(lookup))
    if missing:
        raise ValueError(f"Missing subjects in threshold-0.5 lookup: {missing[:10]}")
    out["p_MDD"] = out[ID_COL].astype(str).map(lambda s: lookup[s]["tabm_p_MDD"]).astype(float)
    out["pred_label"] = out[ID_COL].astype(str).map(lambda s: lookup[s]["tabm_pred_label"]).astype(int)
    out["pred_class"] = out["pred_label"].map({0: "HC", 1: "MDD"})
    out["pred_confidence"] = out[ID_COL].astype(str).map(lambda s: lookup[s]["tabm_pred_confidence"]).astype(float)
    out["true_label"] = out[ID_COL].astype(str).map(lambda s: lookup[s]["true_label"]).astype(int)
    return out


def apply_threshold05_lookup_to_diagnostics(df: pd.DataFrame, lookup: dict, top_k: int) -> pd.DataFrame:
    out = df.copy()
    missing = sorted(set(out["subject"].astype(str)) - set(lookup))
    if missing:
        raise ValueError(f"Missing subjects in threshold-0.5 lookup: {missing[:10]}")
    out["tabm_p_HC"] = out["subject"].astype(str).map(lambda s: lookup[s]["tabm_p_HC"]).astype(float)
    out["tabm_p_MDD"] = out["subject"].astype(str).map(lambda s: lookup[s]["tabm_p_MDD"]).astype(float)
    out["tabm_pred_label"] = out["subject"].astype(str).map(lambda s: lookup[s]["tabm_pred_label"]).astype(int)
    out["tabm_pred_class"] = out["tabm_pred_label"].map({0: "HC", 1: "MDD"})
    out["tabm_pred_confidence"] = out["subject"].astype(str).map(lambda s: lookup[s]["tabm_pred_confidence"]).astype(float)
    out["true_label"] = out["subject"].astype(str).map(lambda s: lookup[s]["true_label"]).astype(int)
    out["tabm_correct"] = out["tabm_pred_label"].astype(int) == out["true_label"].astype(int)

    # Recompute support/oppose after threshold-0.5 prediction overwrite.
    w_hc = out["dirichlet_weight_HC"].astype(float)
    w_mdd = out["dirichlet_weight_MDD"].astype(float)
    pred = out["tabm_pred_label"].astype(int)
    out["support_class"] = pred.map({0: "HC", 1: "MDD"})
    out["oppose_class"] = pred.map({0: "MDD", 1: "HC"})
    out["support_weight"] = np.where(pred.to_numpy() == 1, w_mdd.to_numpy(), w_hc.to_numpy())
    out["oppose_weight"] = np.where(pred.to_numpy() == 1, w_hc.to_numpy(), w_mdd.to_numpy())

    def count_support(row):
        labels = [int(x) for x in str(row["topk_reference_labels"]).split(";") if str(x).strip() != ""]
        return sum(1 for lab in labels if lab == int(row["tabm_pred_label"]))

    out["support_count"] = out.apply(count_support, axis=1).astype(int)
    out["oppose_count"] = int(top_k) - out["support_count"]
    return out


def offline_metrics_from_lookup(lookup: dict) -> dict:
    y_true = np.asarray([v["true_label"] for v in lookup.values()], dtype=int)
    y_pred = np.asarray([v["tabm_pred_label"] for v in lookup.values()], dtype=int)
    score = np.asarray([v["tabm_p_MDD"] for v in lookup.values()], dtype=float)
    return compute_metrics(y_true, y_pred, score)


def selected_lambda_value(row: dict) -> float:
    if "selected_lambda" in row:
        return float(row["selected_lambda"])
    if "lambda" in row:
        return float(row["lambda"])
    raise KeyError(f"Selected lambda row contains neither selected_lambda nor lambda: {list(row.keys())}")


def selected_lambda_metric(row: dict, metric: str):
    if metric in row:
        return row[metric]
    key = f"validation_selected_{metric}"
    if key in row:
        return row[key]
    return pd.NA


def metric_value_from_metrics_df(metrics_df: pd.DataFrame, stage: str, metric: str) -> float:
    row = metrics_df.loc[metrics_df["stage"] == stage]
    if row.empty:
        return float(metrics_df[metric].iloc[-1])
    return float(row[metric].iloc[0])


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    start = pd.Timestamp.utcnow()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    direction_a_root = Path(args.direction_a_root)

    patch_sklearn_for_pretab()

    similarity_metrics = parse_str_list(args.similarity_metrics)
    if not set(similarity_metrics).issubset({"cosine", "euclidean", "pearson"}):
        raise ValueError("--similarity-metrics must contain only cosine, euclidean, pearson")
    local_metric_names = parse_str_list(args.local_instability_metrics)
    instability_cols = []
    for m in local_metric_names:
        if m == "ris":
            instability_cols.append("local_ris_instability")
        elif m == "ros":
            instability_cols.append("local_ros_instability")
        else:
            raise ValueError("--local-instability-metrics must contain only ris and/or ros")
    instability_percentiles = parse_float_list(args.instability_percentiles)
    oppose_thresholds = parse_float_list(args.oppose_weight_thresholds)
    min_counts = parse_int_list(args.min_oppose_counts)
    lambdas = parse_float_list(args.lambda_grid)

    X, y, subjects, feature_cols = load_table(Path(args.features_csv), Path(args.labels_csv))
    train_idx, val_idx, test_idx, _ = load_split_indices(Path(args.splits_csv), subjects, y)
    if args.max_validation_samples is not None:
        val_idx = val_idx[: int(args.max_validation_samples)]
    if args.max_test_samples is not None:
        test_idx = test_idx[: int(args.max_test_samples)]

    selected_ref_df = load_selected_references(Path(args.selected_references_csv), subjects, y)
    selected_ref_df = add_reference_indices(selected_ref_df, subjects)
    ref_idx = selected_ref_df["index"].astype(int).to_numpy()
    ref_subjects = set(selected_ref_df[ID_COL].astype(str).tolist())

    print("Loading TabM checkpoint...", flush=True)
    model = load_tabm(Path(args.ckpt_path), X.iloc[train_idx][feature_cols], y[train_idx], X.iloc[val_idx][feature_cols], y[val_idx])
    predict_function = make_predict_function(model, feature_cols)
    shap_explainer = build_fixed_pmdd_shap_explainer(predict_function, X, feature_cols, train_idx)

    # Official threshold-0.5 baseline.
    val_lookup = make_threshold05_lookup(val_idx, X, y, subjects, feature_cols, predict_function)
    test_lookup = make_threshold05_lookup(test_idx, X, y, subjects, feature_cols, predict_function)
    offline_val_metrics = offline_metrics_from_lookup(val_lookup)
    offline_test_metrics = offline_metrics_from_lookup(test_lookup)
    pd.DataFrame([
        {"stage": "offline_validation_tabm_threshold05", **offline_val_metrics},
        {"stage": "offline_test_tabm_threshold05", **offline_test_metrics},
    ]).to_csv(output_root / "offline_tabm_threshold05_metrics.csv", index=False)
    selected_ref_df.to_csv(output_root / "selected_reference_subjects_used.csv", index=False)

    # Fixed-pMDD SHAP banks. The helper may cache argmax labels, so overwrite labels with official threshold-0.5 after loading.
    ref_shap_df = compute_or_load_shap_bank("reference20_fixed_pmdd", ref_idx, X, y, subjects, feature_cols, predict_function, shap_explainer, output_root, args.reuse_shap_cache)
    val_shap_df = compute_or_load_shap_bank("validation_fixed_pmdd", val_idx, X, y, subjects, feature_cols, predict_function, shap_explainer, output_root, args.reuse_shap_cache)
    test_shap_df = compute_or_load_shap_bank("test_fixed_pmdd", test_idx, X, y, subjects, feature_cols, predict_function, shap_explainer, output_root, args.reuse_shap_cache)

    # Reference pred labels are not used as class evidence. True labels are used.
    # Still overwrite for traceability.
    ref_lookup = make_threshold05_lookup(ref_idx, X, y, subjects, feature_cols, predict_function)
    ref_shap_df = overwrite_shap_df_predictions_threshold05(ref_shap_df, ref_lookup)
    val_shap_df = overwrite_shap_df_predictions_threshold05(val_shap_df, val_lookup)
    test_shap_df = overwrite_shap_df_predictions_threshold05(test_shap_df, test_lookup)

    # Direction A outputs.
    val_inst = read_required(direction_a_root / "directionA_local_instability" / "local_instability_validation.csv", "Direction A validation local instability", args)
    test_inst = read_required(direction_a_root / "directionA_local_instability" / "local_instability_test.csv", "Direction A test local instability", args)

    b_summary_rows = []
    c_summary_rows = []
    b_all_test_rows = []
    c_all_test_rows = []

    for sim_metric in similarity_metrics:
        print(f"\nRunning Direction B threshold-0.5 for similarity={sim_metric}", flush=True)
        val_b = evaluate_split_shap_similarity("validation", val_shap_df, ref_shap_df, subjects, feature_cols, sim_metric, output_root, ref_subjects, args)
        test_b = evaluate_split_shap_similarity("test", test_shap_df, ref_shap_df, subjects, feature_cols, sim_metric, output_root, ref_subjects, args)

        # Force threshold-0.5 official baseline into diagnostic tables.
        val_b = apply_threshold05_lookup_to_diagnostics(val_b, val_lookup, args.top_k)
        test_b = apply_threshold05_lookup_to_diagnostics(test_b, test_lookup, args.top_k)
        b_dir = output_root / "directionB_threshold05_shap_similarity" / f"metric_{sim_metric}"
        b_dir.mkdir(parents=True, exist_ok=True)
        val_b.to_csv(b_dir / "online_validation_diagnostics.csv", index=False)
        test_b.to_csv(b_dir / "online_test_diagnostics.csv", index=False)

        hard_grid = evaluate_hard_flip_grid(val_b, oppose_thresholds, min_counts, args.min_flags_for_rule)
        hard_grid.to_csv(b_dir / "validation_hard_flip_grid.csv", index=False)
        selected_hard = select_hard_flip_rule(hard_grid, args.fusion_objective, offline_val_metrics[args.fusion_objective])
        pd.DataFrame([selected_hard]).to_csv(b_dir / "selected_validation_hard_flip_rule.csv", index=False)
        _, test_hard_metrics = apply_hard_flip_rule(test_b, selected_hard)
        test_hard_metrics.insert(0, "direction", "B_threshold05_shap_similarity")
        test_hard_metrics.insert(1, "similarity_metric", sim_metric)
        test_hard_metrics.to_csv(b_dir / "test_hard_flip_selected_rule_metrics.csv", index=False)

        lambda_grid = evaluate_soft_fusion(val_b, lambdas)
        lambda_grid.to_csv(b_dir / "validation_soft_fusion_lambda_grid.csv", index=False)
        selected_lambda = select_lambda(lambda_grid, args.fusion_objective)
        pd.DataFrame([selected_lambda]).to_csv(b_dir / "selected_validation_lambda.csv", index=False)
        lam = selected_lambda_value(selected_lambda)
        _, test_soft_metrics = apply_lambda_to_split(test_b, lam)
        test_soft_metrics.insert(0, "direction", "B_threshold05_shap_similarity")
        test_soft_metrics.insert(1, "similarity_metric", sim_metric)
        test_soft_metrics.to_csv(b_dir / "test_soft_fusion_metrics.csv", index=False)

        # Sanity checks under official threshold-0.5 baseline.
        if selected_hard.get("selected_policy") == "no_flip":
            hard_acc = metric_value_from_metrics_df(test_hard_metrics, "hard_flip_selected_rule", "accuracy")
            if abs(hard_acc - float(offline_test_metrics["accuracy"])) > 1e-12:
                raise RuntimeError(f"Direction B sanity check failed: no_flip accuracy {hard_acc} != baseline {offline_test_metrics['accuracy']} for {sim_metric}")
        if abs(lam - 1.0) <= 1e-12:
            soft_acc = metric_value_from_metrics_df(test_soft_metrics, "soft_fusion", "accuracy")
            if abs(soft_acc - float(offline_test_metrics["accuracy"])) > 1e-12:
                raise RuntimeError(f"Direction B sanity check failed: lambda=1 accuracy {soft_acc} != baseline {offline_test_metrics['accuracy']} for {sim_metric}")

        b_all_test_rows.extend(test_hard_metrics.to_dict("records"))
        b_all_test_rows.extend(test_soft_metrics.to_dict("records"))
        b_summary_rows.append({
            "direction": "B_threshold05_shap_similarity",
            "similarity_metric": sim_metric,
            "reference_samples_used": len(ref_idx),
            "reference_label_counts": selected_ref_df["true_label"].value_counts().sort_index().to_dict(),
            "validation_baseline_accuracy": offline_val_metrics["accuracy"],
            "test_baseline_accuracy": offline_test_metrics["accuracy"],
            "selected_hard_policy": selected_hard.get("selected_policy"),
            "selected_hard_reason": selected_hard.get("reason"),
            "test_hard_flip_accuracy": metric_value_from_metrics_df(test_hard_metrics, "hard_flip_selected_rule", "accuracy"),
            "test_hard_flip_mcc": metric_value_from_metrics_df(test_hard_metrics, "hard_flip_selected_rule", "mcc"),
            "selected_lambda": lam,
            "validation_soft_fusion_accuracy": selected_lambda_metric(selected_lambda, "accuracy"),
            "test_soft_fusion_accuracy": metric_value_from_metrics_df(test_soft_metrics, "soft_fusion", "accuracy"),
            "test_soft_fusion_mcc": metric_value_from_metrics_df(test_soft_metrics, "soft_fusion", "mcc"),
        })

        # Direction C for this Direction B similarity metric.
        for inst_col in instability_cols:
            if inst_col not in val_inst.columns or inst_col not in test_inst.columns:
                print(f"Skipping Direction C gate {inst_col}: not found in Direction A outputs.", flush=True)
                continue
            print(f"Running Direction C threshold-0.5 for similarity={sim_metric}, gate={inst_col}", flush=True)
            c_dir = output_root / "directionC_threshold05_gated_hybrid" / f"gate_{inst_col}" / f"metric_{sim_metric}"
            c_dir.mkdir(parents=True, exist_ok=True)
            val_c = attach_instability(val_b, val_inst, inst_col, args.nan_as_high_instability)
            test_c = attach_instability(test_b, test_inst, inst_col, args.nan_as_high_instability)
            val_c.to_csv(c_dir / "validation_gated_input_table.csv", index=False)
            test_c.to_csv(c_dir / "test_gated_input_table.csv", index=False)

            inst_eff_col = f"{inst_col}_effective"
            finite_val = val_c[inst_eff_col].astype(float).to_numpy()
            finite_val = finite_val[np.isfinite(finite_val)]
            if finite_val.size == 0:
                print(f"Skipping Direction C {sim_metric}+{inst_col}: no finite validation instability scores.", flush=True)
                continue
            inst_thresholds = [(float(p), float(np.percentile(finite_val, p))) for p in instability_percentiles]

            c_hard_grid = direction_c_hard_flip_grid(val_c, inst_eff_col, inst_thresholds, oppose_thresholds, min_counts, args.min_flags_for_rule)
            c_hard_grid.to_csv(c_dir / "validation_gated_hard_flip_grid.csv", index=False)
            selected_c_hard = select_best_grid_row(c_hard_grid, args.fusion_objective, offline_val_metrics[args.fusion_objective])
            pd.DataFrame([selected_c_hard]).to_csv(c_dir / "selected_validation_gated_hard_flip_rule.csv", index=False)
            _, test_c_hard_metrics = apply_direction_c_hard_flip(test_c, inst_eff_col, selected_c_hard)
            test_c_hard_metrics.insert(0, "direction", "C_threshold05_gated_hybrid")
            test_c_hard_metrics.insert(1, "similarity_metric", sim_metric)
            test_c_hard_metrics.insert(2, "instability_gate", inst_col)
            test_c_hard_metrics.to_csv(c_dir / "test_gated_hard_flip_metrics.csv", index=False)

            c_soft_grid = direction_c_soft_fusion_grid(val_c, inst_eff_col, inst_thresholds, lambdas)
            c_soft_grid.to_csv(c_dir / "validation_gated_soft_fusion_grid.csv", index=False)
            selected_c_soft = select_best_grid_row(c_soft_grid, args.fusion_objective, offline_val_metrics[args.fusion_objective])
            pd.DataFrame([selected_c_soft]).to_csv(c_dir / "selected_validation_gated_soft_fusion.csv", index=False)
            _, test_c_soft_metrics = apply_direction_c_soft_fusion(test_c, inst_eff_col, selected_c_soft)
            test_c_soft_metrics.insert(0, "direction", "C_threshold05_gated_hybrid")
            test_c_soft_metrics.insert(1, "similarity_metric", sim_metric)
            test_c_soft_metrics.insert(2, "instability_gate", inst_col)
            test_c_soft_metrics.to_csv(c_dir / "test_gated_soft_fusion_metrics.csv", index=False)

            # Sanity checks: no_change must equal official threshold-0.5 baseline.
            if selected_c_hard.get("selected_policy") == "no_change":
                hard_acc = metric_value_from_metrics_df(test_c_hard_metrics, "directionC_gated_hard_flip", "accuracy")
                if abs(hard_acc - float(offline_test_metrics["accuracy"])) > 1e-12:
                    raise RuntimeError(f"Direction C hard no_change sanity failed for {sim_metric}+{inst_col}: {hard_acc} != {offline_test_metrics['accuracy']}")
            if selected_c_soft.get("selected_policy") == "no_change":
                soft_acc = metric_value_from_metrics_df(test_c_soft_metrics, "directionC_gated_soft_fusion", "accuracy")
                if abs(soft_acc - float(offline_test_metrics["accuracy"])) > 1e-12:
                    raise RuntimeError(f"Direction C soft no_change sanity failed for {sim_metric}+{inst_col}: {soft_acc} != {offline_test_metrics['accuracy']}")

            c_all_test_rows.extend(test_c_hard_metrics.to_dict("records"))
            c_all_test_rows.extend(test_c_soft_metrics.to_dict("records"))
            c_summary_rows.append({
                "direction": "C_threshold05_gated_hybrid",
                "similarity_metric": sim_metric,
                "instability_gate": inst_col,
                "direction_a_root": str(direction_a_root),
                "validation_baseline_accuracy": offline_val_metrics["accuracy"],
                "test_baseline_accuracy": offline_test_metrics["accuracy"],
                "selected_gated_hard_policy": selected_c_hard.get("selected_policy"),
                "selected_gated_hard_reason": selected_c_hard.get("reason"),
                "test_gated_hard_flip_accuracy": metric_value_from_metrics_df(test_c_hard_metrics, "directionC_gated_hard_flip", "accuracy"),
                "test_gated_hard_flip_mcc": metric_value_from_metrics_df(test_c_hard_metrics, "directionC_gated_hard_flip", "mcc"),
                "selected_gated_soft_policy": selected_c_soft.get("selected_policy"),
                "selected_gated_soft_reason": selected_c_soft.get("reason"),
                "selected_gated_soft_lambda": float(selected_c_soft.get("lambda", np.nan)) if "lambda" in selected_c_soft else np.nan,
                "test_gated_soft_fusion_accuracy": metric_value_from_metrics_df(test_c_soft_metrics, "directionC_gated_soft_fusion", "accuracy"),
                "test_gated_soft_fusion_mcc": metric_value_from_metrics_df(test_c_soft_metrics, "directionC_gated_soft_fusion", "mcc"),
            })

    b_summary_df = pd.DataFrame(b_summary_rows)
    c_summary_df = pd.DataFrame(c_summary_rows)
    b_summary_df.to_csv(output_root / "directionB_threshold05_validation_test_summary.csv", index=False)
    c_summary_df.to_csv(output_root / "directionC_threshold05_validation_test_summary.csv", index=False)
    pd.DataFrame(b_all_test_rows).to_csv(output_root / "directionB_threshold05_all_test_metrics.csv", index=False)
    pd.DataFrame(c_all_test_rows).to_csv(output_root / "directionC_threshold05_all_test_metrics.csv", index=False)

    elapsed = (pd.Timestamp.utcnow() - start).total_seconds() / 60.0
    lines = []
    lines.append("Threshold-0.5 Direction B + Direction C summary")
    lines.append("================================================")
    lines.append("")
    lines.append(f"features_csv: {Path(args.features_csv)}")
    lines.append(f"labels_csv:   {Path(args.labels_csv)}")
    lines.append(f"splits_csv:   {Path(args.splits_csv)}")
    lines.append(f"ckpt_path:    {Path(args.ckpt_path)}")
    lines.append(f"selected_references_csv: {Path(args.selected_references_csv)}")
    lines.append(f"direction_a_root: {direction_a_root}")
    lines.append(f"output_root: {output_root}")
    lines.append(f"train/validation/test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    lines.append(f"reference_samples_used: {len(ref_idx)}")
    lines.append(f"reference_label_counts: {selected_ref_df['true_label'].value_counts().sort_index().to_dict()}")
    lines.append("prediction_rule: official threshold p_MDD >= 0.5")
    lines.append(f"similarity_metrics: {similarity_metrics}")
    lines.append(f"instability_gates: {instability_cols}")
    lines.append(f"elapsed_minutes: {elapsed:.3f}")
    lines.append("")
    lines.append("Offline TabM threshold-0.5 metrics")
    lines.append("-----------------------------------")
    lines.append(pd.DataFrame([
        {"split": "validation", **offline_val_metrics},
        {"split": "test", **offline_test_metrics},
    ]).to_string(index=False))
    lines.append("")
    lines.append("Direction B threshold-0.5 validation-test summary")
    lines.append("--------------------------------------------------")
    lines.append(b_summary_df.to_string(index=False) if not b_summary_df.empty else "No Direction B rows available.")
    lines.append("")
    lines.append("Direction C threshold-0.5 validation-test summary")
    lines.append("--------------------------------------------------")
    lines.append(c_summary_df.to_string(index=False) if not c_summary_df.empty else "No Direction C rows available.")
    lines.append("")
    lines.append("Interpretation reminder: any no_flip/no_change row must equal the threshold-0.5 TabM baseline. If an intervention improves validation but lowers test, it should be rejected as non-generalizable.")
    (output_root / "directionBC_threshold05_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-40:]))


if __name__ == "__main__":
    main()
