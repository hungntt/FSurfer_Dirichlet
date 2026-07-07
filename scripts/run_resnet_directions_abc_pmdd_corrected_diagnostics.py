#!/usr/bin/env python3
"""Direction A/B/C diagnostics with correct RIS/ROS roles and SHAP-space retrieval.

This script intentionally writes to a NEW output directory and does not overwrite
previous Version-A/RIS/ROS outputs.

Purpose
-------
After reading Quantus RelativeInputStability and RelativeOutputStability, RIS/ROS
should be used as local explanation-instability scores under perturbations of the
same query sample, not as query-reference similarity scores.

This script tests three logically separated directions:

Direction A: Correct local Quantus-style RIS/ROS instability.
    - No reference bank.
    - No Top-K.
    - No Dirichlet.
    - Evaluate whether high local instability enriches ResNet errors.

Direction B: Direct fixed-p_MDD SHAP-space reference retrieval.
    - No RIS/ROS.
    - Use cosine/euclidean/pearson distance between signed p_MDD SHAP vectors.
    - Top-K reference retrieval -> class-level Dirichlet -> validation-calibrated
      hard flip and soft fusion -> frozen test evaluation.

Direction C: Hybrid gated correction.
    - Use local RIS/ROS instability as a gate.
    - Only for unstable samples, use Direction-B SHAP Top-K Dirichlet evidence.
    - Evaluate gated hard flip and gated soft fusion with validation-selected rules.

Important interpretation
------------------------
- Fixed p_MDD SHAP is used everywhere. Positive SHAP pushes toward MDD; negative
  SHAP pushes away from MDD, i.e. toward HC.
- Direction A cannot itself decide HC/MDD. It only detects instability.
- Direction B and C can produce class evidence because they use reference labels.
- All thresholds/lambdas are selected on validation and applied once to test.

Runtime warning
---------------
Direction A/C require SHAP explanations for perturbed validation/test samples.
This can be expensive. Start with --n-local-perturbations 5 or 10, then increase
only if the diagnostic is promising.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import contextlib
import math
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

# The existing fixed-pMDD script is used only as a core utility module for
# loading ResNet, computing p_MDD SHAP, and common metrics. It must be present in
# the same folder as this script on the server.
try:
    from run_resnet_pmdd_fixed_shap_ris_dirichlet_diagnostics import (
        PROJECT_DIR,
        FEATURES_CSV,
        LABELS_CSV,
        SPLITS_CSV,
        CKPT_PATH,
        SELECTED_REFERENCES_CSV,
        ID_COL,
        Y_COL,
        CLASS_NAMES,
        RANDOM_STATE,
        EXPECTED_N_FEATURES,
        EPS,
        EPS_MIN,
        parse_float_list,
        parse_int_list,
        parse_str_list,
        patch_sklearn_for_pretab,
        subject_to_safe_name,
        load_table,
        load_split_indices,
        load_selected_references,
        load_resnet,
        make_predict_function,
        build_fixed_pmdd_shap_explainer,
        compute_fixed_pmdd_shap_vector,
        compute_or_load_shap_bank,
        compute_metrics,
        build_error_enrichment_grid,
        evaluate_soft_fusion,
        select_lambda,
        apply_lambda_to_split,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import run_resnet_pmdd_fixed_shap_ris_dirichlet_diagnostics.py. "
        "Please place this script in the same folder as that existing file. "
        f"Original error: {exc}"
    )

OUTPUT_ROOT = PROJECT_DIR / "outputs/resnet_pmdd_directionABC_corrected_diagnostics"
TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correct Direction A/B/C diagnostics: local RIS/ROS, SHAP similarity Top-K, and gated hybrid correction."
    )
    parser.add_argument("--features-csv", type=str, default=str(FEATURES_CSV))
    parser.add_argument("--labels-csv", type=str, default=str(LABELS_CSV))
    parser.add_argument("--splits-csv", type=str, default=str(SPLITS_CSV))
    parser.add_argument("--ckpt-path", type=str, default=str(CKPT_PATH))
    parser.add_argument("--selected-references-csv", type=str, default=str(SELECTED_REFERENCES_CSV))
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))

    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--alpha-prior", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--tau-mode",
        type=str,
        choices=["topk_median", "allref_median", "topk_mean", "allref_mean", "fixed", "none"],
        default="topk_median",
    )
    parser.add_argument("--tau-value", type=float, default=None)
    parser.add_argument(
        "--similarity-metrics",
        type=str,
        default="cosine,euclidean,pearson",
        help="Comma-separated SHAP-space metrics: cosine,euclidean,pearson. All are converted to distances where lower is better.",
    )
    parser.add_argument(
        "--oppose-weight-thresholds",
        type=str,
        default="0.50,0.55,0.60,0.65,0.70,0.75",
    )
    parser.add_argument("--min-oppose-counts", type=str, default="0,3")
    parser.add_argument("--min-flags-for-rule", type=int, default=5)
    parser.add_argument(
        "--lambda-grid",
        type=str,
        default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00",
    )
    parser.add_argument(
        "--fusion-objective",
        type=str,
        choices=["accuracy", "balanced_accuracy", "macro_f1", "mcc"],
        default="accuracy",
    )

    parser.add_argument(
        "--local-instability-metrics",
        type=str,
        default="ris,ros",
        help="Comma-separated local instability metrics for Direction A/C: ris,ros.",
    )
    parser.add_argument(
        "--n-local-perturbations",
        type=int,
        default=10,
        help="Number of local perturbations per validation/test sample. Runtime scales linearly with this.",
    )
    parser.add_argument(
        "--perturbation-mode",
        type=str,
        choices=["relative_uniform", "std_uniform"],
        default="relative_uniform",
        help="relative_uniform: x' = x*(1+Uniform[-scale,scale]). std_uniform: x' = x + Uniform[-scale,scale]*train_std.",
    )
    parser.add_argument("--perturbation-scale", type=float, default=0.02)
    parser.add_argument(
        "--instability-percentiles",
        type=str,
        default="70,80,90",
        help="Validation percentiles used as high-instability gates for Direction A/C.",
    )
    parser.add_argument(
        "--nan-as-high-instability",
        action="store_true",
        help="Treat local RIS/ROS NaN as high instability when creating gates. NaN can occur if perturbations change prediction under strict Quantus semantics.",
    )
    parser.add_argument(
        "--strict-nan-when-prediction-changes",
        action="store_true",
        help="If any perturbation changes predicted class, set the aggregate local RIS/ROS score to NaN, matching the conservative Quantus default behavior.",
    )

    parser.add_argument("--include-self-reference", action="store_true")
    parser.add_argument("--reuse-shap-cache", action="store_true")
    parser.add_argument("--reuse-local-instability", action="store_true")
    parser.add_argument("--reuse-direction-b", action="store_true")
    parser.add_argument("--save-per-query-similarity", action="store_true")
    parser.add_argument("--reuse-per-query-similarity", action="store_true")
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility functions.
# ---------------------------------------------------------------------------


def logit_prob_pair(proba: np.ndarray) -> np.ndarray:
    """Stable log-probability surrogate for output-logit distance.

    Quantus ROS expects output logits. DeepTab's public sklearn-like API exposes
    probabilities. We therefore use log probabilities as a monotone logit-space
    surrogate and report this explicitly in the output files.
    """
    p = np.clip(np.asarray(proba, dtype=float), EPS_MIN, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.log(p)


def compute_shap_vector_for_array(explainer, arr_1d: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    x = np.asarray(arr_1d, dtype=float).reshape(1, -1)
    l1_reg = f"num_features({len(feature_cols)})"
    values = explainer.shap_values(x, nsamples=800, l1_reg=l1_reg)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim == 2:
        values = values[0]
    values = values.reshape(-1)
    if values.shape[0] != len(feature_cols):
        raise ValueError(f"Expected {len(feature_cols)} SHAP values, got {values.shape[0]}")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite SHAP values detected for perturbed sample.")
    return values.astype(float)


def norm_2d_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.linalg.norm(arr, axis=1)


def relative_input_stability_objective(x: np.ndarray, xs: np.ndarray, e_x: np.ndarray, e_xs: np.ndarray) -> float:
    """Quantus-style RIS objective for one local perturbation.

    This follows relative_input_stability.py:
        ||(e_x - e_xs) / e_x|| / ||(x - xs) / x||
    for tabular 2D inputs with batch dimension included.
    """
    x = np.asarray(x, dtype=float).reshape(1, -1)
    xs = np.asarray(xs, dtype=float).reshape(1, -1)
    e_x = np.asarray(e_x, dtype=float).reshape(1, -1)
    e_xs = np.asarray(e_xs, dtype=float).reshape(1, -1)
    nom = (e_x - e_xs) / (e_x + (e_x == 0) * EPS_MIN)
    nom = norm_2d_rows(nom)[0]
    den = (x - xs) / (x + (x == 0) * EPS_MIN)
    den = norm_2d_rows(den)[0]
    den = den + (den == 0) * EPS_MIN
    return float(nom / den)


def relative_output_stability_objective(h_x: np.ndarray, h_xs: np.ndarray, e_x: np.ndarray, e_xs: np.ndarray) -> float:
    """Quantus-style ROS objective for one local perturbation.

    This follows relative_output_stability.py:
        ||(e_x - e_xs) / e_x|| / ||h(x) - h(xs)||
    where h is a log-probability surrogate because raw logits are not exposed.
    """
    h_x = np.asarray(h_x, dtype=float).reshape(1, -1)
    h_xs = np.asarray(h_xs, dtype=float).reshape(1, -1)
    e_x = np.asarray(e_x, dtype=float).reshape(1, -1)
    e_xs = np.asarray(e_xs, dtype=float).reshape(1, -1)
    nom = (e_x - e_xs) / (e_x + (e_x == 0) * EPS_MIN)
    nom = norm_2d_rows(nom)[0]
    den = np.linalg.norm(h_x - h_xs, axis=1)[0]
    den = den + (den == 0) * EPS_MIN
    return float(nom / den)


def perturb_sample(
    x: np.ndarray,
    rng: np.random.Generator,
    mode: str,
    scale: float,
    train_std: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if mode == "relative_uniform":
        noise = rng.uniform(-scale, scale, size=x.shape)
        return x * (1.0 + noise)
    if mode == "std_uniform":
        noise = rng.uniform(-scale, scale, size=x.shape)
        return x + noise * np.asarray(train_std, dtype=float)
    raise ValueError(f"Unknown perturbation mode: {mode}")


def effective_instability_scores(scores: np.ndarray, nan_as_high: bool) -> np.ndarray:
    vals = np.asarray(scores, dtype=float).copy()
    if np.isfinite(vals).all():
        return vals
    if nan_as_high:
        finite = vals[np.isfinite(vals)]
        high = float(np.nanmax(finite)) * 1.05 if finite.size else 1.0
        vals[~np.isfinite(vals)] = high
    return vals


# ---------------------------------------------------------------------------
# Direction A: local Quantus-style RIS/ROS instability.
# ---------------------------------------------------------------------------


def compute_or_load_local_instability(
    split_name: str,
    query_shap_df: pd.DataFrame,
    X: pd.DataFrame,
    feature_cols: list[str],
    predict_function,
    shap_explainer,
    output_root: Path,
    args: argparse.Namespace,
    train_std: np.ndarray,
) -> pd.DataFrame:
    out_dir = output_root / "directionA_local_instability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"local_instability_{split_name}.csv"
    if args.reuse_local_instability and out_path.exists():
        return pd.read_csv(out_path)

    rng = np.random.default_rng(RANDOM_STATE + (0 if split_name == "validation" else 1000))
    metrics_requested = set(parse_str_list(args.local_instability_metrics))
    if not metrics_requested.issubset({"ris", "ros"}):
        raise ValueError("--local-instability-metrics must contain only ris and/or ros.")

    rows = []
    pert_rows_all = []
    print(
        f"\nDirection A: computing local RIS/ROS instability for split={split_name}, "
        f"n={len(query_shap_df)}, perturbations={args.n_local_perturbations}"
    )
    for q_pos, (_, qrow) in enumerate(query_shap_df.iterrows(), start=1):
        subject = str(qrow[ID_COL])
        print(f"  [{q_pos}/{len(query_shap_df)}] local instability for {split_name} subject {subject}")
        idx = int(qrow["index"])
        x = X.iloc[idx][feature_cols].astype(float).to_numpy()
        e_x = qrow[feature_cols].astype(float).to_numpy()
        proba_x = np.asarray(predict_function(pd.DataFrame([x], columns=feature_cols))[0], dtype=float)
        pred_x = int(np.argmax(proba_x))
        h_x = logit_prob_pair(proba_x.reshape(1, -1))[0]

        ris_vals = []
        ros_vals = []
        pred_changed_flags = []
        for p_i in range(int(args.n_local_perturbations)):
            xs = perturb_sample(x, rng, args.perturbation_mode, args.perturbation_scale, train_std)
            proba_xs = np.asarray(predict_function(pd.DataFrame([xs], columns=feature_cols))[0], dtype=float)
            pred_xs = int(np.argmax(proba_xs))
            pred_changed = pred_xs != pred_x
            pred_changed_flags.append(pred_changed)
            h_xs = logit_prob_pair(proba_xs.reshape(1, -1))[0]
            e_xs = compute_shap_vector_for_array(shap_explainer, xs, feature_cols)

            ris_val = relative_input_stability_objective(x, xs, e_x, e_xs) if "ris" in metrics_requested else np.nan
            ros_val = relative_output_stability_objective(h_x, h_xs, e_x, e_xs) if "ros" in metrics_requested else np.nan
            if args.strict_nan_when_prediction_changes and pred_changed:
                ris_val = np.nan if "ris" in metrics_requested else ris_val
                ros_val = np.nan if "ros" in metrics_requested else ros_val
            ris_vals.append(ris_val)
            ros_vals.append(ros_val)
            pert_rows_all.append(
                {
                    "split": split_name,
                    "subject": subject,
                    "perturbation_id": p_i,
                    "true_label": int(qrow["true_label"]),
                    "tabm_pred_label": pred_x,
                    "perturbed_pred_label": pred_xs,
                    "prediction_changed": bool(pred_changed),
                    "original_p_MDD": float(proba_x[1]),
                    "perturbed_p_MDD": float(proba_xs[1]),
                    "ris_value": float(ris_val) if np.isfinite(ris_val) else np.nan,
                    "ros_value": float(ros_val) if np.isfinite(ros_val) else np.nan,
                }
            )

        ris_arr = np.asarray(ris_vals, dtype=float)
        ros_arr = np.asarray(ros_vals, dtype=float)
        any_changed = bool(np.any(pred_changed_flags))
        row = {
            "split": split_name,
            "subject": subject,
            "true_label": int(qrow["true_label"]),
            "true_class": CLASS_NAMES[int(qrow["true_label"])],
            "tabm_pred_label": pred_x,
            "tabm_pred_class": CLASS_NAMES[pred_x],
            "tabm_p_MDD": float(proba_x[1]),
            "tabm_pred_confidence": float(proba_x[1] if pred_x == 1 else 1.0 - proba_x[1]),
            "tabm_correct": bool(pred_x == int(qrow["true_label"])),
            "n_local_perturbations": int(args.n_local_perturbations),
            "perturbation_mode": args.perturbation_mode,
            "perturbation_scale": float(args.perturbation_scale),
            "output_space_for_ros": "log_probability_surrogate",
            "any_prediction_changed_under_perturbation": any_changed,
            "prediction_change_rate": float(np.mean(pred_changed_flags)),
        }
        if "ris" in metrics_requested:
            row["local_ris_instability"] = float(np.nanmax(ris_arr)) if np.isfinite(ris_arr).any() else np.nan
            row["local_ris_nan_rate"] = float(np.mean(~np.isfinite(ris_arr)))
            row["local_ris_mean_finite"] = float(np.nanmean(ris_arr)) if np.isfinite(ris_arr).any() else np.nan
        if "ros" in metrics_requested:
            row["local_ros_instability"] = float(np.nanmax(ros_arr)) if np.isfinite(ros_arr).any() else np.nan
            row["local_ros_nan_rate"] = float(np.mean(~np.isfinite(ros_arr)))
            row["local_ros_mean_finite"] = float(np.nanmean(ros_arr)) if np.isfinite(ros_arr).any() else np.nan
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / f"local_instability_{split_name}_progress.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    pd.DataFrame(pert_rows_all).to_csv(out_dir / f"local_instability_{split_name}_perturbation_details.csv", index=False)
    return out


def build_instability_flagging_grid(
    df: pd.DataFrame,
    instability_col: str,
    percentiles: list[float],
    nan_as_high: bool,
) -> pd.DataFrame:
    scores_eff = effective_instability_scores(df[instability_col].astype(float).to_numpy(), nan_as_high=nan_as_high)
    finite = scores_eff[np.isfinite(scores_eff)]
    rows = []
    baseline_error_rate = 1.0 - float(df["tabm_correct"].mean())
    for pct in percentiles:
        if finite.size == 0:
            threshold = np.nan
            flag = np.zeros(len(df), dtype=bool)
        else:
            threshold = float(np.percentile(finite, pct))
            flag = scores_eff >= threshold
        n_flag = int(flag.sum())
        flagged_errors = int(((~df["tabm_correct"].astype(bool)) & flag).sum())
        flagged_correct = int((df["tabm_correct"].astype(bool) & flag).sum())
        n_kept = int((~flag).sum())
        kept_acc = float(df.loc[~flag, "tabm_correct"].mean()) if n_kept else np.nan
        flag_precision = float(flagged_errors / n_flag) if n_flag else np.nan
        error_recall = float(flagged_errors / int((~df["tabm_correct"].astype(bool)).sum())) if int((~df["tabm_correct"].astype(bool)).sum()) else np.nan
        rows.append(
            {
                "instability_col": instability_col,
                "percentile": float(pct),
                "threshold": threshold,
                "n_flagged": n_flag,
                "n_kept": n_kept,
                "coverage_kept": float(n_kept / len(df)) if len(df) else np.nan,
                "baseline_error_rate": baseline_error_rate,
                "flagged_tabm_errors": flagged_errors,
                "flagged_tabm_correct": flagged_correct,
                "flag_precision_for_errors": flag_precision,
                "error_enrichment_over_baseline": float(flag_precision / baseline_error_rate) if baseline_error_rate > 0 and np.isfinite(flag_precision) else np.nan,
                "error_detection_recall": error_recall,
                "kept_accuracy": kept_acc,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Direction B: fixed-pMDD SHAP-space similarity retrieval.
# ---------------------------------------------------------------------------


def shap_distance(query_vec: np.ndarray, ref_vecs: np.ndarray, metric: str) -> np.ndarray:
    q = np.asarray(query_vec, dtype=float).reshape(1, -1)
    R = np.asarray(ref_vecs, dtype=float)
    if metric == "euclidean":
        return np.linalg.norm(R - q, axis=1)
    if metric == "cosine":
        qn = np.linalg.norm(q, axis=1)[0]
        rn = np.linalg.norm(R, axis=1)
        denom = np.where((qn * rn) <= EPS, 1.0, qn * rn)
        cos = (R @ q.reshape(-1)) / denom
        cos = np.clip(cos, -1.0, 1.0)
        return 1.0 - cos
    if metric == "pearson":
        q0 = q.reshape(-1) - float(np.mean(q))
        R0 = R - R.mean(axis=1, keepdims=True)
        qn = np.linalg.norm(q0)
        rn = np.linalg.norm(R0, axis=1)
        denom = np.where((qn * rn) <= EPS, 1.0, qn * rn)
        corr = (R0 @ q0.reshape(-1)) / denom
        corr = np.clip(corr, -1.0, 1.0)
        return 1.0 - corr
    raise ValueError(f"Unknown SHAP similarity metric: {metric}")


def compute_tau_generic(df_all: pd.DataFrame, df_topk: pd.DataFrame, score_col: str, tau_mode: str, tau_value: float | None) -> float:
    if tau_mode == "fixed":
        if tau_value is None or tau_value <= 0:
            raise ValueError("--tau-value must be positive when --tau-mode fixed.")
        return float(tau_value)
    if tau_mode == "none":
        return 1.0
    if tau_value is not None:
        raise ValueError("--tau-value should only be used with --tau-mode fixed.")
    if tau_mode == "topk_median":
        tau = float(np.median(df_topk[score_col].astype(float).to_numpy()))
    elif tau_mode == "allref_median":
        tau = float(np.median(df_all[score_col].astype(float).to_numpy()))
    elif tau_mode == "topk_mean":
        tau = float(np.mean(df_topk[score_col].astype(float).to_numpy()))
    elif tau_mode == "allref_mean":
        tau = float(np.mean(df_all[score_col].astype(float).to_numpy()))
    else:
        raise ValueError(f"Unknown tau_mode: {tau_mode}")
    if not np.isfinite(tau) or tau <= 0:
        positive = df_all.loc[df_all[score_col].astype(float) > 0, score_col].astype(float).to_numpy()
        tau = float(np.median(positive)) if positive.size > 0 else 1.0
    return max(tau, EPS)


def compute_dirichlet_from_distance(
    sim_df: pd.DataFrame,
    query_pred_label: int,
    top_k: int,
    alpha_prior: float,
    beta: float,
    tau_mode: str,
    tau_value: float | None,
) -> tuple[pd.DataFrame, dict]:
    df_all = sim_df.sort_values(["similarity_distance", "reference_subject"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    if len(df_all) < top_k:
        raise ValueError(f"Not enough reference samples: available={len(df_all)}, top_k={top_k}")
    df_all["similarity_rank"] = np.arange(1, len(df_all) + 1, dtype=int)
    df_all["is_top_k"] = df_all["similarity_rank"] <= int(top_k)
    df_topk = df_all.head(top_k).copy().reset_index(drop=True)
    tau = compute_tau_generic(df_all, df_topk, "similarity_distance", tau_mode, tau_value)
    df_topk["reference_label_used"] = df_topk["reference_true_label"].astype(int)
    df_topk["reference_class_name"] = df_topk["reference_label_used"].map(CLASS_NAMES)
    df_topk["support_or_oppose"] = np.where(df_topk["reference_label_used"] == int(query_pred_label), "support", "oppose")
    df_topk["similarity_tau"] = tau
    df_topk["similarity_scaled_distance"] = df_topk["similarity_distance"].astype(float) / (tau + EPS)
    df_topk["similarity_evidence"] = np.exp(-float(beta) * df_topk["similarity_scaled_distance"].astype(float))

    evidence_hc = float(df_topk.loc[df_topk["reference_label_used"] == 0, "similarity_evidence"].sum())
    evidence_mdd = float(df_topk.loc[df_topk["reference_label_used"] == 1, "similarity_evidence"].sum())
    alpha_hc = float(alpha_prior + evidence_hc)
    alpha_mdd = float(alpha_prior + evidence_mdd)
    total_alpha = float(alpha_hc + alpha_mdd)
    weight_hc = float(alpha_hc / total_alpha)
    weight_mdd = float(alpha_mdd / total_alpha)
    if int(query_pred_label) == 1:
        support_class, oppose_class = "MDD", "HC"
        support_weight, oppose_weight = weight_mdd, weight_hc
    else:
        support_class, oppose_class = "HC", "MDD"
        support_weight, oppose_weight = weight_hc, weight_mdd
    probs = np.clip(np.asarray([weight_hc, weight_mdd]), EPS, 1.0)
    entropy = float(-np.sum(probs * np.log(probs)))
    summary = {
        "tau": tau,
        "evidence_HC": evidence_hc,
        "evidence_MDD": evidence_mdd,
        "alpha_HC": alpha_hc,
        "alpha_MDD": alpha_mdd,
        "total_alpha": total_alpha,
        "dirichlet_weight_HC": weight_hc,
        "dirichlet_weight_MDD": weight_mdd,
        "support_class": support_class,
        "oppose_class": oppose_class,
        "support_weight": support_weight,
        "oppose_weight": oppose_weight,
        "support_count": int((df_topk["support_or_oppose"] == "support").sum()),
        "oppose_count": int((df_topk["support_or_oppose"] == "oppose").sum()),
        "predictive_entropy": entropy,
        "normalized_entropy": float(entropy / math.log(2.0)),
        "evidence_uncertainty_2_over_total_alpha": float(2.0 / total_alpha),
    }
    return df_topk, summary


def evaluate_split_shap_similarity(
    split_name: str,
    query_shap_df: pd.DataFrame,
    ref_shap_df: pd.DataFrame,
    subjects: np.ndarray,
    feature_cols: list[str],
    metric_name: str,
    output_root: Path,
    reference_subjects: set[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    out_dir = output_root / "directionB_shap_similarity" / f"metric_{metric_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"online_{split_name}_diagnostics.csv"
    if args.reuse_direction_b and out_path.exists():
        return pd.read_csv(out_path)

    per_query_dir = out_dir / "per_query_similarity" / split_name
    if args.save_per_query_similarity or args.reuse_per_query_similarity:
        per_query_dir.mkdir(parents=True, exist_ok=True)

    ref_vecs = ref_shap_df[feature_cols].astype(float).to_numpy()
    rows_out = []
    print(f"\nDirection B: SHAP-space Top-K metric={metric_name}, split={split_name}, n={len(query_shap_df)}")
    for q_pos, (_, qrow) in enumerate(query_shap_df.iterrows(), start=1):
        query_subject = str(qrow[ID_COL])
        safe_query = subject_to_safe_name(query_subject)
        cache_path = per_query_dir / f"{safe_query}_{metric_name}_similarity_all_references.csv"
        print(f"  [{q_pos}/{len(query_shap_df)}] {split_name} {metric_name} query {query_subject}")
        query_vec = qrow[feature_cols].astype(float).to_numpy()
        query_true_label = int(qrow["true_label"])
        query_pred_label = int(qrow["pred_label"])
        query_p_mdd = float(qrow["p_MDD"])
        query_pred_confidence = float(qrow["pred_confidence"])

        if args.reuse_per_query_similarity and cache_path.exists():
            sim_df = pd.read_csv(cache_path)
        else:
            distances = shap_distance(query_vec, ref_vecs, metric_name)
            sim_rows = []
            for r_pos, (_, refrow) in enumerate(ref_shap_df.iterrows()):
                ref_subject = str(refrow[ID_COL])
                if (not args.include_self_reference) and ref_subject == query_subject:
                    continue
                sim_rows.append(
                    {
                        "query_subject": query_subject,
                        "query_true_label": query_true_label,
                        "query_pred_label": query_pred_label,
                        "query_p_MDD": query_p_mdd,
                        "query_pred_confidence": query_pred_confidence,
                        "shap_target": "p_MDD_fixed",
                        "shap_values": "signed_raw_no_normalization",
                        "similarity_metric": metric_name,
                        "reference_subject": ref_subject,
                        "reference_true_label": int(refrow["true_label"]),
                        "reference_pred_label": int(refrow["pred_label"]),
                        "reference_p_MDD": float(refrow["p_MDD"]),
                        "same_pred_label": query_pred_label == int(refrow["pred_label"]),
                        "query_in_reference_bank": query_subject in reference_subjects,
                        "self_reference_excluded": (query_subject in reference_subjects) and (not args.include_self_reference),
                        "similarity_distance": float(distances[r_pos]),
                    }
                )
            sim_df = pd.DataFrame(sim_rows)
            if args.save_per_query_similarity:
                sim_df.sort_values(["similarity_distance", "reference_subject"]).to_csv(cache_path, index=False)

        topk_df, dsum = compute_dirichlet_from_distance(
            sim_df=sim_df,
            query_pred_label=query_pred_label,
            top_k=args.top_k,
            alpha_prior=args.alpha_prior,
            beta=args.beta,
            tau_mode=args.tau_mode,
            tau_value=args.tau_value,
        )
        rows_out.append(
            {
                "split": split_name,
                "direction": "B_shap_similarity_topk_dirichlet",
                "shap_target": "p_MDD_fixed",
                "shap_values": "signed_raw_no_normalization",
                "similarity_metric": metric_name,
                "subject": query_subject,
                "true_label": query_true_label,
                "true_class": CLASS_NAMES[query_true_label],
                "tabm_pred_label": query_pred_label,
                "tabm_pred_class": CLASS_NAMES[query_pred_label],
                "tabm_p_MDD": query_p_mdd,
                "tabm_pred_confidence": query_pred_confidence,
                "tabm_correct": query_pred_label == query_true_label,
                "top_k": int(args.top_k),
                "alpha_prior": float(args.alpha_prior),
                "beta": float(args.beta),
                "tau_mode": args.tau_mode,
                "tau": float(dsum["tau"]),
                "evidence_HC": float(dsum["evidence_HC"]),
                "evidence_MDD": float(dsum["evidence_MDD"]),
                "alpha_HC": float(dsum["alpha_HC"]),
                "alpha_MDD": float(dsum["alpha_MDD"]),
                "total_alpha": float(dsum["total_alpha"]),
                "dirichlet_weight_HC": float(dsum["dirichlet_weight_HC"]),
                "dirichlet_weight_MDD": float(dsum["dirichlet_weight_MDD"]),
                "support_class": dsum["support_class"],
                "oppose_class": dsum["oppose_class"],
                "support_weight": float(dsum["support_weight"]),
                "oppose_weight": float(dsum["oppose_weight"]),
                "support_count": int(dsum["support_count"]),
                "oppose_count": int(dsum["oppose_count"]),
                "predictive_entropy": float(dsum["predictive_entropy"]),
                "normalized_entropy": float(dsum["normalized_entropy"]),
                "evidence_uncertainty_2_over_total_alpha": float(dsum["evidence_uncertainty_2_over_total_alpha"]),
                "min_topk_distance": float(topk_df["similarity_distance"].astype(float).min()),
                "median_topk_distance": float(topk_df["similarity_distance"].astype(float).median()),
                "mean_topk_distance": float(topk_df["similarity_distance"].astype(float).mean()),
                "topk_reference_subjects": ";".join(topk_df["reference_subject"].astype(str).tolist()),
                "topk_reference_labels": ";".join(topk_df["reference_label_used"].astype(str).tolist()),
                "topk_distances": ";".join([f"{v:.10g}" for v in topk_df["similarity_distance"].astype(float).tolist()]),
                "query_in_reference_bank": bool(query_subject in reference_subjects),
                "self_reference_excluded": bool((query_subject in reference_subjects) and (not args.include_self_reference)),
            }
        )
        pd.DataFrame(rows_out).to_csv(out_dir / f"online_{split_name}_diagnostics_progress.csv", index=False)
    out = pd.DataFrame(rows_out)
    out.to_csv(out_path, index=False)
    return out


def evaluate_hard_flip_grid(df: pd.DataFrame, thresholds: list[float], min_counts: list[int], min_flags: int) -> pd.DataFrame:
    rows = []
    y_true = df["true_label"].astype(int).to_numpy()
    base_pred = df["tabm_pred_label"].astype(int).to_numpy()
    p_tabm = df["tabm_p_MDD"].astype(float).to_numpy()
    base_metrics = compute_metrics(y_true, base_pred, p_tabm)
    for th in thresholds:
        for mc in min_counts:
            flag = (df["oppose_weight"].astype(float).to_numpy() >= float(th)) & (df["oppose_count"].astype(int).to_numpy() >= int(mc))
            n_flag = int(flag.sum())
            pred = base_pred.copy()
            pred[flag] = 1 - pred[flag]
            # Score after hard flip is for metrics needing ranking; use hard predictions as score fallback.
            metrics = compute_metrics(y_true, pred, pred.astype(float))
            flagged_errors = int(((base_pred != y_true) & flag).sum())
            flagged_correct = int(((base_pred == y_true) & flag).sum())
            rows.append(
                {
                    "oppose_weight_threshold": float(th),
                    "min_oppose_count": int(mc),
                    "n_flagged": n_flag,
                    "flagged_tabm_errors": flagged_errors,
                    "flagged_tabm_correct": flagged_correct,
                    "flag_precision_for_errors": float(flagged_errors / n_flag) if n_flag else np.nan,
                    "net_correct_minus_broken_if_flipped": int(flagged_errors - flagged_correct),
                    "baseline_accuracy": float(base_metrics["accuracy"]),
                    "flip_accuracy": float(metrics["accuracy"]),
                    "accuracy_delta_if_flipped": float(metrics["accuracy"] - base_metrics["accuracy"]),
                    "baseline_mcc": float(base_metrics["mcc"]),
                    "flip_mcc": float(metrics["mcc"]),
                    "mcc_delta_if_flipped": float(metrics["mcc"] - base_metrics["mcc"]),
                    "eligible_for_selection": bool(n_flag >= min_flags),
                }
            )
    return pd.DataFrame(rows)


def select_hard_flip_rule(grid: pd.DataFrame, objective: str, baseline_value: float) -> dict:
    metric_col = "flip_accuracy" if objective == "accuracy" else {
        "balanced_accuracy": "flip_balanced_accuracy",
        "macro_f1": "flip_macro_f1",
        "mcc": "flip_mcc",
    }.get(objective, "flip_accuracy")
    # Our hard flip grid currently stores accuracy and mcc. For other objectives, use accuracy conservatively.
    if metric_col not in grid.columns:
        metric_col = "flip_accuracy"
    eligible = grid.loc[grid["eligible_for_selection"].astype(bool)].copy()
    if eligible.empty:
        return {"selected_policy": "no_flip", "reason": "No candidate reached min_flags_for_rule."}
    eligible = eligible.sort_values([metric_col, "n_flagged"], ascending=[False, False])
    best = eligible.iloc[0].to_dict()
    if float(best[metric_col]) <= float(baseline_value) + 1e-12:
        return {"selected_policy": "no_flip", "reason": "No candidate improved validation objective over offline ResNet."}
    best["selected_policy"] = "hard_flip"
    best["reason"] = "Best validation hard-flip rule improved objective."
    return best


def apply_hard_flip_rule(df: pd.DataFrame, selected: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    if selected.get("selected_policy") != "hard_flip":
        out["final_pred_label"] = out["tabm_pred_label"].astype(int)
        out["final_p_MDD"] = out["tabm_p_MDD"].astype(float)
        out["rule_flagged"] = False
    else:
        th = float(selected["oppose_weight_threshold"])
        mc = int(selected["min_oppose_count"])
        flag = (out["oppose_weight"].astype(float) >= th) & (out["oppose_count"].astype(int) >= mc)
        out["rule_flagged"] = flag
        out["final_pred_label"] = out["tabm_pred_label"].astype(int)
        out.loc[flag, "final_pred_label"] = 1 - out.loc[flag, "tabm_pred_label"].astype(int)
        out["final_p_MDD"] = out["tabm_p_MDD"].astype(float)
        out.loc[flag, "final_p_MDD"] = out.loc[flag, "final_pred_label"].astype(float)
    metrics = compute_metrics(
        out["true_label"].astype(int).to_numpy(),
        out["final_pred_label"].astype(int).to_numpy(),
        out["final_p_MDD"].astype(float).to_numpy(),
    )
    return out, pd.DataFrame([{**{"stage": "hard_flip_selected_rule", "n_flagged": int(out["rule_flagged"].sum())}, **metrics}])


# ---------------------------------------------------------------------------
# Direction C: local instability gate + SHAP Top-K Dirichlet.
# ---------------------------------------------------------------------------


def attach_instability(df_b: pd.DataFrame, df_a: pd.DataFrame, instability_col: str, nan_as_high: bool) -> pd.DataFrame:
    cols = ["subject", instability_col, "any_prediction_changed_under_perturbation", "prediction_change_rate"]
    merged = df_b.merge(df_a[cols], on="subject", how="left")
    merged[f"{instability_col}_effective"] = effective_instability_scores(
        merged[instability_col].astype(float).to_numpy(), nan_as_high=nan_as_high
    )
    return merged


def direction_c_hard_flip_grid(
    df: pd.DataFrame,
    instability_col_eff: str,
    instability_thresholds: list[tuple[float, float]],
    oppose_thresholds: list[float],
    min_counts: list[int],
    min_flags: int,
) -> pd.DataFrame:
    rows = []
    y_true = df["true_label"].astype(int).to_numpy()
    base_pred = df["tabm_pred_label"].astype(int).to_numpy()
    p_tabm = df["tabm_p_MDD"].astype(float).to_numpy()
    base_metrics = compute_metrics(y_true, base_pred, p_tabm)
    instability = df[instability_col_eff].astype(float).to_numpy()
    for pct, inst_th in instability_thresholds:
        unstable = instability >= inst_th
        for ow_th in oppose_thresholds:
            for mc in min_counts:
                flag = unstable & (df["oppose_weight"].astype(float).to_numpy() >= float(ow_th)) & (df["oppose_count"].astype(int).to_numpy() >= int(mc))
                n_flag = int(flag.sum())
                pred = base_pred.copy()
                pred[flag] = 1 - pred[flag]
                metrics = compute_metrics(y_true, pred, pred.astype(float))
                flagged_errors = int(((base_pred != y_true) & flag).sum())
                flagged_correct = int(((base_pred == y_true) & flag).sum())
                rows.append(
                    {
                        "instability_percentile": float(pct),
                        "instability_threshold": float(inst_th),
                        "oppose_weight_threshold": float(ow_th),
                        "min_oppose_count": int(mc),
                        "n_flagged": n_flag,
                        "flagged_tabm_errors": flagged_errors,
                        "flagged_tabm_correct": flagged_correct,
                        "flag_precision_for_errors": float(flagged_errors / n_flag) if n_flag else np.nan,
                        "net_correct_minus_broken_if_flipped": int(flagged_errors - flagged_correct),
                        "baseline_accuracy": float(base_metrics["accuracy"]),
                        "flip_accuracy": float(metrics["accuracy"]),
                        "accuracy_delta_if_flipped": float(metrics["accuracy"] - base_metrics["accuracy"]),
                        "baseline_mcc": float(base_metrics["mcc"]),
                        "flip_mcc": float(metrics["mcc"]),
                        "mcc_delta_if_flipped": float(metrics["mcc"] - base_metrics["mcc"]),
                        "eligible_for_selection": bool(n_flag >= min_flags),
                    }
                )
    return pd.DataFrame(rows)


def direction_c_soft_fusion_grid(
    df: pd.DataFrame,
    instability_col_eff: str,
    instability_thresholds: list[tuple[float, float]],
    lambdas: list[float],
) -> pd.DataFrame:
    rows = []
    y_true = df["true_label"].astype(int).to_numpy()
    p_tabm = df["tabm_p_MDD"].astype(float).to_numpy()
    w_dir = df["dirichlet_weight_MDD"].astype(float).to_numpy()
    instability = df[instability_col_eff].astype(float).to_numpy()
    base_metrics = compute_metrics(y_true, (p_tabm >= 0.5).astype(int), p_tabm)
    for pct, inst_th in instability_thresholds:
        unstable = instability >= inst_th
        for lam in lambdas:
            p_final = p_tabm.copy()
            p_final[unstable] = float(lam) * p_tabm[unstable] + (1.0 - float(lam)) * w_dir[unstable]
            y_pred = (p_final >= 0.5).astype(int)
            metrics = compute_metrics(y_true, y_pred, p_final)
            rows.append(
                {
                    "instability_percentile": float(pct),
                    "instability_threshold": float(inst_th),
                    "lambda": float(lam),
                    "n_unstable_fused": int(unstable.sum()),
                    "baseline_accuracy": float(base_metrics["accuracy"]),
                    "accuracy_delta": float(metrics["accuracy"] - base_metrics["accuracy"]),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def select_best_grid_row(grid: pd.DataFrame, objective: str, baseline_value: float, metric_prefix: str | None = None) -> dict:
    if metric_prefix:
        metric_col = f"{metric_prefix}_{objective}"
    elif objective in grid.columns:
        metric_col = objective
    elif objective == "mcc" and "flip_mcc" in grid.columns:
        metric_col = "flip_mcc"
    elif objective == "accuracy" and "flip_accuracy" in grid.columns:
        metric_col = "flip_accuracy"
    elif "accuracy" in grid.columns:
        metric_col = "accuracy"
    elif "flip_accuracy" in grid.columns:
        metric_col = "flip_accuracy"
    else:
        raise ValueError(f"Cannot find objective column for {objective}. Available columns: {list(grid.columns)}")

    if "eligible_for_selection" in grid.columns:
        candidates = grid.loc[grid["eligible_for_selection"].astype(bool)].copy()
    else:
        candidates = grid.copy()
    if candidates.empty:
        return {"selected_policy": "no_change", "reason": "No eligible candidate."}
    candidates = candidates.sort_values([metric_col], ascending=[False])
    best = candidates.iloc[0].to_dict()
    if float(best[metric_col]) <= float(baseline_value) + 1e-12:
        best["selected_policy"] = "no_change"
        best["reason"] = "No candidate improved validation objective over offline ResNet."
    else:
        best["selected_policy"] = "selected"
        best["reason"] = "Best validation candidate improved objective."
    best["selection_metric_col"] = metric_col
    return best


def apply_direction_c_soft_fusion(df: pd.DataFrame, instability_col_eff: str, selected: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    if selected.get("selected_policy") != "selected":
        out["final_p_MDD"] = out["tabm_p_MDD"].astype(float)
        out["final_pred_label"] = out["tabm_pred_label"].astype(int)
        out["unstable_fused"] = False
    else:
        lam = float(selected["lambda"])
        inst_th = float(selected["instability_threshold"])
        unstable = out[instability_col_eff].astype(float) >= inst_th
        out["unstable_fused"] = unstable
        out["final_p_MDD"] = out["tabm_p_MDD"].astype(float)
        out.loc[unstable, "final_p_MDD"] = lam * out.loc[unstable, "tabm_p_MDD"].astype(float) + (1.0 - lam) * out.loc[unstable, "dirichlet_weight_MDD"].astype(float)
        out["final_pred_label"] = (out["final_p_MDD"].astype(float) >= 0.5).astype(int)
    metrics = compute_metrics(out["true_label"].astype(int).to_numpy(), out["final_pred_label"].astype(int).to_numpy(), out["final_p_MDD"].astype(float).to_numpy())
    return out, pd.DataFrame([{**{"stage": "directionC_gated_soft_fusion", "n_unstable_fused": int(out["unstable_fused"].sum())}, **metrics}])


def apply_direction_c_hard_flip(df: pd.DataFrame, instability_col_eff: str, selected: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    if selected.get("selected_policy") != "selected":
        out["final_pred_label"] = out["tabm_pred_label"].astype(int)
        out["final_p_MDD"] = out["tabm_p_MDD"].astype(float)
        out["rule_flagged"] = False
    else:
        inst_th = float(selected["instability_threshold"])
        ow_th = float(selected["oppose_weight_threshold"])
        mc = int(selected["min_oppose_count"])
        flag = (out[instability_col_eff].astype(float) >= inst_th) & (out["oppose_weight"].astype(float) >= ow_th) & (out["oppose_count"].astype(int) >= mc)
        out["rule_flagged"] = flag
        out["final_pred_label"] = out["tabm_pred_label"].astype(int)
        out.loc[flag, "final_pred_label"] = 1 - out.loc[flag, "tabm_pred_label"].astype(int)
        out["final_p_MDD"] = out["tabm_p_MDD"].astype(float)
        out.loc[flag, "final_p_MDD"] = out.loc[flag, "final_pred_label"].astype(float)
    metrics = compute_metrics(out["true_label"].astype(int).to_numpy(), out["final_pred_label"].astype(int).to_numpy(), out["final_p_MDD"].astype(float).to_numpy())
    return out, pd.DataFrame([{**{"stage": "directionC_gated_hard_flip", "n_flagged": int(out["rule_flagged"].sum())}, **metrics}])


# ---------------------------------------------------------------------------
# Main orchestration.
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    start = pd.Timestamp.utcnow()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    patch_sklearn_for_pretab()

    features_csv = Path(args.features_csv)
    labels_csv = Path(args.labels_csv)
    splits_csv = Path(args.splits_csv)
    ckpt_path = Path(args.ckpt_path)
    selected_refs_csv = Path(args.selected_references_csv)

    similarity_metrics = parse_str_list(args.similarity_metrics)
    if not set(similarity_metrics).issubset({"cosine", "euclidean", "pearson"}):
        raise ValueError("--similarity-metrics must contain only cosine, euclidean, pearson.")
    oppose_thresholds = parse_float_list(args.oppose_weight_thresholds)
    min_counts = parse_int_list(args.min_oppose_counts)
    lambdas = parse_float_list(args.lambda_grid)
    instability_percentiles = parse_float_list(args.instability_percentiles)

    X, y, subjects, feature_cols = load_table(features_csv, labels_csv)
    train_idx, val_idx, test_idx, split_df = load_split_indices(splits_csv, subjects, y)
    if args.max_validation_samples is not None:
        val_idx = val_idx[: int(args.max_validation_samples)]
    if args.max_test_samples is not None:
        test_idx = test_idx[: int(args.max_test_samples)]
    selected_ref_df = load_selected_references(selected_refs_csv, subjects, y)
    ref_idx = selected_ref_df["index"].astype(int).to_numpy()
    ref_subjects = set(selected_ref_df[ID_COL].astype(str).tolist())

    print("Loading ResNet checkpoint...")
    model = load_resnet(ckpt_path, X.iloc[train_idx][feature_cols], y[train_idx], X.iloc[val_idx][feature_cols], y[val_idx])
    predict_function = make_predict_function(model, feature_cols)
    shap_explainer = build_fixed_pmdd_shap_explainer(predict_function, X, feature_cols, train_idx)

    # Offline predictions are saved for reference.
    offline_val = pd.DataFrame(
        {
            "subject": subjects[val_idx].astype(str),
            "true_label": y[val_idx].astype(int),
            "p_MDD": predict_function(X.iloc[val_idx][feature_cols])[:, 1],
        }
    )
    offline_val["pred_label"] = (offline_val["p_MDD"].astype(float) >= 0.5).astype(int)
    offline_test = pd.DataFrame(
        {
            "subject": subjects[test_idx].astype(str),
            "true_label": y[test_idx].astype(int),
            "p_MDD": predict_function(X.iloc[test_idx][feature_cols])[:, 1],
        }
    )
    offline_test["pred_label"] = (offline_test["p_MDD"].astype(float) >= 0.5).astype(int)
    offline_val_metrics = compute_metrics(offline_val["true_label"].to_numpy(), offline_val["pred_label"].to_numpy(), offline_val["p_MDD"].to_numpy())
    offline_test_metrics = compute_metrics(offline_test["true_label"].to_numpy(), offline_test["pred_label"].to_numpy(), offline_test["p_MDD"].to_numpy())
    pd.DataFrame([{**{"stage": "offline_validation_tabm"}, **offline_val_metrics}, {**{"stage": "offline_test_tabm"}, **offline_test_metrics}]).to_csv(output_root / "offline_tabm_metrics.csv", index=False)

    # Fixed-pMDD SHAP banks.
    ref_shap_df = compute_or_load_shap_bank(
        "reference20_fixed_pmdd", ref_idx, X, y, subjects, feature_cols, predict_function, shap_explainer, output_root, args.reuse_shap_cache
    )
    val_shap_df = compute_or_load_shap_bank(
        "validation_fixed_pmdd", val_idx, X, y, subjects, feature_cols, predict_function, shap_explainer, output_root, args.reuse_shap_cache
    )
    test_shap_df = compute_or_load_shap_bank(
        "test_fixed_pmdd", test_idx, X, y, subjects, feature_cols, predict_function, shap_explainer, output_root, args.reuse_shap_cache
    )

    # Direction A local instability.
    train_std = X.iloc[train_idx][feature_cols].astype(float).std(axis=0, ddof=0).to_numpy()
    train_std = np.where(np.isfinite(train_std) & (train_std > EPS), train_std, 1.0)
    val_inst = compute_or_load_local_instability("validation", val_shap_df, X, feature_cols, predict_function, shap_explainer, output_root, args, train_std)
    test_inst = compute_or_load_local_instability("test", test_shap_df, X, feature_cols, predict_function, shap_explainer, output_root, args, train_std)

    dirA_rows = []
    for inst_col in [c for c in ["local_ris_instability", "local_ros_instability"] if c in val_inst.columns]:
        grid = build_instability_flagging_grid(val_inst, inst_col, instability_percentiles, args.nan_as_high_instability)
        grid.to_csv(output_root / "directionA_local_instability" / f"validation_flagging_grid_{inst_col}.csv", index=False)
        # Apply the best enrichment threshold as a diagnostic to test, not as a selected performance rule.
        if not grid.empty:
            best = grid.sort_values(["error_enrichment_over_baseline", "n_flagged"], ascending=[False, False]).iloc[0].to_dict()
            dirA_rows.append({"direction": "A_local_instability", **best})
            threshold = float(best["threshold"])
            scores_test = effective_instability_scores(test_inst[inst_col].astype(float).to_numpy(), args.nan_as_high_instability)
            flag = scores_test >= threshold
            test_flagged_errors = int(((~test_inst["tabm_correct"].astype(bool)) & flag).sum())
            test_n_flag = int(flag.sum())
            test_flag_precision = float(test_flagged_errors / test_n_flag) if test_n_flag else np.nan
            dirA_rows[-1].update({
                "test_n_flagged_at_validation_threshold": test_n_flag,
                "test_flagged_errors_at_validation_threshold": test_flagged_errors,
                "test_flag_precision_for_errors_at_validation_threshold": test_flag_precision,
                "test_baseline_error_rate": float(1.0 - test_inst["tabm_correct"].mean()),
            })
    pd.DataFrame(dirA_rows).to_csv(output_root / "directionA_summary.csv", index=False)

    # Direction B and C.
    summary_rows = []
    all_test_metric_rows = []
    local_metric_cols = [c for c in ["local_ris_instability", "local_ros_instability"] if c in val_inst.columns]
    for sim_metric in similarity_metrics:
        val_b = evaluate_split_shap_similarity("validation", val_shap_df, ref_shap_df, subjects, feature_cols, sim_metric, output_root, ref_subjects, args)
        test_b = evaluate_split_shap_similarity("test", test_shap_df, ref_shap_df, subjects, feature_cols, sim_metric, output_root, ref_subjects, args)
        b_dir = output_root / "directionB_shap_similarity" / f"metric_{sim_metric}"

        # Direction B hard flip.
        hard_grid = evaluate_hard_flip_grid(val_b, oppose_thresholds, min_counts, args.min_flags_for_rule)
        hard_grid.to_csv(b_dir / "validation_hard_flip_grid.csv", index=False)
        selected_hard = select_hard_flip_rule(hard_grid, args.fusion_objective, offline_val_metrics[args.fusion_objective])
        pd.DataFrame([selected_hard]).to_csv(b_dir / "selected_validation_hard_flip_rule.csv", index=False)
        _, test_hard_metrics = apply_hard_flip_rule(test_b, selected_hard)
        test_hard_metrics.insert(0, "direction", "B_shap_similarity")
        test_hard_metrics.insert(1, "similarity_metric", sim_metric)
        test_hard_metrics.to_csv(b_dir / "test_hard_flip_selected_rule_metrics.csv", index=False)

        # Direction B soft fusion.
        lambda_grid = evaluate_soft_fusion(val_b, lambdas)
        lambda_grid.to_csv(b_dir / "validation_soft_fusion_lambda_grid.csv", index=False)
        selected_lambda = select_lambda(lambda_grid, args.fusion_objective)
        pd.DataFrame([selected_lambda]).to_csv(b_dir / "selected_validation_lambda.csv", index=False)
        _, test_soft_metrics = apply_lambda_to_split(test_b, float(selected_lambda["lambda"]))
        test_soft_metrics.insert(0, "direction", "B_shap_similarity")
        test_soft_metrics.insert(1, "similarity_metric", sim_metric)
        test_soft_metrics.to_csv(b_dir / "test_soft_fusion_metrics.csv", index=False)

        all_test_metric_rows.extend(test_hard_metrics.to_dict("records"))
        all_test_metric_rows.extend(test_soft_metrics.to_dict("records"))
        summary_rows.append(
            {
                "direction": "B_shap_similarity",
                "similarity_metric": sim_metric,
                "validation_baseline_accuracy": offline_val_metrics["accuracy"],
                "test_baseline_accuracy": offline_test_metrics["accuracy"],
                "selected_hard_policy": selected_hard.get("selected_policy"),
                "selected_hard_reason": selected_hard.get("reason"),
                "test_hard_flip_accuracy": float(test_hard_metrics["accuracy"].iloc[0]),
                "test_hard_flip_mcc": float(test_hard_metrics["mcc"].iloc[0]),
                "selected_lambda": float(selected_lambda["lambda"]),
                "validation_soft_fusion_accuracy": float(selected_lambda.get("accuracy", np.nan)),
                "test_soft_fusion_accuracy": float(test_soft_metrics["accuracy"].iloc[0]),
                "test_soft_fusion_mcc": float(test_soft_metrics["mcc"].iloc[0]),
            }
        )

        # Direction C: gated by each local instability metric.
        for inst_col in local_metric_cols:
            c_dir = output_root / "directionC_gated_hybrid" / f"gate_{inst_col}" / f"metric_{sim_metric}"
            c_dir.mkdir(parents=True, exist_ok=True)
            val_c = attach_instability(val_b, val_inst, inst_col, args.nan_as_high_instability)
            test_c = attach_instability(test_b, test_inst, inst_col, args.nan_as_high_instability)
            inst_eff_col = f"{inst_col}_effective"
            finite_val = val_c[inst_eff_col].astype(float).to_numpy()
            finite_val = finite_val[np.isfinite(finite_val)]
            if finite_val.size == 0:
                continue
            inst_thresholds = [(float(p), float(np.percentile(finite_val, p))) for p in instability_percentiles]

            c_hard_grid = direction_c_hard_flip_grid(val_c, inst_eff_col, inst_thresholds, oppose_thresholds, min_counts, args.min_flags_for_rule)
            c_hard_grid.to_csv(c_dir / "validation_gated_hard_flip_grid.csv", index=False)
            selected_c_hard = select_best_grid_row(c_hard_grid, args.fusion_objective, offline_val_metrics[args.fusion_objective])
            pd.DataFrame([selected_c_hard]).to_csv(c_dir / "selected_validation_gated_hard_flip_rule.csv", index=False)
            _, test_c_hard_metrics = apply_direction_c_hard_flip(test_c, inst_eff_col, selected_c_hard)
            test_c_hard_metrics.insert(0, "direction", "C_gated_hybrid")
            test_c_hard_metrics.insert(1, "similarity_metric", sim_metric)
            test_c_hard_metrics.insert(2, "instability_gate", inst_col)
            test_c_hard_metrics.to_csv(c_dir / "test_gated_hard_flip_metrics.csv", index=False)

            c_soft_grid = direction_c_soft_fusion_grid(val_c, inst_eff_col, inst_thresholds, lambdas)
            c_soft_grid.to_csv(c_dir / "validation_gated_soft_fusion_grid.csv", index=False)
            selected_c_soft = select_best_grid_row(c_soft_grid, args.fusion_objective, offline_val_metrics[args.fusion_objective])
            pd.DataFrame([selected_c_soft]).to_csv(c_dir / "selected_validation_gated_soft_fusion.csv", index=False)
            _, test_c_soft_metrics = apply_direction_c_soft_fusion(test_c, inst_eff_col, selected_c_soft)
            test_c_soft_metrics.insert(0, "direction", "C_gated_hybrid")
            test_c_soft_metrics.insert(1, "similarity_metric", sim_metric)
            test_c_soft_metrics.insert(2, "instability_gate", inst_col)
            test_c_soft_metrics.to_csv(c_dir / "test_gated_soft_fusion_metrics.csv", index=False)

            all_test_metric_rows.extend(test_c_hard_metrics.to_dict("records"))
            all_test_metric_rows.extend(test_c_soft_metrics.to_dict("records"))
            summary_rows.append(
                {
                    "direction": "C_gated_hybrid",
                    "similarity_metric": sim_metric,
                    "instability_gate": inst_col,
                    "validation_baseline_accuracy": offline_val_metrics["accuracy"],
                    "test_baseline_accuracy": offline_test_metrics["accuracy"],
                    "selected_gated_hard_policy": selected_c_hard.get("selected_policy"),
                    "selected_gated_hard_reason": selected_c_hard.get("reason"),
                    "test_gated_hard_flip_accuracy": float(test_c_hard_metrics["accuracy"].iloc[0]),
                    "test_gated_hard_flip_mcc": float(test_c_hard_metrics["mcc"].iloc[0]),
                    "selected_gated_soft_policy": selected_c_soft.get("selected_policy"),
                    "selected_gated_soft_reason": selected_c_soft.get("reason"),
                    "selected_gated_soft_lambda": float(selected_c_soft.get("lambda", np.nan)) if "lambda" in selected_c_soft else np.nan,
                    "test_gated_soft_fusion_accuracy": float(test_c_soft_metrics["accuracy"].iloc[0]),
                    "test_gated_soft_fusion_mcc": float(test_c_soft_metrics["mcc"].iloc[0]),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "directionABC_validation_test_summary.csv", index=False)
    pd.DataFrame(all_test_metric_rows).to_csv(output_root / "directionABC_all_test_metrics.csv", index=False)

    elapsed_min = (pd.Timestamp.utcnow() - start).total_seconds() / 60.0
    lines = []
    lines.append("Direction A/B/C corrected diagnostic summary")
    lines.append("===========================================")
    lines.append("")
    lines.append(f"output_root: {output_root}")
    lines.append(f"features_csv: {features_csv}")
    lines.append(f"labels_csv:   {labels_csv}")
    lines.append(f"splits_csv:   {splits_csv}")
    lines.append(f"ckpt_path:    {ckpt_path}")
    lines.append(f"selected_references_csv: {selected_refs_csv}")
    lines.append(f"train/validation/test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    lines.append(f"selected reference samples used: {len(ref_idx)}")
    lines.append(f"top_k: {args.top_k}")
    lines.append(f"alpha_prior: {args.alpha_prior}")
    lines.append(f"beta: {args.beta}")
    lines.append(f"tau_mode: {args.tau_mode}")
    lines.append(f"similarity_metrics: {similarity_metrics}")
    lines.append(f"local_instability_metrics: {parse_str_list(args.local_instability_metrics)}")
    lines.append(f"n_local_perturbations: {args.n_local_perturbations}")
    lines.append(f"perturbation_mode: {args.perturbation_mode}")
    lines.append(f"perturbation_scale: {args.perturbation_scale}")
    lines.append(f"nan_as_high_instability: {args.nan_as_high_instability}")
    lines.append(f"strict_nan_when_prediction_changes: {args.strict_nan_when_prediction_changes}")
    lines.append(f"elapsed_minutes: {elapsed_min:.3f}")
    lines.append("")
    lines.append("Offline ResNet metrics")
    lines.append("--------------------")
    lines.append(pd.DataFrame([{**{"split": "validation"}, **offline_val_metrics}, {**{"split": "test"}, **offline_test_metrics}]).to_string(index=False))
    lines.append("")
    lines.append("Direction A summary")
    lines.append("-------------------")
    if (output_root / "directionA_summary.csv").exists():
        lines.append(pd.read_csv(output_root / "directionA_summary.csv").to_string(index=False))
    else:
        lines.append("No Direction A summary available.")
    lines.append("")
    lines.append("Direction B/C validation-test summary")
    lines.append("-------------------------------------")
    if not summary_df.empty:
        lines.append(summary_df.to_string(index=False))
    else:
        lines.append("No Direction B/C summary rows available.")
    lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append("Direction A: useful only if high local RIS/ROS has strong validation and test error enrichment.")
    lines.append("Direction B: valid reference retrieval because cosine/euclidean/pearson directly compare signed fixed-p_MDD SHAP vectors.")
    lines.append("Direction C: valid hybrid because RIS/ROS is used only as an instability gate, not as a reference similarity metric.")
    lines.append("A test improvement is credible only when the rule/lambda was selected on validation and then improves test metrics.")
    (output_root / "directionABC_corrected_diagnostic_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-20:]))


if __name__ == "__main__":
    main()
