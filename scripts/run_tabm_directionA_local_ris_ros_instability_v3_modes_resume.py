#!/usr/bin/env python3
"""Direction A v3: local RIS/ROS instability with resume support and extra perturbation modes.

This is a drop-in replacement for run_tabm_directionA_local_ris_ros_instability_v2_resume.py.
It uses Quantus-style RIS/ROS correctly as local perturbation instability scores.

Main fixes:
- Adds perturbation modes: relative_gaussian and std_gaussian.
- Allows copying/reusing fixed-pMDD validation/test SHAP caches from an existing root.

Inherited v2 fix:
- If local_instability_<split>_progress.csv exists, the script resumes from the
  next unfinished subject instead of restarting the whole validation/test split.
- If local_instability_<split>.csv exists and --reuse-local-instability is used,
  the final file is reused directly.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import numpy as np
import pandas as pd

try:
    from run_tabm_directions_abc_pmdd_corrected_diagnostics import (
        PROJECT_DIR,
        FEATURES_CSV,
        LABELS_CSV,
        SPLITS_CSV,
        CKPT_PATH,
        CLASS_NAMES,
        EPS,
        ID_COL,
        RANDOM_STATE,
        parse_float_list,
        parse_str_list,
        patch_sklearn_for_pretab,
        load_table,
        load_split_indices,
        load_tabm,
        make_predict_function,
        build_fixed_pmdd_shap_explainer,
        compute_or_load_shap_bank,
        compute_metrics,
        build_instability_flagging_grid,
        effective_instability_scores,
        logit_prob_pair,
        compute_shap_vector_for_array,
        relative_input_stability_objective,
        relative_output_stability_objective,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import run_tabm_directions_abc_pmdd_corrected_diagnostics.py. "
        "Please place this script in the same folder as that file. "
        f"Original error: {exc}"
    )

OUTPUT_ROOT = PROJECT_DIR / "outputs/tabm_pmdd_directionABC_split_corrected_diagnostics"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direction A v3: local RIS/ROS instability with resume support and extra perturbation modes.")
    p.add_argument("--features-csv", type=str, default=str(FEATURES_CSV))
    p.add_argument("--labels-csv", type=str, default=str(LABELS_CSV))
    p.add_argument("--splits-csv", type=str, default=str(SPLITS_CSV))
    p.add_argument("--ckpt-path", type=str, default=str(CKPT_PATH))
    p.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    p.add_argument("--local-instability-metrics", type=str, default="ris,ros", help="ris,ros or one of them")
    p.add_argument("--n-local-perturbations", type=int, default=10)
    p.add_argument("--perturbation-mode", type=str, choices=["relative_uniform", "relative_gaussian", "std_uniform", "std_gaussian"], default="relative_uniform")
    p.add_argument("--perturbation-scale", type=float, default=0.02)
    p.add_argument("--instability-percentiles", type=str, default="70,80,90")
    p.add_argument("--strict-nan-when-prediction-changes", action="store_true")
    p.add_argument("--nan-as-high-instability", action="store_true")
    p.add_argument("--reuse-shap-cache", action="store_true")
    p.add_argument("--shap-cache-source-root", type=str, default=None, help="Optional existing output root whose shap_cache validation/test fixed-pMDD files will be copied into this output root before computing Direction A.")
    p.add_argument("--reuse-local-instability", action="store_true")
    p.add_argument("--resume-local-instability", action="store_true", default=True)
    p.add_argument("--no-resume-local-instability", dest="resume_local_instability", action="store_false")
    p.add_argument("--max-validation-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    return p.parse_args()



def perturb_sample(
    x: np.ndarray,
    rng: np.random.Generator,
    mode: str,
    scale: float,
    train_std: np.ndarray,
) -> np.ndarray:
    """Perturb one sample for local RIS/ROS.

    Modes:
      relative_uniform:  x * (1 + U[-scale, scale])
      relative_gaussian: x * (1 + N(0, scale))
      std_uniform:       x + U[-scale, scale] * train_std
      std_gaussian:      x + N(0, scale) * train_std

    No clipping is applied, to keep the experiment controlled and comparable with
    the previous Direction A implementation.
    """
    x = np.asarray(x, dtype=float)
    train_std = np.asarray(train_std, dtype=float)
    if mode == "relative_uniform":
        noise = rng.uniform(-scale, scale, size=x.shape)
        return x * (1.0 + noise)
    if mode == "relative_gaussian":
        noise = rng.normal(loc=0.0, scale=scale, size=x.shape)
        return x * (1.0 + noise)
    if mode == "std_uniform":
        noise = rng.uniform(-scale, scale, size=x.shape)
        return x + noise * train_std
    if mode == "std_gaussian":
        noise = rng.normal(loc=0.0, scale=scale, size=x.shape)
        return x + noise * train_std
    raise ValueError(f"Unknown perturbation mode: {mode}")


def copy_fixed_pmdd_shap_cache_if_requested(source_root: str | None, output_root: Path) -> None:
    """Copy validation/test fixed-pMDD SHAP caches from an old root if available.

    This avoids recomputing original validation/test SHAP vectors for every new
    Direction A perturbation sensitivity run. Perturbed SHAP vectors still must be
    computed because RIS/ROS depend on the new perturbation setting.
    """
    if not source_root:
        return
    src_cache = Path(source_root) / "shap_cache"
    dst_cache = output_root / "shap_cache"
    dst_cache.mkdir(parents=True, exist_ok=True)
    names = [
        "pmdd_shap_validation_fixed_pmdd_wide_raw.csv",
        "pmdd_shap_test_fixed_pmdd_wide_raw.csv",
    ]
    for name in names:
        src = src_cache / name
        dst = dst_cache / name
        if dst.exists():
            print(f"SHAP cache already exists, keeping: {dst}")
            continue
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied reusable SHAP cache: {src} -> {dst}")
        else:
            print(f"Reusable SHAP cache not found, will recompute if needed: {src}")

def offline_predictions(subjects, y, idx, X, feature_cols, predict_function, split_name: str) -> tuple[pd.DataFrame, dict]:
    probs = predict_function(X.iloc[idx][feature_cols])[:, 1]
    df = pd.DataFrame({
        "split": split_name,
        "subject": subjects[idx].astype(str),
        "true_label": y[idx].astype(int),
        "p_MDD": probs.astype(float),
    })
    df["pred_label"] = (df["p_MDD"].astype(float) >= 0.5).astype(int)
    df["pred_class"] = df["pred_label"].map(CLASS_NAMES)
    df["true_class"] = df["true_label"].map(CLASS_NAMES)
    metrics = compute_metrics(df["true_label"].to_numpy(), df["pred_label"].to_numpy(), df["p_MDD"].to_numpy())
    return df, metrics


def compute_or_resume_local_instability(
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
    """Compute local RIS/ROS instability, resuming from the progress CSV if present."""
    out_dir = output_root / "directionA_local_instability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"local_instability_{split_name}.csv"
    progress_path = out_dir / f"local_instability_{split_name}_progress.csv"
    pert_details_path = out_dir / f"local_instability_{split_name}_perturbation_details.csv"

    if args.reuse_local_instability and out_path.exists():
        print(f"Reusing final local instability file for {split_name}: {out_path}")
        return pd.read_csv(out_path)

    metrics_requested = set(parse_str_list(args.local_instability_metrics))
    if not metrics_requested.issubset({"ris", "ros"}):
        raise ValueError("--local-instability-metrics must contain only ris and/or ros.")

    rows: list[dict] = []
    done_subjects: set[str] = set()
    if args.resume_local_instability and progress_path.exists():
        progress_df = pd.read_csv(progress_path)
        if "subject" in progress_df.columns:
            rows = progress_df.to_dict("records")
            done_subjects = set(progress_df["subject"].astype(str).tolist())
            print(
                f"Resuming {split_name} local instability from progress file: "
                f"{len(done_subjects)}/{len(query_shap_df)} subjects already done."
            )

    # Keep existing perturbation details if available; otherwise start a fresh detail list.
    pert_rows_all: list[dict] = []
    if args.resume_local_instability and pert_details_path.exists():
        try:
            pert_rows_all = pd.read_csv(pert_details_path).to_dict("records")
        except Exception:
            pert_rows_all = []

    rng = np.random.default_rng(RANDOM_STATE + (0 if split_name == "validation" else 1000))

    print(
        f"\nDirection A v3: computing local RIS/ROS instability for split={split_name}, "
        f"n={len(query_shap_df)}, perturbations={args.n_local_perturbations}"
    )

    for q_pos, (_, qrow) in enumerate(query_shap_df.iterrows(), start=1):
        subject = str(qrow[ID_COL])
        if subject in done_subjects:
            continue

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
        new_pert_rows = []

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
            new_pert_rows.append(
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
        done_subjects.add(subject)
        pert_rows_all.extend(new_pert_rows)
        pd.DataFrame(rows).to_csv(progress_path, index=False)
        pd.DataFrame(pert_rows_all).to_csv(pert_details_path, index=False)

    out = pd.DataFrame(rows)
    # Keep the original split order.
    order = {str(row[ID_COL]): i for i, (_, row) in enumerate(query_shap_df.iterrows())}
    if "subject" in out.columns:
        out["_order"] = out["subject"].astype(str).map(order)
        out = out.sort_values("_order").drop(columns=["_order"])
    out.to_csv(out_path, index=False)
    return out


def main() -> None:
    args = parse_args()
    start = pd.Timestamp.utcnow()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    copy_fixed_pmdd_shap_cache_if_requested(args.shap_cache_source_root, output_root)
    patch_sklearn_for_pretab()

    instability_percentiles = parse_float_list(args.instability_percentiles)
    local_metrics = parse_str_list(args.local_instability_metrics)
    if not set(local_metrics).issubset({"ris", "ros"}):
        raise ValueError("--local-instability-metrics must contain only ris and/or ros")

    X, y, subjects, feature_cols = load_table(Path(args.features_csv), Path(args.labels_csv))
    train_idx, val_idx, test_idx, split_df = load_split_indices(Path(args.splits_csv), subjects, y)
    if args.max_validation_samples is not None:
        val_idx = val_idx[: int(args.max_validation_samples)]
    if args.max_test_samples is not None:
        test_idx = test_idx[: int(args.max_test_samples)]

    print("Loading TabM checkpoint...")
    model = load_tabm(Path(args.ckpt_path), X.iloc[train_idx][feature_cols], y[train_idx], X.iloc[val_idx][feature_cols], y[val_idx])
    predict_function = make_predict_function(model, feature_cols)
    shap_explainer = build_fixed_pmdd_shap_explainer(predict_function, X, feature_cols, train_idx)

    offline_val, offline_val_metrics = offline_predictions(subjects, y, val_idx, X, feature_cols, predict_function, "validation")
    offline_test, offline_test_metrics = offline_predictions(subjects, y, test_idx, X, feature_cols, predict_function, "test")
    offline_val.to_csv(output_root / "offline_validation_tabm_predictions.csv", index=False)
    offline_test.to_csv(output_root / "offline_test_tabm_predictions.csv", index=False)
    pd.DataFrame([
        {"stage": "offline_validation_tabm", **offline_val_metrics},
        {"stage": "offline_test_tabm", **offline_test_metrics},
    ]).to_csv(output_root / "offline_tabm_metrics.csv", index=False)

    val_shap_df = compute_or_load_shap_bank(
        "validation_fixed_pmdd", val_idx, X, y, subjects, feature_cols, predict_function, shap_explainer, output_root, args.reuse_shap_cache
    )
    test_shap_df = compute_or_load_shap_bank(
        "test_fixed_pmdd", test_idx, X, y, subjects, feature_cols, predict_function, shap_explainer, output_root, args.reuse_shap_cache
    )

    train_std = X.iloc[train_idx][feature_cols].astype(float).std(axis=0, ddof=0).to_numpy()
    train_std = np.where(np.isfinite(train_std) & (train_std > EPS), train_std, 1.0)
    val_inst = compute_or_resume_local_instability("validation", val_shap_df, X, feature_cols, predict_function, shap_explainer, output_root, args, train_std)
    test_inst = compute_or_resume_local_instability("test", test_shap_df, X, feature_cols, predict_function, shap_explainer, output_root, args, train_std)

    out_dir = output_root / "directionA_local_instability"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for inst_col in [c for c in ["local_ris_instability", "local_ros_instability"] if c in val_inst.columns]:
        grid = build_instability_flagging_grid(val_inst, inst_col, instability_percentiles, args.nan_as_high_instability)
        grid.to_csv(out_dir / f"validation_flagging_grid_{inst_col}.csv", index=False)
        if grid.empty:
            continue
        best = grid.sort_values(["error_enrichment_over_baseline", "n_flagged"], ascending=[False, False]).iloc[0].to_dict()
        threshold = float(best["threshold"])
        scores_test = effective_instability_scores(test_inst[inst_col].astype(float).to_numpy(), args.nan_as_high_instability)
        flag_test = scores_test >= threshold
        test_flagged_errors = int(((~test_inst["tabm_correct"].astype(bool)) & flag_test).sum())
        test_n_flagged = int(flag_test.sum())
        test_flag_precision = float(test_flagged_errors / test_n_flagged) if test_n_flagged else np.nan
        rows.append({
            "direction": "A_local_instability",
            "instability_col": inst_col,
            "selected_validation_percentile": best.get("percentile"),
            "selected_validation_threshold": threshold,
            "validation_flag_precision_for_errors": best.get("flag_precision_for_errors"),
            "validation_error_enrichment_over_baseline": best.get("error_enrichment_over_baseline"),
            "validation_n_flagged": best.get("n_flagged"),
            "validation_kept_accuracy": best.get("kept_accuracy"),
            "test_n_flagged_at_validation_threshold": test_n_flagged,
            "test_flagged_errors_at_validation_threshold": test_flagged_errors,
            "test_flag_precision_for_errors_at_validation_threshold": test_flag_precision,
            "test_baseline_error_rate": float(1.0 - test_inst["tabm_correct"].mean()),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / "directionA_summary.csv", index=False)

    elapsed = (pd.Timestamp.utcnow() - start).total_seconds() / 60.0
    lines = []
    lines.append("Direction A v3 local RIS/ROS instability summary")
    lines.append("================================================")
    lines.append("")
    lines.append(f"output_root: {output_root}")
    lines.append(f"train/validation/test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    lines.append(f"local_instability_metrics: {local_metrics}")
    lines.append(f"n_local_perturbations: {args.n_local_perturbations}")
    lines.append(f"perturbation_mode: {args.perturbation_mode}")
    lines.append(f"perturbation_scale: {args.perturbation_scale}")
    lines.append(f"nan_as_high_instability: {args.nan_as_high_instability}")
    lines.append(f"strict_nan_when_prediction_changes: {args.strict_nan_when_prediction_changes}")
    lines.append(f"resume_local_instability: {args.resume_local_instability}")
    lines.append(f"elapsed_minutes: {elapsed:.3f}")
    lines.append("")
    lines.append("Offline TabM metrics")
    lines.append("--------------------")
    lines.append(pd.DataFrame([
        {"split": "validation", **offline_val_metrics},
        {"split": "test", **offline_test_metrics},
    ]).to_string(index=False))
    lines.append("")
    lines.append("Direction A error-enrichment summary")
    lines.append("------------------------------------")
    lines.append(summary_df.to_string(index=False) if not summary_df.empty else "No Direction A rows available.")
    lines.append("")
    lines.append("Interpretation: Direction A can flag unstable predictions, but it does not produce HC/MDD class evidence by itself.")
    (output_root / "directionA_local_instability_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-20:]))


if __name__ == "__main__":
    main()
