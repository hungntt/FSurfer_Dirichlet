#!/usr/bin/env python3
"""Direction C threshold-0.5 sensitivity test for instability percentiles 50/60.

This script is intentionally lightweight:
  - It does NOT recompute SHAP.
  - It does NOT recompute Direction B.
  - It reads the completed threshold-0.5 Direction B diagnostics.
  - It reads the completed Direction A RIS/ROS instability diagnostics.
  - It reruns only Direction C rule selection/evaluation with percentiles 50 and 60.

Default input folders:
  Direction A:
    outputs/tabm_pmdd_directionABC_split_corrected_diagnostics
  Direction B threshold-0.5:
    outputs/tabm_pmdd_directionBC_threshold05_currentA_diagnostics

Default output folder:
    outputs/tabm_pmdd_directionC_threshold05_p50p60_reuseB_currentA_diagnostics

Prediction rule remains the official report rule:
    predicted MDD if p_MDD >= 0.5
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
        parse_float_list,
        parse_int_list,
        parse_str_list,
        compute_metrics,
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
DEFAULT_DIRECTION_B_ROOT = PROJECT_DIR / "outputs/tabm_pmdd_directionBC_threshold05_currentA_diagnostics"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs/tabm_pmdd_directionC_threshold05_p50p60_reuseB_currentA_diagnostics"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reuse existing threshold-0.5 Direction B outputs and rerun Direction C for instability percentiles 50/60."
    )
    p.add_argument("--direction-a-root", type=str, default=str(DEFAULT_DIRECTION_A_ROOT))
    p.add_argument("--direction-b-root", type=str, default=str(DEFAULT_DIRECTION_B_ROOT))
    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--similarity-metrics", type=str, default="cosine,euclidean,pearson")
    p.add_argument("--local-instability-metrics", type=str, default="ris,ros")
    p.add_argument("--instability-percentiles", type=str, default="50,60")
    p.add_argument("--oppose-weight-thresholds", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75")
    p.add_argument("--min-oppose-counts", type=str, default="0,3")
    p.add_argument("--min-flags-for-rule", type=int, default=5)
    p.add_argument(
        "--lambda-grid",
        type=str,
        default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00",
    )
    p.add_argument("--fusion-objective", type=str, choices=["accuracy", "balanced_accuracy", "macro_f1", "mcc"], default="accuracy")
    p.add_argument("--nan-as-high-instability", action="store_true")
    return p.parse_args()


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def offline_metrics_from_direction_b_table(df: pd.DataFrame) -> dict:
    required = ["true_label", "tabm_pred_label", "tabm_p_MDD"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Direction B diagnostic table is missing columns {missing}. Available columns: {list(df.columns)}")
    return compute_metrics(
        df["true_label"].astype(int).to_numpy(),
        df["tabm_pred_label"].astype(int).to_numpy(),
        df["tabm_p_MDD"].astype(float).to_numpy(),
    )


def metric_value_from_metrics_df(metrics_df: pd.DataFrame, stage: str, metric: str) -> float:
    if "stage" in metrics_df.columns:
        row = metrics_df.loc[metrics_df["stage"] == stage]
        if not row.empty:
            return float(row[metric].iloc[0])
    return float(metrics_df[metric].iloc[-1])


def parse_instability_columns(metric_names: list[str]) -> list[str]:
    cols = []
    for m in metric_names:
        if m == "ris":
            cols.append("local_ris_instability")
        elif m == "ros":
            cols.append("local_ros_instability")
        else:
            raise ValueError("--local-instability-metrics must contain only ris and/or ros")
    return cols


def ensure_same_baseline(reference: dict | None, current: dict, name: str) -> dict:
    if reference is None:
        return current
    for key in ["accuracy", "balanced_accuracy", "f1", "macro_f1", "mcc"]:
        if abs(float(reference[key]) - float(current[key])) > 1e-12:
            raise RuntimeError(f"Baseline mismatch for {name}: metric {key}: {current[key]} != {reference[key]}")
    return reference


def main() -> None:
    args = parse_args()
    start = time.time()

    direction_a_root = Path(args.direction_a_root)
    direction_b_root = Path(args.direction_b_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    similarity_metrics = parse_str_list(args.similarity_metrics)
    instability_cols = parse_instability_columns(parse_str_list(args.local_instability_metrics))
    instability_percentiles = parse_float_list(args.instability_percentiles)
    oppose_thresholds = parse_float_list(args.oppose_weight_thresholds)
    min_counts = parse_int_list(args.min_oppose_counts)
    lambdas = parse_float_list(args.lambda_grid)

    if not set(similarity_metrics).issubset({"cosine", "euclidean", "pearson"}):
        raise ValueError("--similarity-metrics must contain only cosine, euclidean, pearson")

    val_inst = read_csv_required(
        direction_a_root / "directionA_local_instability" / "local_instability_validation.csv",
        "Direction A validation local instability",
    )
    test_inst = read_csv_required(
        direction_a_root / "directionA_local_instability" / "local_instability_test.csv",
        "Direction A test local instability",
    )

    summary_rows = []
    all_test_rows = []
    offline_val_metrics = None
    offline_test_metrics = None

    for sim_metric in similarity_metrics:
        b_dir = direction_b_root / "directionB_threshold05_shap_similarity" / f"metric_{sim_metric}"
        val_b = read_csv_required(b_dir / "online_validation_diagnostics.csv", f"Direction B validation diagnostics for {sim_metric}")
        test_b = read_csv_required(b_dir / "online_test_diagnostics.csv", f"Direction B test diagnostics for {sim_metric}")

        val_metrics_current = offline_metrics_from_direction_b_table(val_b)
        test_metrics_current = offline_metrics_from_direction_b_table(test_b)
        offline_val_metrics = ensure_same_baseline(offline_val_metrics, val_metrics_current, f"validation {sim_metric}")
        offline_test_metrics = ensure_same_baseline(offline_test_metrics, test_metrics_current, f"test {sim_metric}")

        for inst_col in instability_cols:
            if inst_col not in val_inst.columns or inst_col not in test_inst.columns:
                print(f"Skipping {inst_col}: column not found in Direction A outputs.", flush=True)
                continue

            print(f"Running Direction C p50/p60 reuse-B: similarity={sim_metric}, gate={inst_col}", flush=True)
            c_dir = output_root / "directionC_threshold05_p50p60_reuseB" / f"gate_{inst_col}" / f"metric_{sim_metric}"
            c_dir.mkdir(parents=True, exist_ok=True)

            val_c = attach_instability(val_b, val_inst, inst_col, args.nan_as_high_instability)
            test_c = attach_instability(test_b, test_inst, inst_col, args.nan_as_high_instability)
            val_c.to_csv(c_dir / "validation_gated_input_table.csv", index=False)
            test_c.to_csv(c_dir / "test_gated_input_table.csv", index=False)

            inst_eff_col = f"{inst_col}_effective"
            finite_val = val_c[inst_eff_col].astype(float).to_numpy()
            finite_val = finite_val[np.isfinite(finite_val)]
            if finite_val.size == 0:
                print(f"Skipping {sim_metric}+{inst_col}: no finite validation instability values.", flush=True)
                continue

            inst_thresholds = [(float(p), float(np.percentile(finite_val, p))) for p in instability_percentiles]

            hard_grid = direction_c_hard_flip_grid(
                val_c,
                inst_eff_col,
                inst_thresholds,
                oppose_thresholds,
                min_counts,
                args.min_flags_for_rule,
            )
            hard_grid.to_csv(c_dir / "validation_gated_hard_flip_grid.csv", index=False)
            selected_hard = select_best_grid_row(hard_grid, args.fusion_objective, offline_val_metrics[args.fusion_objective])
            pd.DataFrame([selected_hard]).to_csv(c_dir / "selected_validation_gated_hard_flip_rule.csv", index=False)
            _, test_hard_metrics = apply_direction_c_hard_flip(test_c, inst_eff_col, selected_hard)
            test_hard_metrics.insert(0, "direction", "C_threshold05_p50p60_reuseB")
            test_hard_metrics.insert(1, "similarity_metric", sim_metric)
            test_hard_metrics.insert(2, "instability_gate", inst_col)
            test_hard_metrics.to_csv(c_dir / "test_gated_hard_flip_metrics.csv", index=False)

            soft_grid = direction_c_soft_fusion_grid(val_c, inst_eff_col, inst_thresholds, lambdas)
            soft_grid.to_csv(c_dir / "validation_gated_soft_fusion_grid.csv", index=False)
            selected_soft = select_best_grid_row(soft_grid, args.fusion_objective, offline_val_metrics[args.fusion_objective])
            pd.DataFrame([selected_soft]).to_csv(c_dir / "selected_validation_gated_soft_fusion.csv", index=False)
            _, test_soft_metrics = apply_direction_c_soft_fusion(test_c, inst_eff_col, selected_soft)
            test_soft_metrics.insert(0, "direction", "C_threshold05_p50p60_reuseB")
            test_soft_metrics.insert(1, "similarity_metric", sim_metric)
            test_soft_metrics.insert(2, "instability_gate", inst_col)
            test_soft_metrics.to_csv(c_dir / "test_gated_soft_fusion_metrics.csv", index=False)

            if selected_hard.get("selected_policy") == "no_change":
                hard_acc = metric_value_from_metrics_df(test_hard_metrics, "directionC_gated_hard_flip", "accuracy")
                if abs(hard_acc - float(offline_test_metrics["accuracy"])) > 1e-12:
                    raise RuntimeError(f"Hard no_change sanity failed for {sim_metric}+{inst_col}: {hard_acc} != {offline_test_metrics['accuracy']}")
            if selected_soft.get("selected_policy") == "no_change":
                soft_acc = metric_value_from_metrics_df(test_soft_metrics, "directionC_gated_soft_fusion", "accuracy")
                if abs(soft_acc - float(offline_test_metrics["accuracy"])) > 1e-12:
                    raise RuntimeError(f"Soft no_change sanity failed for {sim_metric}+{inst_col}: {soft_acc} != {offline_test_metrics['accuracy']}")

            all_test_rows.extend(test_hard_metrics.to_dict("records"))
            all_test_rows.extend(test_soft_metrics.to_dict("records"))
            summary_rows.append({
                "direction": "C_threshold05_p50p60_reuseB",
                "similarity_metric": sim_metric,
                "instability_gate": inst_col,
                "direction_a_root": str(direction_a_root),
                "direction_b_root": str(direction_b_root),
                "tested_instability_percentiles": ";".join(str(int(p)) if float(p).is_integer() else str(p) for p in instability_percentiles),
                "validation_baseline_accuracy": offline_val_metrics["accuracy"],
                "test_baseline_accuracy": offline_test_metrics["accuracy"],
                "selected_gated_hard_policy": selected_hard.get("selected_policy"),
                "selected_gated_hard_percentile": selected_hard.get("instability_percentile", np.nan),
                "selected_gated_hard_reason": selected_hard.get("reason"),
                "test_gated_hard_flip_accuracy": metric_value_from_metrics_df(test_hard_metrics, "directionC_gated_hard_flip", "accuracy"),
                "test_gated_hard_flip_mcc": metric_value_from_metrics_df(test_hard_metrics, "directionC_gated_hard_flip", "mcc"),
                "selected_gated_soft_policy": selected_soft.get("selected_policy"),
                "selected_gated_soft_percentile": selected_soft.get("instability_percentile", np.nan),
                "selected_gated_soft_reason": selected_soft.get("reason"),
                "selected_gated_soft_lambda": float(selected_soft.get("lambda", np.nan)) if "lambda" in selected_soft else np.nan,
                "test_gated_soft_fusion_accuracy": metric_value_from_metrics_df(test_soft_metrics, "directionC_gated_soft_fusion", "accuracy"),
                "test_gated_soft_fusion_mcc": metric_value_from_metrics_df(test_soft_metrics, "directionC_gated_soft_fusion", "mcc"),
            })

    summary_df = pd.DataFrame(summary_rows)
    all_test_df = pd.DataFrame(all_test_rows)
    summary_df.to_csv(output_root / "directionC_threshold05_p50p60_validation_test_summary.csv", index=False)
    all_test_df.to_csv(output_root / "directionC_threshold05_p50p60_all_test_metrics.csv", index=False)

    elapsed = (time.time() - start) / 60.0
    lines = []
    lines.append("Threshold-0.5 Direction C p50/p60 reuse-existing-B summary")
    lines.append("============================================================")
    lines.append("")
    lines.append(f"direction_a_root: {direction_a_root}")
    lines.append(f"direction_b_root: {direction_b_root}")
    lines.append(f"output_root: {output_root}")
    lines.append("prediction_rule: official threshold p_MDD >= 0.5")
    lines.append("recomputed_shap: False")
    lines.append("recomputed_direction_b: False")
    lines.append(f"similarity_metrics: {similarity_metrics}")
    lines.append(f"instability_gates: {instability_cols}")
    lines.append(f"instability_percentiles: {instability_percentiles}")
    lines.append(f"elapsed_minutes: {elapsed:.3f}")
    lines.append("")
    if offline_val_metrics is not None and offline_test_metrics is not None:
        lines.append("Offline TabM threshold-0.5 metrics recovered from Direction B diagnostics")
        lines.append("-----------------------------------------------------------------------")
        lines.append(pd.DataFrame([
            {"split": "validation", **offline_val_metrics},
            {"split": "test", **offline_test_metrics},
        ]).to_string(index=False))
        lines.append("")
    lines.append("Direction C threshold-0.5 p50/p60 validation-test summary")
    lines.append("----------------------------------------------------------")
    lines.append(summary_df.to_string(index=False) if not summary_df.empty else "No Direction C rows available.")
    lines.append("")
    lines.append("Interpretation reminder: this is a sensitivity check only. Any no_change row is the original TabM baseline, not a correction improvement. Any selected intervention must exceed the test baseline to count as a real test improvement.")

    summary_txt = output_root / "directionC_threshold05_p50p60_summary.txt"
    summary_txt.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
