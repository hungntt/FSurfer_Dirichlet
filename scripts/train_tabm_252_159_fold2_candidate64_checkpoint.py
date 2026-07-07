#!/usr/bin/env python3
from pathlib import Path
import gc
import importlib
import inspect
import json
import random
import shutil
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
import sklearn.utils.validation as sklearn_validation


_ORIGINAL_CHECK_ARRAY = sklearn_validation.check_array
_CHECK_ARRAY_PARAMS = set(inspect.signature(_ORIGINAL_CHECK_ARRAY).parameters)


def _check_array_compat(*args, ensure_all_finite=None, force_all_finite=None, **kwargs):
    """Compatibility wrapper for pretab PLE with older scikit-learn versions."""
    finite_value = ensure_all_finite if ensure_all_finite is not None else force_all_finite

    if finite_value is not None:
        if "ensure_all_finite" in _CHECK_ARRAY_PARAMS:
            kwargs["ensure_all_finite"] = finite_value
        elif "force_all_finite" in _CHECK_ARRAY_PARAMS:
            kwargs["force_all_finite"] = finite_value

    return _ORIGINAL_CHECK_ARRAY(*args, **kwargs)


def patch_check_array_compat(patch_pretab: bool = False):
    sklearn_validation.check_array = _check_array_compat

    if not patch_pretab:
        return

    for module_name in ("pretab.transformers.ple.ple",):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "check_array"):
            module.check_array = _check_array_compat


patch_check_array_compat()
from deeptab.models import TabMClassifier
patch_check_array_compat(patch_pretab=True)


BASE_DIR = Path(
    "/home/vault/empkins/tpD/D02/Students/Loc/20260611_Test_FreeSurfer/"
    "external/MDD_SSRI_structural_prediction/classification/srpbs_Toan"
)
FEATURES_CSV = BASE_DIR / "ml_features_252_159.csv"
LABELS_CSV = BASE_DIR / "ml_labels_252_159.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "tabm_252_159_fold2_candidate64_checkpoint"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_TXT = OUTPUT_DIR / "report.txt"
CONFIG_JSON = OUTPUT_DIR / "config.json"
SPLITS_CSV = OUTPUT_DIR / "splits.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
STATE_DICT_PT = OUTPUT_DIR / "task_model_state_dict.pt"
CHECKPOINT_INVENTORY_TXT = OUTPUT_DIR / "checkpoint_inventory.txt"

ID_COL = "subject"
Y_COL = "label"
FEATURE_COLS = []

SEED = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3
TARGET_OUTER_FOLD = 2
TARGET_CANDIDATE_ID = 64
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
THRESHOLD = 0.5


BEST_PARAMS = {
    "numerical_preprocessing": "ple",
    "n_bins": 20,
    "treat_all_integers_as_numerical": True,
    "task": "classification",
    "use_embeddings": False,
    "batch_norm": False,
    "layer_norm": False,
    "norm": None,
    "use_glu": False,
    "ensemble_scaling_in": True,
    "ensemble_scaling_out": True,
    "ensemble_bias": True,
    "average_ensembles": True,
    "model_type": "mini",
    "ensemble_size": 32,
    "scaling_init": "ones",
    "dropout": 0.3,
    "layer_sizes": [256],
    "lr": 1e-4,
    "weight_decay": 1e-4,
}

EXPECTED_GRIDSEARCH_FOLD2 = {
    "inner": {
        "accuracy": 0.7593,
        "balanced_accuracy": 0.7593,
        "f1": 0.7616,
        "macro_f1": 0.7593,
        "precision": 0.7524,
        "average_precision": 0.8056,
        "roc_auc": 0.8077,
        "recall_sensitivity": 0.7711,
        "specificity": 0.7475,
        "mcc": 0.5188,
    },
    "test": {
        "accuracy": 0.7129,
        "balanced_accuracy": 0.7124,
        "f1": 0.7290,
        "macro_f1": 0.7119,
        "precision": 0.6964,
        "average_precision": 0.8234,
        "roc_auc": 0.7839,
        "recall_sensitivity": 0.7647,
        "specificity": 0.6600,
        "mcc": 0.4272,
    },
}

METRIC_ORDER = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "macro_f1",
    "precision",
    "average_precision",
    "roc_auc",
    "recall_sensitivity",
    "specificity",
    "mcc",
]

PRETTY = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1",
    "macro_f1": "Macro F1",
    "precision": "Precision",
    "average_precision": "Average Precision",
    "roc_auc": "ROC AUC",
    "recall_sensitivity": "Recall/Sensitivity",
    "specificity": "Specificity",
    "mcc": "MCC",
}


def device_config():
    if torch.cuda.is_available():
        return "gpu", 1
    return "cpu", "auto"


ACCELERATOR, DEVICES = device_config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def prepare_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_xy():
    global FEATURE_COLS

    feats = pd.read_csv(FEATURES_CSV)
    labs = pd.read_csv(LABELS_CSV)

    required_feature_cols = {ID_COL}
    required_label_cols = {ID_COL, Y_COL}
    missing_feature_cols = required_feature_cols - set(feats.columns)
    missing_label_cols = required_label_cols - set(labs.columns)

    if missing_feature_cols:
        raise ValueError(f"Missing columns in feature file: {missing_feature_cols}")
    if missing_label_cols:
        raise ValueError(f"Missing columns in label file: {missing_label_cols}")

    if feats[ID_COL].duplicated().any():
        duplicated = feats.loc[feats[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(f"Duplicate subject values found in feature file: {duplicated[:10]}")
    if labs[ID_COL].duplicated().any():
        duplicated = labs.loc[labs[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(f"Duplicate subject values found in label file: {duplicated[:10]}")

    df = feats.merge(labs[[ID_COL, Y_COL]], on=ID_COL, how="inner")
    if len(df) != len(feats) or len(df) != len(labs):
        raise ValueError(
            f"Feature/label subject mismatch after merge: "
            f"features={len(feats)}, labels={len(labs)}, merged={len(df)}"
        )

    df = df.sort_values(ID_COL).reset_index(drop=True)
    FEATURE_COLS = [col for col in df.columns if col not in [ID_COL, Y_COL]]

    if not FEATURE_COLS:
        raise ValueError("No feature columns found.")

    non_numeric_cols = [
        col for col in FEATURE_COLS if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_cols:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric_cols}")

    if df[FEATURE_COLS].isna().any().any():
        bad_cols = df[FEATURE_COLS].columns[df[FEATURE_COLS].isna().any()].tolist()
        raise ValueError(f"Missing values found in feature columns: {bad_cols[:20]}")

    X = df[FEATURE_COLS].astype(float)
    y = df[Y_COL].to_numpy(dtype=int)
    subjects = df[ID_COL].astype(str).to_numpy()

    unique_labels = sorted(np.unique(y).tolist())
    if unique_labels != [0, 1]:
        raise ValueError(f"Labels must be binary 0/1. Found: {unique_labels}")

    return X, y, subjects


def model_kwargs(params: dict):
    excluded = {"lr", "weight_decay"}
    return {key: value for key, value in params.items() if key not in excluded}


def positive_scores(model, X):
    probs = np.asarray(model.predict_proba(X), dtype=float)
    probs = np.squeeze(probs)

    if probs.ndim == 1:
        return probs.astype(float)

    if probs.ndim == 2:
        if probs.shape[1] == 1:
            return probs[:, 0].astype(float)
        return probs[:, 1].astype(float)

    if probs.ndim == 3:
        n_samples = len(X)
        if probs.shape[0] == n_samples:
            probs = probs.mean(axis=1)
        elif probs.shape[1] == n_samples:
            probs = probs.mean(axis=0)
        else:
            probs = probs.mean(axis=0)

        if probs.ndim != 2:
            raise ValueError(f"Unexpected TabM probability shape after ensembling: {probs.shape}")
        if probs.shape[1] == 1:
            return probs[:, 0].astype(float)
        return probs[:, 1].astype(float)

    raise ValueError(f"Unexpected TabM probability shape: {probs.shape}")


def metrics_from_scores(y_true, y_score, threshold: float = THRESHOLD):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "average_precision": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def get_fold2_split(X, y):
    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    outer_train_idx = None
    test_idx = None

    for fold, (train_idx, fold_test_idx) in enumerate(outer.split(X, y), 1):
        if fold == TARGET_OUTER_FOLD:
            outer_train_idx = train_idx
            test_idx = fold_test_idx
            break

    if outer_train_idx is None or test_idx is None:
        raise RuntimeError(f"Could not find outer fold {TARGET_OUTER_FOLD}")

    X_outer_train = X.iloc[outer_train_idx].reset_index(drop=True)
    y_outer_train = y[outer_train_idx]

    inner = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=SEED + TARGET_OUTER_FOLD,
    )
    train_rel_idx, val_rel_idx = next(inner.split(X_outer_train, y_outer_train))

    train_idx = outer_train_idx[train_rel_idx]
    val_idx = outer_train_idx[val_rel_idx]

    return train_idx, val_idx, test_idx


def save_splits(subjects, y, train_idx, val_idx, test_idx):
    rows = []
    for split_name, indices in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:
        for idx in indices:
            rows.append({"subject": subjects[idx], "label": int(y[idx]), "split": split_name})
    pd.DataFrame(rows).sort_values(["split", "subject"]).to_csv(SPLITS_CSV, index=False)


def save_predictions(subjects, y, split_scores):
    rows = []
    for split_name, indices, scores in split_scores:
        preds = (scores >= THRESHOLD).astype(int)
        for idx, score, pred in zip(indices, scores, preds):
            rows.append(
                {
                    "subject": subjects[idx],
                    "split": split_name,
                    "y_true": int(y[idx]),
                    "y_score": float(score),
                    "y_pred_threshold_0_5": int(pred),
                }
            )
    pd.DataFrame(rows).sort_values(["split", "subject"]).to_csv(PREDICTIONS_CSV, index=False)


def checkpoint_files():
    return sorted([path for path in CHECKPOINT_DIR.rglob("*") if path.is_file()])


def ensure_named_checkpoint():
    ckpt_files = [path for path in checkpoint_files() if path.suffix == ".ckpt"]
    best_ckpt = CHECKPOINT_DIR / "best_model.ckpt"

    if best_ckpt.exists():
        return best_ckpt

    if ckpt_files:
        source = sorted(ckpt_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        shutil.copy2(source, best_ckpt)
        return best_ckpt

    return None


def save_checkpoint_inventory():
    files = checkpoint_files()
    with open(CHECKPOINT_INVENTORY_TXT, "w", encoding="utf-8") as f:
        if not files:
            f.write("No checkpoint files were found under the checkpoint directory.\n")
            return
        for path in files:
            try:
                size_mb = path.stat().st_size / (1024 ** 2)
            except OSError:
                size_mb = float("nan")
            f.write(f"{path.resolve()}\t{size_mb:.3f} MB\n")


def save_config(X, y, train_idx, val_idx, test_idx, train_seed):
    config = {
        "script_purpose": "Train one TabM model with Fold 2 Candidate 64 best parameters and save checkpoint.",
        "features_csv": str(FEATURES_CSV),
        "labels_csv": str(LABELS_CSV),
        "output_dir": str(OUTPUT_DIR),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "target_outer_fold": TARGET_OUTER_FOLD,
        "target_candidate_id": TARGET_CANDIDATE_ID,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "seed": SEED,
        "train_seed": train_seed,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "threshold": THRESHOLD,
        "accelerator": ACCELERATOR,
        "devices": DEVICES,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "class_counts": {
            "0_HC": int((y == 0).sum()),
            "1_MDD": int((y == 1).sum()),
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "best_params": BEST_PARAMS,
        "expected_gridsearch_fold2_metrics_from_previous_report": EXPECTED_GRIDSEARCH_FOLD2,
        "notes": [
            "The train/validation/test split matches final_refit_and_test() in train_tabm_252_159_fixed_grid_f1_v4.py for outer fold 2.",
            "The test metrics should match the previous Fold 2 Test line when software versions, input files, feature order, device visibility, and stochastic behavior are unchanged.",
            "The previous Inner metrics were averaged over three inner-CV models and are not metrics of this single saved checkpoint.",
        ],
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def train_model(X_train, y_train, X_val, y_val, seed):
    set_seed(seed)
    model = TabMClassifier(**model_kwargs(BEST_PARAMS))
    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        lr=BEST_PARAMS["lr"],
        weight_decay=BEST_PARAMS["weight_decay"],
        accelerator=ACCELERATOR,
        devices=DEVICES,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        checkpoint_path=str(CHECKPOINT_DIR),
    )
    return model


def write_metric_line(f, split_name, m):
    f.write(f"{split_name:<12}")
    f.write(f"Acc={m['accuracy']:.4f}, ")
    f.write(f"BalAcc={m['balanced_accuracy']:.4f}, ")
    f.write(f"F1={m['f1']:.4f}, ")
    f.write(f"MacroF1={m['macro_f1']:.4f}, ")
    f.write(f"Prec={m['precision']:.4f}, ")
    f.write(f"AP={m['average_precision']:.4f}, ")
    f.write(f"ROC-AUC={m['roc_auc']:.4f}, ")
    f.write(f"Sens={m['recall_sensitivity']:.4f}, ")
    f.write(f"Spec={m['specificity']:.4f}, ")
    f.write(f"MCC={m['mcc']:.4f}\n")


def write_report(train_idx, val_idx, test_idx, metrics_by_split, best_ckpt):
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TABM FOLD 2 CANDIDATE 64 FIXED-PARAMETER CHECKPOINT TRAINING\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Feature file: {FEATURES_CSV}\n")
        f.write(f"Label file:   {LABELS_CSV}\n")
        f.write(f"Output directory: {OUTPUT_DIR}\n")
        f.write(f"Checkpoint directory: {CHECKPOINT_DIR}\n")
        f.write(f"Best checkpoint: {best_ckpt if best_ckpt is not None else 'not found'}\n")
        f.write(f"Configuration JSON: {CONFIG_JSON}\n")
        f.write(f"Split CSV: {SPLITS_CSV}\n")
        f.write(f"Prediction CSV: {PREDICTIONS_CSV}\n")
        f.write(f"Checkpoint inventory: {CHECKPOINT_INVENTORY_TXT}\n")
        f.write(f"Task-model state_dict backup: {STATE_DICT_PT}\n\n")

        f.write(f"Target outer fold: {TARGET_OUTER_FOLD}\n")
        f.write(f"Best candidate ID: {TARGET_CANDIDATE_ID}\n")
        f.write(f"Outer CV: {OUTER_SPLITS}-fold StratifiedKFold, shuffle=True, random_state={SEED}\n")
        f.write(
            f"Validation split: first split of {INNER_SPLITS}-fold StratifiedKFold "
            f"on the outer-training subset, random_state={SEED + TARGET_OUTER_FOLD}\n"
        )
        f.write(f"Training seed: {SEED + 10000 + TARGET_OUTER_FOLD}\n")
        f.write(f"Train/validation/test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}\n")
        f.write(f"Max epochs: {MAX_EPOCHS}\n")
        f.write(f"Patience: {PATIENCE}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Threshold for reported metrics: {THRESHOLD}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        f.write(f"Training accelerator: {ACCELERATOR}, devices: {DEVICES}\n")
        f.write("DeepTab source code modification: disabled\n")
        f.write("sklearn/pretab check_array compatibility patch: enabled\n\n")

        f.write("Best Params:\n")
        f.write(json.dumps(BEST_PARAMS, indent=4) + "\n\n")

        f.write("Metrics at threshold 0.5:\n")
        for split_name in ("Train", "Validation", "Test"):
            write_metric_line(f, split_name, metrics_by_split[split_name.lower()])

        f.write("\nDetailed metric table:\n")
        f.write(f"{'Metric':<24} | {'Train':>12} | {'Validation':>12} | {'Test':>12}\n")
        f.write("-" * 70 + "\n")
        for key in METRIC_ORDER:
            f.write(
                f"{PRETTY[key]:<24} | "
                f"{metrics_by_split['train'][key]:>12.4f} | "
                f"{metrics_by_split['validation'][key]:>12.4f} | "
                f"{metrics_by_split['test'][key]:>12.4f}\n"
            )

        f.write("\nPrevious Fold 2 GridSearch reference, provided by user:\n")
        write_metric_line(f, "Inner-ref", EXPECTED_GRIDSEARCH_FOLD2["inner"])
        write_metric_line(f, "Test-ref", EXPECTED_GRIDSEARCH_FOLD2["test"])

        f.write("\nImportant note:\n")
        f.write(
            "The previous Inner metrics were produced by inner cross-validation for candidate selection. "
            "They are not expected to equal the validation metrics of this single final checkpoint. "
            "The Test metrics should match the previous Fold 2 Test line if the environment and stochastic behavior are unchanged.\n"
        )

        f.write("\nSaved checkpoint files:\n")
        ckpt_list = checkpoint_files()
        if ckpt_list:
            for path in ckpt_list:
                f.write(f"  {path.resolve()}\n")
        else:
            f.write("  No checkpoint files were found under the checkpoint directory.\n")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    prepare_output_dir()

    X, y, subjects = load_xy()
    train_idx, val_idx, test_idx = get_fold2_split(X, y)
    train_seed = SEED + 10000 + TARGET_OUTER_FOLD

    save_splits(subjects, y, train_idx, val_idx, test_idx)
    save_config(X, y, train_idx, val_idx, test_idx, train_seed)

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Training accelerator: {ACCELERATOR}, devices: {DEVICES}")
    print(f"Training TabM Fold {TARGET_OUTER_FOLD} Candidate {TARGET_CANDIDATE_ID}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Train/validation/test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    model = train_model(X_train, y_train, X_val, y_val, seed=train_seed)

    train_scores = positive_scores(model, X_train)
    val_scores = positive_scores(model, X_val)
    test_scores = positive_scores(model, X_test)

    metrics_by_split = {
        "train": metrics_from_scores(y_train, train_scores, THRESHOLD),
        "validation": metrics_from_scores(y_val, val_scores, THRESHOLD),
        "test": metrics_from_scores(y_test, test_scores, THRESHOLD),
    }

    save_predictions(
        subjects,
        y,
        [
            ("train", train_idx, train_scores),
            ("validation", val_idx, val_scores),
            ("test", test_idx, test_scores),
        ],
    )

    if hasattr(model, "task_model"):
        torch.save(model.task_model.state_dict(), STATE_DICT_PT)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_ckpt = ensure_named_checkpoint()
    save_checkpoint_inventory()
    write_report(train_idx, val_idx, test_idx, metrics_by_split, best_ckpt)

    print(f"Done. Report saved to: {REPORT_TXT.resolve()}")
    print(f"Configuration saved to: {CONFIG_JSON.resolve()}")
    print(f"Checkpoints saved under: {CHECKPOINT_DIR.resolve()}")
    if best_ckpt is not None:
        print(f"Best checkpoint: {best_ckpt.resolve()}")


if __name__ == "__main__":
    main()
