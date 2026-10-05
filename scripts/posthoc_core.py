#!/usr/bin/env python3
"""Shared utilities for the cross-validated SHAP-Dirichlet refinement experiments.

This module replaces the two helper modules imported by the Direction A/B/C
scripts (run_tabm_directions_abc_pmdd_corrected_diagnostics.py and
run_resnet_pmdd_fixed_shap_ris_dirichlet_diagnostics.py), which are not part of
this repository. The logic follows those scripts:

- fixed-p_MDD KernelSHAP explanations,
- Quantus-style local RIS/ROS instability under feature perturbations,
- SHAP-space Top-K retrieval with exp(-beta * d / tau) evidence and a
  Dirichlet(alpha_prior + evidence) class estimate,
- validation-selected hard flip and soft fusion rules.

Two changes make the experiments reproducible across all outer folds:

1. pretab's PLE transform clamps the last bin edge to the maximum of the
   current batch, so a subject's prediction depends on which other subjects are
   in the batch. It also loops over rows in Python. We replace it with a
   vectorized transform without the batch-dependent clamp. Training inputs are
   unaffected because the training maximum always exceeds the last tree split.
2. Inference bypasses the Lightning trainer and calls the task model directly,
   which keeps KernelSHAP fast.

Run order (per backbone in {tabm, resnet}, outer fold in 1..5, split seed):
    python run_cv_gridsearch.py --backbone B --fold F [--cand-start 1 --cand-end 36]
    python run_cv_gridsearch.py --backbone B --fold F --finalize
    python run_cv_posthoc_artifacts.py --backbone B --fold F
then, once all folds exist:
    python analyze_cv_refinement.py
    python analyze_cv_reliability.py --seeds 42
    python make_paper_tables.py --seeds 42
    python make_paper_figures.py
"""
from __future__ import annotations

from pathlib import Path
import logging
import os
import random
import re
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FEATURES_CSV = DATA_DIR / "ml_features_252_159.csv"
LABELS_CSV = DATA_DIR / "ml_labels_252_159.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

ID_COL = "subject"
Y_COL = "label"
CLASS_NAMES = {0: "HC", 1: "MDD"}
SEED = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
THRESHOLD = 0.5
EPS = 1e-12
EPS_MIN = 1e-6


# ---------------------------------------------------------------------------
# Hyperparameter grids (Appendix Table "GridSearch configuration").
# ---------------------------------------------------------------------------

TABM_FIXED = {
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
}
TABM_GRID = {
    "layer_sizes": [[128], [256]],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [5e-5, 1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-4, 5e-4, 1e-3],
}

RESNET_FIXED = {
    "numerical_preprocessing": "ple",
    "n_bins": 20,
    "treat_all_integers_as_numerical": True,
    "task": "classification",
    "norm": True,
    "skip_connections": True,
    "average_embeddings": True,
}
RESNET_PROFILES = [
    {"num_blocks": 2, "layer_sizes": [128, 64, 32]},
    {"num_blocks": 3, "layer_sizes": [256, 128, 64]},
]
RESNET_GRID = {
    "profile": list(range(len(RESNET_PROFILES))),
    "dropout": [0.1, 0.2, 0.3],
    "lr": [5e-5, 1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
}


SKLEARN_BACKBONES = ("logreg", "svm", "rf")


def grid_candidates(backbone: str) -> list[dict]:
    """Return the candidate configurations of a backbone in a fixed order."""
    out = []
    if backbone == "logreg":
        return [{"model": "logreg", "C": c} for c in (1e-3, 1e-2, 1e-1, 1.0, 10.0)]
    if backbone == "svm":
        return [{"model": "svm", "C": c, "gamma": g} for c in (0.1, 1.0, 10.0) for g in ("scale", 1e-3, 1e-2)]
    if backbone == "rf":
        return [{"model": "rf", "max_depth": d, "min_samples_leaf": l, "max_features": f}
                for d in (None, 5, 10) for l in (1, 5) for f in ("sqrt", 0.3)]
    if backbone == "tabm":
        for ls in TABM_GRID["layer_sizes"]:
            for do in TABM_GRID["dropout"]:
                for lr in TABM_GRID["lr"]:
                    for wd in TABM_GRID["weight_decay"]:
                        out.append({**TABM_FIXED, "layer_sizes": list(ls), "dropout": do, "lr": lr, "weight_decay": wd})
    elif backbone == "resnet":
        for prof in RESNET_GRID["profile"]:
            for do in RESNET_GRID["dropout"]:
                for lr in RESNET_GRID["lr"]:
                    for wd in RESNET_GRID["weight_decay"]:
                        out.append({**RESNET_FIXED, **RESNET_PROFILES[prof], "dropout": do, "lr": lr, "weight_decay": wd})
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return out


# ---------------------------------------------------------------------------
# Environment patches.
# ---------------------------------------------------------------------------

_PATCHED = False


def _fast_ple_transform(self, X):
    """Vectorized, batch-independent version of pretab PLETransformer.transform."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    cache = getattr(self, "_fast_cache", None)
    if cache is None:
        cache = []
        for conds in self.conditions_:
            compiled = [compile(c, "<ple>", "eval") for c in conds]
            locs = sorted({float(m) for s in conds for m in re.findall(self.pattern, s)})
            cache.append((compiled, np.asarray(locs, dtype=np.float64)))
        self._fast_cache = cache
    n = X.shape[0]
    rows = np.arange(n)
    blocks = []
    for col, (compiled, locs) in enumerate(cache):
        feature = X[:, col]
        if locs.size == 0:
            blocks.append(np.zeros((n, self.n_bins)))
            continue
        enc = np.zeros(n, dtype=np.int64)
        for i, code in enumerate(compiled):
            enc += np.asarray(eval(code, {}, {"feature": feature}), dtype=np.int64) * (i + 1)
        b = enc - 1
        n_loc = locs.size
        ple = np.zeros((n, n_loc + 1))
        hi = feature >= locs[-1]
        lo = (~hi) & (feature <= locs[0])
        mid = ~(hi | lo)
        prefix = (np.arange(n_loc + 1)[None, :] < b[:, None]) & (hi | mid)[:, None]
        ple[prefix] = 1.0
        bm1 = np.clip(b - 1, 0, n_loc - 1)
        bc = np.clip(b, 0, n_loc - 1)
        width = locs[bc] - locs[bm1]
        width = np.where(np.abs(width) < EPS, 1.0, width)
        val = np.where(hi | lo, feature, (feature - locs[bm1]) / width)
        ple[rows, b] = val
        blocks.append(ple)
    return np.hstack(blocks).astype(np.float32)


def patch_environment() -> None:
    """Silence noisy logging, patch PLE, and remove a debug print in DeepTab TabM."""
    global _PATCHED
    if _PATCHED:
        return
    warnings.filterwarnings("ignore")
    for name in ("lightning", "lightning.pytorch", "pytorch_lightning", "lightning.fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    import pretab.transformers.ple.ple as ple_mod
    ple_mod.PLETransformer.transform = _fast_ple_transform
    import deeptab.base_models.tabm as tabm_mod
    tabm_mod.print = lambda *a, **k: None
    _PATCHED = True


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Data and splits.
# ---------------------------------------------------------------------------


def load_table(features_csv: Path = FEATURES_CSV, labels_csv: Path = LABELS_CSV):
    feats = pd.read_csv(features_csv)
    labs = pd.read_csv(labels_csv)
    df = feats.merge(labs[[ID_COL, Y_COL]], on=ID_COL, how="inner")
    if len(df) != len(feats) or len(df) != len(labs):
        raise ValueError("Feature/label subject mismatch after merge.")
    df = df.sort_values(ID_COL).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in (ID_COL, Y_COL)]
    X = df[feature_cols].astype(float)
    y = df[Y_COL].to_numpy(dtype=int)
    subjects = df[ID_COL].astype(str).to_numpy()
    return X, y, subjects, feature_cols


def outer_split(X, y, fold: int, split_seed: int = SEED):
    """Return (outer_train_idx, test_idx) for 1-based outer fold `fold`."""
    from sklearn.model_selection import StratifiedKFold
    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=split_seed)
    for k, (tr, te) in enumerate(outer.split(X, y), 1):
        if k == fold:
            return tr, te
    raise ValueError(f"Fold {fold} not found")


def inner_splits(X, y, outer_train_idx, fold: int, split_seed: int = SEED):
    """Return the inner (train_idx, val_idx) pairs in absolute indices."""
    from sklearn.model_selection import StratifiedKFold
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=split_seed + fold)
    Xo = X.iloc[outer_train_idx]
    yo = y[outer_train_idx]
    return [(outer_train_idx[a], outer_train_idx[b]) for a, b in inner.split(Xo, yo)]


def final_split(X, y, fold: int, split_seed: int = SEED):
    """Train/validation/test split of the final refit (first inner split)."""
    tr_o, te = outer_split(X, y, fold, split_seed)
    tr, va = inner_splits(X, y, tr_o, fold, split_seed)[0]
    return tr, va, te


# ---------------------------------------------------------------------------
# Models.
# ---------------------------------------------------------------------------


def _model_kwargs(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in ("lr", "weight_decay")}


def train_sklearn_model(params: dict, X_train, y_train, seed: int):
    """Fit a standardized scikit-learn classifier (no early stopping is needed)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    kind = params["model"]
    if kind == "logreg":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=5000))
    elif kind == "svm":
        clf = make_pipeline(StandardScaler(), SVC(C=params["C"], gamma=params["gamma"], kernel="rbf", probability=True, random_state=seed))
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
                                     max_features=params["max_features"], random_state=seed, n_jobs=2)
    else:
        raise ValueError(kind)
    clf.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=int))
    clf.n_epochs_trained_ = -1
    return clf


def train_model(backbone: str, params: dict, X_train, y_train, X_val, y_val, seed: int, accelerator: str | None = None):
    """Fit one classifier. DeepTab models use early stopping on (X_val, y_val)."""
    if backbone in SKLEARN_BACKBONES:
        return train_sklearn_model(params, X_train, y_train, seed)
    import torch
    patch_environment()
    from deeptab.models import ResNetClassifier, TabMClassifier
    cls = {"tabm": TabMClassifier, "resnet": ResNetClassifier}[backbone]
    if accelerator is None:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    model = cls(**_model_kwargs(params))
    ckpt_dir = tempfile.mkdtemp(prefix="ckpt_", dir=str(OUTPUT_DIR / "_tmp_ckpt") if (OUTPUT_DIR / "_tmp_ckpt").exists() else None)
    try:
        model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            batch_size=BATCH_SIZE,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            checkpoint_path=ckpt_dir,
        )
        model.n_epochs_trained_ = int(getattr(model.trainer, "current_epoch", -1))
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return model


class FastPredictor:
    has_mc_dropout = True

    """Batch-independent p(class) predictor that calls the task model directly.

    The fitted imputer -> min-max -> PLE pipeline of every feature is compiled
    into one GPU computation, which is equivalent to the patched pretab
    transform but avoids the per-feature sklearn ColumnTransformer overhead.
    """

    def __init__(self, model, feature_cols: list[str], batch_rows: int = 4096):
        import torch
        self.model = model
        self.feature_cols = list(feature_cols)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tm = model.task_model
        self.tm.eval()
        self.tm.to(self.dev)
        self.dm = model.data_module
        self.batch_rows = batch_rows
        self.n_calls = 0
        self._compile_preprocessing()

    def _compile_preprocessing(self) -> None:
        import torch
        if self.dm.cat_feature_info:
            raise ValueError("Categorical features are not supported by FastPredictor.")
        pipes = {name: pipe for name, pipe, _ in self.dm.preprocessor.column_transformer.transformers_}
        col_index, scale, offset, fill, locs_list = [], [], [], [], []
        for feat in self.dm.num_feature_info:
            pipe = pipes[f"num_{feat}"]
            names = [n for n, _ in pipe.steps]
            if names != ["imputer", "minmax", "ple"]:
                raise ValueError(f"Unexpected preprocessing pipeline for {feat}: {names}")
            ple = pipe.named_steps["ple"]
            locs = sorted({float(m) for s in ple.conditions_[0] for m in re.findall(ple.pattern, s)})
            if not locs:
                raise ValueError(f"Feature {feat} has no PLE split.")
            col_index.append(self.feature_cols.index(feat))
            scale.append(float(pipe.named_steps["minmax"].scale_[0]))
            offset.append(float(pipe.named_steps["minmax"].min_[0]))
            fill.append(float(pipe.named_steps["imputer"].statistics_[0]))
            locs_list.append(locs)
        n_feat = len(locs_list)
        l_max = max(len(l) for l in locs_list)
        locs_pad = np.full((n_feat, l_max), np.inf)
        for j, l in enumerate(locs_list):
            locs_pad[j, : len(l)] = l
        n_loc = np.asarray([len(l) for l in locs_list])
        keep = np.arange(l_max + 1)[None, :] <= n_loc[:, None]
        t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)
        self._col_index = t(col_index, torch.long)
        self._scale = t(scale)
        self._offset = t(offset)
        self._fill = t(fill)
        self._locs = t(locs_pad)
        self._first = t([l[0] for l in locs_list])
        self._last = t([l[-1] for l in locs_list])
        self._keep = t(keep.reshape(-1), torch.bool)
        self._l_max = l_max

    def encode(self, arr):
        """Return the PLE-encoded model input as one (n, sum_j(n_loc_j + 1)) tensor."""
        import torch
        x = torch.as_tensor(arr, dtype=torch.float64, device=self.dev)[:, self._col_index]
        x = torch.where(torch.isfinite(x), x, self._fill[None, :])
        v = x * self._scale[None, :] + self._offset[None, :]
        n, f = v.shape
        b = torch.searchsorted(self._locs, v.T.contiguous(), right=False).T  # (n, f)
        hi = v >= self._last[None, :]
        lo = (~hi) & (v <= self._first[None, :])
        mid = ~(hi | lo)
        pos = torch.arange(self._l_max + 1, device=self.dev)
        enc = ((pos[None, None, :] < b[:, :, None]) & (hi | mid)[:, :, None]).to(torch.float64)
        bm1 = (b - 1).clamp(min=0, max=self._l_max - 1)
        bc = b.clamp(min=0, max=self._l_max - 1)
        idx_f = torch.arange(f, device=self.dev)[None, :].expand(n, f)
        lo_edge = self._locs[idx_f, bm1]
        hi_edge = self._locs[idx_f, bc]
        width = torch.where(torch.isfinite(hi_edge - lo_edge) & ((hi_edge - lo_edge).abs() > EPS), hi_edge - lo_edge, torch.ones_like(v))
        val = torch.where(hi | lo, v, (v - lo_edge) / width)
        enc.scatter_(2, b.clamp(max=self._l_max)[:, :, None], val[:, :, None])
        return enc.reshape(n, -1)[:, self._keep].to(torch.float32)

    def predict_proba(self, X) -> np.ndarray:
        import torch
        if isinstance(X, pd.DataFrame):
            arr = X[self.feature_cols].to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        self.tm.to(self.dev)
        self.tm.eval()
        outs = []
        for s in range(0, arr.shape[0], self.batch_rows):
            num = [self.encode(arr[s : s + self.batch_rows])]
            with torch.no_grad():
                logits = self.tm(num, [], [])
            if getattr(self.tm.estimator, "returns_ensemble", False) and logits.dim() == 3:
                logits = logits.mean(dim=1)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            if logits.shape[1] == 1:
                p1 = torch.sigmoid(logits[:, 0])
                pr = torch.stack([1.0 - p1, p1], dim=1)
            else:
                pr = torch.softmax(logits, dim=1)
            outs.append(pr.float().cpu().numpy())
        self.n_calls += 1
        return np.concatenate(outs, axis=0).astype(np.float64)

    def p_mdd(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def p_mdd_tensor(self, Xt) -> np.ndarray:
        """p_MDD for a float64 tensor already on the model device (no CPU round trip)."""
        import torch
        self.tm.to(self.dev)
        self.tm.eval()
        outs = []
        for s in range(0, Xt.shape[0], self.batch_rows):
            with torch.no_grad():
                logits = self.tm([self.encode(Xt[s : s + self.batch_rows])], [], [])
            if getattr(self.tm.estimator, "returns_ensemble", False) and logits.dim() == 3:
                logits = logits.mean(dim=1)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            p1 = torch.sigmoid(logits[:, 0]) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)[:, 1]
            outs.append(p1.float().cpu().numpy())
        return np.concatenate(outs).astype(np.float64)

    def p_mdd_mc_dropout(self, X, n_passes: int = 30, seed: int = 0) -> np.ndarray:
        """p_MDD under Monte Carlo dropout, shape (n_passes, n)."""
        import torch
        arr = X[self.feature_cols].to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.atleast_2d(np.asarray(X, dtype=float))
        enc = [self.encode(arr)]
        self.tm.to(self.dev)
        out = []
        try:
            self.tm.train()
            for t in range(n_passes):
                torch.manual_seed(seed + t)
                with torch.no_grad():
                    logits = self.tm(enc, [], [])
                if logits.dim() == 1:
                    logits = logits.unsqueeze(1)
                p1 = torch.sigmoid(logits[:, 0]) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)[:, 1]
                out.append(p1.float().cpu().numpy())
        finally:
            self.tm.eval()
        return np.stack(out).astype(np.float64)


class SklearnPredictor:
    """Same interface as FastPredictor for scikit-learn classifiers."""

    has_mc_dropout = False

    def __init__(self, model, feature_cols: list[str]):
        self.model = model
        self.feature_cols = list(feature_cols)

    def predict_proba(self, X) -> np.ndarray:
        arr = X[self.feature_cols].to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.atleast_2d(np.asarray(X, dtype=float))
        return np.asarray(self.model.predict_proba(arr), dtype=np.float64)

    def p_mdd(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    p_mdd_tensor = None

    def p_mdd_mc_dropout(self, X, n_passes: int = 30, seed: int = 0) -> np.ndarray:
        p = self.p_mdd(X)
        return np.repeat(p[None, :], n_passes, axis=0)


def make_predictor(backbone: str, model, feature_cols: list[str]):
    if backbone in SKLEARN_BACKBONES:
        return SklearnPredictor(model, feature_cols)
    return FastPredictor(model, feature_cols)


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def compute_metrics(y_true, y_pred, y_score) -> dict:
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
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "average_precision": average_precision_score(y_true, y_score) if both else np.nan,
        "roc_auc": roc_auc_score(y_true, y_score) if both else np.nan,
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# SHAP.
# ---------------------------------------------------------------------------


class FixedPMDDExplainer:
    """KernelSHAP for the fixed p_MDD output, with common random numbers.

    Re-seeding before every call gives every query and its perturbations the
    same coalition samples. Differences between two explanations then reflect
    the model rather than Monte Carlo noise of KernelSHAP.
    """

    def __init__(self, predictor: FastPredictor, X_background: np.ndarray, n_background: int = 20, nsamples: int = 800, seed: int = 0):
        import shap
        self.predictor = predictor
        self.nsamples = int(nsamples)
        self.seed = int(seed)
        bg = shap.kmeans(np.asarray(X_background, dtype=float), int(n_background))
        self.explainer = shap.KernelExplainer(lambda a: predictor.p_mdd(a), bg, link="identity")
        self.expected_value = float(np.asarray(self.explainer.expected_value).reshape(-1)[0])

    def explain(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            np.random.seed(self.seed)
            vals = self.explainer.shap_values(arr[i : i + 1], nsamples=self.nsamples, l1_reg=False, silent=True)
            vals = np.asarray(vals, dtype=float).reshape(-1)
            if vals.shape[0] != arr.shape[1]:
                raise ValueError(f"Unexpected SHAP shape {vals.shape}")
            out[i] = vals
        return out


class BatchedKernelSHAP:
    """Exact batched equivalent of FixedPMDDExplainer.

    Because every explanation is computed with the same random seed, shap's
    KernelExplainer draws the same coalition masks and kernel weights for every
    input whose features all vary against the background. The Shapley estimate
    is then a fixed linear map of the coalition values, so many inputs can be
    explained with one batched model call. The masks, weights, and background
    are taken from shap itself, and `check()` compares the result with shap.
    """

    def __init__(self, f, X_background: np.ndarray, n_background: int = 20, nsamples: int = 800, seed: int = 0,
                 chunk_rows: int = 16000, device: str | None = None, f_tensor=None):
        import shap
        import torch
        self.f = f
        self.f_tensor = f_tensor
        self.nsamples = int(nsamples)
        self.seed = int(seed)
        self.chunk_rows = int(chunk_rows)
        bg = shap.kmeans(np.asarray(X_background, dtype=float), int(n_background))
        self.ref = shap.KernelExplainer(lambda a: f(a), bg, link="identity")
        self.expected_value = float(np.asarray(self.ref.expected_value).reshape(-1)[0])
        x0 = np.asarray(X_background, dtype=float)[:1]
        np.random.seed(self.seed)
        self.ref.shap_values(x0, nsamples=self.nsamples, l1_reg=False, silent=True)
        d = x0.shape[1]
        if self.ref.M != d:
            raise ValueError("Not all features vary against the background; batched KernelSHAP is not exact.")
        n_used = int(self.ref.nsamplesAdded)
        self.mask = self.ref.maskMatrix[:n_used].astype(np.float64)
        self.kw = self.ref.kernelWeights[:n_used].astype(np.float64)
        self.bg = np.asarray(self.ref.data.data, dtype=np.float64)
        self.bgw = np.asarray(self.ref.data.weights, dtype=np.float64)
        self.fnull = float(np.asarray(self.ref.fnull).reshape(-1)[0])
        X = self.mask[:, :-1] - self.mask[:, [-1]]
        WX = self.kw[:, None] * X
        self.A = np.linalg.solve(X.T @ WX, WX.T)  # (d-1, S)
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._mask_t = torch.as_tensor(self.mask > 0.5, device=self.dev)
        self._bg_t = torch.as_tensor(self.bg, device=self.dev)

    def _coalition_values(self, X: np.ndarray) -> np.ndarray:
        """Weighted background average of f for every input and coalition, shape (n, S)."""
        import torch
        n, d = X.shape
        S, N = self.mask.shape[0], self.bg.shape[0]
        per_input = S * N
        step = max(1, self.chunk_rows // per_input)
        out = np.zeros((n, S))
        Xt = torch.as_tensor(X, device=self.dev)
        for s in range(0, n, step):
            xb = Xt[s : s + step]  # (b, d)
            synth = torch.where(self._mask_t[None, :, None, :], xb[:, None, None, :], self._bg_t[None, None, :, :])  # (b,S,N,d)
            flat = synth.reshape(-1, d)
            if self.f_tensor is not None:
                y = self.f_tensor(flat)
            else:
                y = self.f(flat.cpu().numpy())
            y = np.asarray(y, dtype=np.float64).reshape(len(xb), S, N)
            out[s : s + step] = (y * self.bgw[None, None, :]).sum(-1) / self.bgw.sum()
        return out

    def explain(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        fx = np.asarray(self.f(X), dtype=np.float64).reshape(-1)
        ey = self._coalition_values(X)
        diff = fx - self.fnull
        ey_adj2 = (ey - self.fnull) - self.mask[None, :, -1] * diff[:, None]
        w = ey_adj2 @ self.A.T
        phi = np.zeros_like(X)
        phi[:, :-1] = w
        phi[:, -1] = diff - w.sum(axis=1)
        phi[np.abs(phi) < 1e-10] = 0.0
        return phi

    def check(self, X, atol: float = 1e-5) -> float:
        """Max absolute difference to shap's own KernelExplainer on rows of X."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        ours = self.explain(X)
        ref = []
        for i in range(X.shape[0]):
            np.random.seed(self.seed)
            ref.append(np.asarray(self.ref.shap_values(X[i : i + 1], nsamples=self.nsamples, l1_reg=False, silent=True)).reshape(-1))
        err = float(np.max(np.abs(ours - np.asarray(ref))))
        if err > atol:
            raise AssertionError(f"Batched KernelSHAP differs from shap by {err:.2e}")
        return err


# ---------------------------------------------------------------------------
# Local RIS/ROS instability (Quantus-style objectives).
# ---------------------------------------------------------------------------


def perturb(x: np.ndarray, rng: np.random.Generator, mode: str, scale: float, train_std: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if mode == "relative_uniform":
        return x * (1.0 + rng.uniform(-scale, scale, size=x.shape))
    if mode == "relative_gaussian":
        return x * (1.0 + rng.normal(0.0, scale, size=x.shape))
    if mode == "std_uniform":
        return x + rng.uniform(-scale, scale, size=x.shape) * train_std
    if mode == "std_gaussian":
        return x + rng.normal(0.0, scale, size=x.shape) * train_std
    raise ValueError(f"Unknown perturbation mode: {mode}")


def logit_prob_pair(proba: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=float), EPS_MIN, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.log(p)


def ris_objective(x, xs, e_x, e_xs) -> np.ndarray:
    """||(e_x - e_xs)/e_x|| / ||(x - xs)/x|| for each row of xs/e_xs."""
    x = np.asarray(x, dtype=float).reshape(1, -1)
    e_x = np.asarray(e_x, dtype=float).reshape(1, -1)
    xs = np.atleast_2d(xs)
    e_xs = np.atleast_2d(e_xs)
    nom = np.linalg.norm((e_x - e_xs) / (e_x + (e_x == 0) * EPS_MIN), axis=1)
    den = np.linalg.norm((x - xs) / (x + (x == 0) * EPS_MIN), axis=1)
    den = den + (den == 0) * EPS_MIN
    return nom / den


def ros_objective(h_x, h_xs, e_x, e_xs) -> np.ndarray:
    """||(e_x - e_xs)/e_x|| / ||h(x) - h(xs)|| with a log-probability surrogate h."""
    e_x = np.asarray(e_x, dtype=float).reshape(1, -1)
    e_xs = np.atleast_2d(e_xs)
    nom = np.linalg.norm((e_x - e_xs) / (e_x + (e_x == 0) * EPS_MIN), axis=1)
    den = np.linalg.norm(np.atleast_2d(h_x) - np.atleast_2d(h_xs), axis=1)
    den = den + (den == 0) * EPS_MIN
    return nom / den


# ---------------------------------------------------------------------------
# SHAP-space retrieval and Dirichlet evidence.
# ---------------------------------------------------------------------------


def pairwise_distance(Q: np.ndarray, R: np.ndarray, metric: str) -> np.ndarray:
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    if metric == "euclidean":
        d2 = (Q**2).sum(1)[:, None] + (R**2).sum(1)[None, :] - 2.0 * Q @ R.T
        return np.sqrt(np.clip(d2, 0.0, None))
    if metric in ("cosine", "pearson"):
        if metric == "pearson":
            Q = Q - Q.mean(axis=1, keepdims=True)
            R = R - R.mean(axis=1, keepdims=True)
        qn = np.linalg.norm(Q, axis=1)[:, None]
        rn = np.linalg.norm(R, axis=1)[None, :]
        denom = qn * rn
        denom = np.where(denom <= EPS, 1.0, denom)
        return 1.0 - np.clip((Q @ R.T) / denom, -1.0, 1.0)
    raise ValueError(f"Unknown metric {metric}")


def dirichlet_evidence(D: np.ndarray, ref_labels: np.ndarray, top_k: int, beta: float = 1.0, alpha_prior: float = 1.0,
                       exclude_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Top-K retrieval and Dirichlet class estimate for every query row of D.

    Distances are scaled by the median Top-K distance of each query
    (tau_mode = topk_median in the original scripts).
    """
    D = np.array(D, dtype=float, copy=True)
    if exclude_mask is not None:
        D[exclude_mask] = np.inf
    order = np.argsort(D, axis=1, kind="mergesort")[:, :top_k]
    d_top = np.take_along_axis(D, order, axis=1)
    lab_top = np.asarray(ref_labels, dtype=int)[order]
    tau = np.median(d_top, axis=1)
    tau = np.where(np.isfinite(tau) & (tau > 0), tau, 1.0)
    w = np.exp(-beta * d_top / (tau[:, None] + EPS))
    e_hc = (w * (lab_top == 0)).sum(axis=1)
    e_mdd = (w * (lab_top == 1)).sum(axis=1)
    a_hc = alpha_prior + e_hc
    a_mdd = alpha_prior + e_mdd
    tot = a_hc + a_mdd
    return pd.DataFrame({
        "evidence_HC": e_hc,
        "evidence_MDD": e_mdd,
        "dir_p_HC": a_hc / tot,
        "dir_p_MDD": a_mdd / tot,
        "n_MDD_topk": (lab_top == 1).sum(axis=1),
        "dir_vacuity": 2.0 / tot,
    })
