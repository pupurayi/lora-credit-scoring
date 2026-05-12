"""
Train classical baselines (Logistic Regression, Random Forest, XGBoost) on
the preprocessed Home Credit and German Credit datasets.

Each model is trained with five random seeds (42-46). Recorded per run:
test AUC, precision/recall/F1 at threshold 0.5, KS statistic, wall time,
peak memory, and test-set predictions for later DeLong/McNemar tests.

Usage:
    python models/train_baselines.py --dataset home_credit
    python models/train_baselines.py --dataset german
    python models/train_baselines.py --dataset all

Outputs (per dataset):
    results/{dataset}/{model}_seed{seed}.json
    results/{dataset}/{model}_seed{seed}_preds.npz
    results/{dataset}/baseline_summary.csv
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
TARGET_COL = "TARGET"
ID_COL_CANDIDATES = {"SK_ID_CURR", "ID"}

RESULTS_DIR = PROJECT_ROOT / "code" / "results"


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for binary classification."""
    pos = np.sort(y_score[y_true == 1])
    neg = np.sort(y_score[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    all_vals = np.unique(np.concatenate([pos, neg]))
    cdf_pos = np.searchsorted(pos, all_vals, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, all_vals, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def load_split(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a given dataset/split. Drops ID columns."""
    path = PROCESSED_DIR / dataset / f"{split}.parquet"
    df = pd.read_parquet(path)
    y = df[TARGET_COL].to_numpy().astype(np.int8)
    drop_cols = [TARGET_COL] + [c for c in df.columns if c in ID_COL_CANDIDATES]
    X = df.drop(columns=drop_cols).to_numpy(dtype=np.float32)
    return X, y


def scale_for_lr(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xva = sc.transform(X_val)
    Xte = sc.transform(X_test)
    return Xtr, Xva, Xte


def class_weight_dict(y_train: np.ndarray) -> dict[int, float]:
    pos_rate = y_train.mean()
    ratio = (1 - pos_rate) / pos_rate
    return {0: 1.0, 1: float(ratio)}


# ──────────────────────────────────────────────────────────────────────────────
# Model trainers
# ──────────────────────────────────────────────────────────────────────────────
def train_lr(seed: int, Xtr, ytr, Xva, yva, Xte, yte) -> tuple[float, np.ndarray, np.ndarray]:
    Xtr_s, Xva_s, Xte_s = scale_for_lr(Xtr, Xva, Xte)
    cw = class_weight_dict(ytr)
    t0 = time.time()
    clf = LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs",
        max_iter=2000, class_weight=cw, random_state=seed, n_jobs=-1,
    )
    clf.fit(Xtr_s, ytr)
    train_s = time.time() - t0
    proba_val = clf.predict_proba(Xva_s)[:, 1]
    proba_te = clf.predict_proba(Xte_s)[:, 1]
    return train_s, proba_val, proba_te


def train_rf(seed: int, Xtr, ytr, Xva, yva, Xte, yte) -> tuple[float, np.ndarray, np.ndarray]:
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=10,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    train_s = time.time() - t0
    proba_val = clf.predict_proba(Xva)[:, 1]
    proba_te = clf.predict_proba(Xte)[:, 1]
    return train_s, proba_val, proba_te


def train_xgb(seed: int, Xtr, ytr, Xva, yva, Xte, yte) -> tuple[float, np.ndarray, np.ndarray]:
    pos_rate = ytr.mean()
    scale_pos = float((1 - pos_rate) / pos_rate)
    t0 = time.time()
    clf = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=scale_pos, eval_metric="auc",
        tree_method="hist", random_state=seed, n_jobs=-1,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    train_s = time.time() - t0
    proba_val = clf.predict_proba(Xva)[:, 1]
    proba_te = clf.predict_proba(Xte)[:, 1]
    return train_s, proba_val, proba_te


MODELS = {
    "lr":  ("Logistic Regression", train_lr),
    "rf":  ("Random Forest",       train_rf),
    "xgb": ("XGBoost",              train_xgb),
}


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────
def metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (proba >= thr).astype(np.int8)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "ks":        ks_statistic(y_true, proba),
        "threshold": thr,
        "n_test":    int(len(y_true)),
        "n_test_pos": int(y_true.sum()),
    }


def run_one(dataset: str, model_key: str) -> None:
    name, trainer = MODELS[model_key]
    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {name} on {dataset} ===")
    print("[load] reading processed parquet splits ...")
    Xtr, ytr = load_split(dataset, "train")
    Xva, yva = load_split(dataset, "val")
    Xte, yte = load_split(dataset, "test")
    print(f"  shapes: train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    for seed in SEEDS:
        print(f"[seed {seed}] training ...")
        tracemalloc.start()
        train_s, proba_val, proba_te = trainer(seed, Xtr, ytr, Xva, yva, Xte, yte)
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        m_test = metrics_at_threshold(yte, proba_te)
        m_val = {"val_auc": float(roc_auc_score(yva, proba_val))}

        out = {
            "dataset": dataset,
            "model": model_key,
            "model_name": name,
            "seed": int(seed),
            "train_seconds": round(train_s, 3),
            "peak_memory_mb": round(peak_mem / 1e6, 2),
            **m_val,
            **m_test,
        }
        json_path = out_dir / f"{model_key}_seed{seed}.json"
        json_path.write_text(json.dumps(out, indent=2))

        # Save test predictions for later DeLong / McNemar
        npz_path = out_dir / f"{model_key}_seed{seed}_preds.npz"
        np.savez_compressed(npz_path, y_test=yte, proba_test=proba_te)

        print(f"  AUC={m_test['auc']:.4f}  F1={m_test['f1']:.4f}  "
              f"KS={m_test['ks']:.4f}  train={train_s:.1f}s  "
              f"peak_mem={peak_mem/1e6:.1f}MB  -> {json_path.name}")
        gc.collect()


def summarise(dataset: str) -> None:
    """Aggregate per-seed JSON into a mean +/- std table."""
    rows = []
    for js in sorted((RESULTS_DIR / dataset).glob("*_seed*.json")):
        rows.append(json.loads(js.read_text()))
    if not rows:
        return
    df = pd.DataFrame(rows)
    agg = df.groupby(["model", "model_name"]).agg(
        auc_mean=("auc", "mean"),
        auc_std =("auc", "std"),
        f1_mean =("f1",  "mean"),
        f1_std  =("f1",  "std"),
        ks_mean =("ks",  "mean"),
        ks_std  =("ks",  "std"),
        train_mean=("train_seconds", "mean"),
        train_std =("train_seconds", "std"),
        peak_mem_mean=("peak_memory_mb", "mean"),
    ).reset_index()
    out_path = RESULTS_DIR / dataset / "baseline_summary.csv"
    agg.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n[summary] {out_path}")
    print(agg.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["home_credit", "german", "all"],
                    default="all")
    ap.add_argument("--model", choices=["lr", "rf", "xgb", "all"],
                    default="all")
    args = ap.parse_args()

    datasets = ["home_credit", "german"] if args.dataset == "all" else [args.dataset]
    models = ["lr", "rf", "xgb"] if args.model == "all" else [args.model]

    for ds in datasets:
        for m in models:
            run_one(ds, m)
        summarise(ds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
