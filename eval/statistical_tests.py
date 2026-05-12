"""
Statistical significance tests for the credit-scoring models.

Reads every saved *_preds.npz from results/{dataset}/ and computes pairwise
DeLong tests for AUC equality, pairwise McNemar tests for error-pattern
equality, and a Friedman test for global rank significance across datasets.

Usage:
    python eval/statistical_tests.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common import PROJECT_ROOT  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "code" / "results"
DATASETS = ["home_credit", "german"]

# Pretty names used in tables / paper
DISPLAY = {
    "lr":   "Logistic Regression",
    "rf":   "Random Forest",
    "xgb":  "XGBoost",
    "full": "MLP (Full FT)",
    "lora": "MLP (LoRA)",
}
MODEL_ORDER = ["lr", "rf", "xgb", "full", "lora"]


# ──────────────────────────────────────────────────────────────────────────────
# DeLong's test (vectorised implementation from Sun & Xu, 2014)
# ──────────────────────────────────────────────────────────────────────────────
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank used by DeLong's algorithm."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # mid-rank, 1-indexed
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def _delong_components(predictions_sorted_transposed: np.ndarray,
                       label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    """
    The fast DeLong algorithm.
    `predictions_sorted_transposed` is shape (k_models, n_samples) with
    positives in the FIRST `label_1_count` columns.
    Returns (aucs, covariance_matrix).
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        tx[r] = _compute_midrank(predictions_sorted_transposed[r, :m])
        ty[r] = _compute_midrank(predictions_sorted_transposed[r, m:])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.array([[float(sx)]])
        sy = np.array([[float(sy)]])
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true: np.ndarray, p1: np.ndarray,
                    p2: np.ndarray) -> dict:
    """
    Two-sided DeLong test of AUC equality between two correlated ROC curves.
    Returns dict with aucs, diff, z, p, ci_diff_low, ci_diff_high.
    """
    order = np.argsort(-y_true.astype(np.int8))
    yt = y_true[order]
    p_sorted = np.stack([p1[order], p2[order]], axis=0)
    m = int(yt.sum())  # positives count
    aucs, cov = _delong_components(p_sorted, m)
    diff = aucs[0] - aucs[1]
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        z = 0.0
        p = 1.0
        se = 0.0
    else:
        se = float(np.sqrt(var_diff))
        z = float(diff / se)
        p = float(2.0 * stats.norm.sf(abs(z)))
    return {
        "auc1": float(aucs[0]),
        "auc2": float(aucs[1]),
        "diff": float(diff),
        "se_diff": se,
        "z": z,
        "p": p,
        "ci_diff_low":  float(diff - 1.959963984540054 * se),
        "ci_diff_high": float(diff + 1.959963984540054 * se),
    }


def delong_auc_ci(y_true: np.ndarray, p: np.ndarray,
                  alpha: float = 0.05) -> tuple[float, float, float]:
    """Single-model AUC with DeLong-style 95% CI."""
    order = np.argsort(-y_true.astype(np.int8))
    yt = y_true[order]
    p_sorted = p[order].reshape(1, -1)
    m = int(yt.sum())
    aucs, cov = _delong_components(p_sorted, m)
    auc = float(aucs[0])
    se = float(np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else 0.0
    z = stats.norm.ppf(1 - alpha / 2)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


# ──────────────────────────────────────────────────────────────────────────────
# McNemar
# ──────────────────────────────────────────────────────────────────────────────
def mcnemar_test(y_true, p1, p2, threshold=0.5) -> dict:
    yh1 = (p1 >= threshold).astype(np.int8)
    yh2 = (p2 >= threshold).astype(np.int8)
    c1 = (yh1 == y_true)
    c2 = (yh2 == y_true)
    b = int(((c1) & (~c2)).sum())  # 1 right, 2 wrong
    c = int(((~c1) & (c2)).sum())  # 1 wrong, 2 right
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p": 1.0, "odds_ratio": float("nan")}
    # Continuity-corrected chi-square
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = float(stats.chi2.sf(chi2, df=1))
    odds = (b + 0.5) / (c + 0.5)  # Haldane correction
    return {"b": b, "c": c, "chi2": float(chi2),
            "p": p, "odds_ratio": float(odds)}


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation: pick the seed=42 predictions for each model
# ──────────────────────────────────────────────────────────────────────────────
def load_preds(dataset: str, model_key: str, seed: int = 42):
    path = RESULTS_DIR / dataset / f"{model_key}_seed{seed}_preds.npz"
    if not path.exists():
        return None, None
    d = np.load(path)
    return d["y_test"].astype(np.int8), d["proba_test"].astype(np.float64)


def build_delong_table(dataset: str) -> pd.DataFrame:
    rows = []
    auc_cache = {}
    for m in MODEL_ORDER:
        y, p = load_preds(dataset, m)
        if y is None:
            continue
        auc, lo, hi = delong_auc_ci(y, p)
        auc_cache[m] = (y, p, auc, lo, hi)
        rows.append({
            "model": DISPLAY[m],
            "key":   m,
            "auc":   round(auc, 4),
            "auc_ci_low":  round(lo, 4),
            "auc_ci_high": round(hi, 4),
        })
    auc_df = pd.DataFrame(rows)

    # Pairwise comparisons against LoRA
    if "lora" in auc_cache:
        y_l, p_l, *_ = auc_cache["lora"]
        for m in MODEL_ORDER:
            if m == "lora" or m not in auc_cache:
                continue
            y_m, p_m, *_ = auc_cache[m]
            # Sanity: same y_test
            assert np.array_equal(y_l, y_m), \
                f"y_test mismatch between lora and {m} for {dataset}"
            r = delong_roc_test(y_l, p_l, p_m)
            auc_df.loc[auc_df["key"] == m, "delong_diff_vs_lora"] = round(r["diff"], 4)
            auc_df.loc[auc_df["key"] == m, "delong_z"] = round(r["z"], 3)
            auc_df.loc[auc_df["key"] == m, "delong_p"] = r["p"]
    return auc_df


def build_mcnemar_table(dataset: str) -> pd.DataFrame:
    """Pairwise McNemar between every pair of models, seed 42."""
    rows = []
    cache = {}
    for m in MODEL_ORDER:
        y, p = load_preds(dataset, m)
        if y is not None:
            cache[m] = (y, p)
    keys = list(cache.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            mi, mj = keys[i], keys[j]
            yi, pi = cache[mi]
            yj, pj = cache[mj]
            assert np.array_equal(yi, yj)
            r = mcnemar_test(yi, pi, pj)
            rows.append({
                "model_1": DISPLAY[mi],
                "model_2": DISPLAY[mj],
                "b_(1right_2wrong)": r["b"],
                "c_(1wrong_2right)": r["c"],
                "chi2":  round(r["chi2"], 3),
                "p":     r["p"],
                "odds_ratio": round(r["odds_ratio"], 3),
            })
    return pd.DataFrame(rows)


def friedman_across_all() -> dict:
    """
    Friedman test treating each dataset as a block and each model as a treatment.
    Per-block AUC = mean across all available seeds (matches what Table 2 in
    the paper reports). Using single-seed AUC would give a less stable ranking,
    especially on the small German Credit test set.
    """
    import json
    auc_matrix = []
    models_present = []
    for m in MODEL_ORDER:
        row = []
        ok = True
        for ds in DATASETS:
            seed_files = sorted((RESULTS_DIR / ds).glob(f"{m}_seed*.json"))
            if not seed_files:
                ok = False
                break
            aucs = [json.loads(p.read_text())["auc"] for p in seed_files]
            row.append(float(np.mean(aucs)))
        if ok:
            auc_matrix.append(row)
            models_present.append(m)

    arr = np.array(auc_matrix)  # shape (n_models, n_datasets)
    if arr.shape[0] < 3:
        return {"note": "need >=3 models for Friedman", "n_models": arr.shape[0]}
    # scipy expects each argument = one treatment across blocks
    stat, p = stats.friedmanchisquare(*[arr[i] for i in range(arr.shape[0])])
    # Average ranks across blocks (rank 1 = best AUC = highest)
    ranks = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        ranks[:, j] = stats.rankdata(-arr[:, j])
    avg_ranks = ranks.mean(axis=1)
    return {
        "statistic": float(stat),
        "p": float(p),
        "df": int(arr.shape[0] - 1),
        "n_datasets": int(arr.shape[1]),
        "models": [DISPLAY[m] for m in models_present],
        "auc_matrix": arr.tolist(),
        "avg_ranks": dict(zip([DISPLAY[m] for m in models_present],
                              [round(float(r), 3) for r in avg_ranks])),
    }


# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        d_tab = build_delong_table(ds)
        m_tab = build_mcnemar_table(ds)
        out_d = RESULTS_DIR / ds / "delong_table.csv"
        out_m = RESULTS_DIR / ds / "mcnemar_table.csv"
        d_tab.to_csv(out_d, index=False)
        m_tab.to_csv(out_m, index=False)
        print(f"[delong]   {out_d}")
        print(d_tab.to_string(index=False))
        print(f"\n[mcnemar]  {out_m}  ({len(m_tab)} pairs)")
        print(m_tab.to_string(index=False))

    print("\n=== Friedman across all datasets ===")
    f = friedman_across_all()
    print(json.dumps(f, indent=2))
    (RESULTS_DIR / "friedman.json").write_text(json.dumps(f, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
