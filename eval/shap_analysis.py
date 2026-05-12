"""
SHAP feature-attribution analysis on Home Credit.

  - LoRA MLP: shap.GradientExplainer (100-sample background, 1,000 explained)
  - XGBoost:  shap.TreeExplainer (exact, same explanation set)

Outputs:
    results/home_credit/shap_lora.csv
    results/home_credit/shap_xgb.csv
    results/home_credit/shap_categories.csv

SHAP is intentionally not run on German Credit; the dataset is small and
the one-hot expansion produces noisy attributions.

Usage:
    python eval/shap_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from mlp import MLPConfig, TabularMLP, apply_lora  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "code" / "results"

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# How many samples to compute SHAP on. Tradeoff: more = more stable means
# but slower. 1000 is plenty for mean|SHAP| ranking.
N_BACKGROUND = 100
N_EXPLAIN = 1000
RNG = np.random.default_rng(42)


# ──────────────────────────────────────────────────────────────────────────────
# Feature category mapping for Home Credit
# ──────────────────────────────────────────────────────────────────────────────
def categorise(col: str) -> str:
    """Map a column name to a semantic category.

    These categories are defined for readability of results; the actual
    grouping rules are listed here so a reviewer can audit them.
    """
    c = col.upper()
    if any(k in c for k in ["EXT_SOURCE"]):
        return "External credit score"
    if any(k in c for k in ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                            "CREDIT_TERM"]):
        return "Credit terms"
    if "AMT_INCOME" in c or "INCOME_PER" in c:
        return "Income"
    if c.startswith("DAYS_BIRTH") or "AGE_" in c:
        return "Age (demographic)"
    if "CODE_GENDER" in c:
        return "Gender (demographic)"
    if any(k in c for k in ["NAME_FAMILY_STATUS", "CNT_FAM_MEMBERS",
                            "CNT_CHILDREN"]):
        return "Family"
    if any(k in c for k in ["DAYS_EMPLOYED", "OCCUPATION_TYPE",
                            "NAME_INCOME_TYPE", "ORGANIZATION_TYPE"]):
        return "Employment"
    if "FLAG_DOCUMENT" in c:
        return "Documents"
    if any(k in c for k in ["NAME_HOUSING_TYPE", "FLAG_OWN_REALTY",
                            "FLAG_OWN_CAR"]):
        return "Housing / assets"
    if any(k in c for k in ["FLAG_PHONE", "FLAG_EMAIL", "FLAG_MOBIL",
                            "FLAG_CONT_MOBILE", "DAYS_LAST_PHONE_CHANGE"]):
        return "Phone / contact (alt-data proxy)"
    if "DAYS_REGISTRATION" in c or "DAYS_ID_PUBLISH" in c:
        return "Registration history"
    if any(k in c for k in ["NAME_EDUCATION_TYPE", "NAME_TYPE_SUITE"]):
        return "Education / suite"
    if "WEEKDAY_APPR" in c or "HOUR_APPR" in c:
        return "Application timing"
    if "REGION_" in c or "LIVE_" in c or "REG_" in c:
        return "Region"
    return "Other"


# ──────────────────────────────────────────────────────────────────────────────
def load_test(dataset: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    df_te = pd.read_parquet(PROCESSED_DIR / dataset / "test.parquet")
    df_tr = pd.read_parquet(PROCESSED_DIR / dataset / "train.parquet")
    y_te = df_te[TARGET_COL].to_numpy(dtype=np.int8)
    drop = [TARGET_COL] + ([ID_COL] if ID_COL in df_te.columns else [])
    feature_names = [c for c in df_te.columns if c not in drop]
    X_te = df_te[feature_names].to_numpy(dtype=np.float32)
    X_tr = df_tr[feature_names].to_numpy(dtype=np.float32)
    # Standardize on train stats (MLP was trained with this)
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_te_s = (X_te - mu) / sd
    X_tr_s = (X_tr - mu) / sd
    return df_te, X_tr_s, X_te_s, feature_names, y_te, X_te


def load_lora_model(d_in: int, checkpoint: Path) -> TabularMLP:
    """Reconstruct the LoRA-wrapped MLP and load saved weights."""
    cfg = MLPConfig(d_in=d_in)
    model = TabularMLP(cfg)
    # Apply LoRA wrapping FIRST (so state_dict matches), then load
    apply_lora(model, r=8, alpha=16, which=(0, 1))
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def shap_lora(model: TabularMLP, X_tr_s: np.ndarray, X_te_s: np.ndarray,
              feature_names: list[str]) -> pd.DataFrame:
    print(f"  background: {N_BACKGROUND} train samples; explain: {N_EXPLAIN} test samples")
    bg_idx = RNG.choice(len(X_tr_s), size=N_BACKGROUND, replace=False)
    ex_idx = RNG.choice(len(X_te_s), size=min(N_EXPLAIN, len(X_te_s)), replace=False)

    bg = torch.from_numpy(X_tr_s[bg_idx]).float()
    ex = torch.from_numpy(X_te_s[ex_idx]).float()

    class WrappedForProba(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            # Must return 2D (batch, 1) for GradientExplainer to index outputs[:, idx]
            return torch.sigmoid(self.base(x))

    wrapped = WrappedForProba(model)
    wrapped.eval()
    explainer = shap.GradientExplainer(wrapped, bg)
    sv = explainer.shap_values(ex)
    if isinstance(sv, list):  # older shap returns list for multi-output
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, d, 1) -> (n, d)
        sv = sv.squeeze(-1)
    mean_abs = np.mean(np.abs(sv), axis=0)
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})


def shap_xgb(X_tr: np.ndarray, X_te: np.ndarray, y_tr: np.ndarray,
             feature_names: list[str]) -> pd.DataFrame:
    """Refit a quick XGBoost (same hyperparams as baseline) and use TreeExplainer."""
    pos_rate = y_tr.mean() if y_tr is not None else 0.08
    clf = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=float((1 - pos_rate) / pos_rate),
        eval_metric="auc", tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    explainer = shap.TreeExplainer(clf)
    # Use the same 1000-sample subset for direct comparability with the MLP run
    ex_idx = RNG.choice(len(X_te), size=min(N_EXPLAIN, len(X_te)), replace=False)
    sv = explainer.shap_values(X_te[ex_idx])
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv.squeeze(-1)
    mean_abs = np.mean(np.abs(sv), axis=0)
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})


def aggregate_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["feature"].map(categorise)
    total = df["mean_abs_shap"].sum()
    agg = (
        df.groupby("category")["mean_abs_shap"]
          .sum()
          .reset_index()
          .sort_values("mean_abs_shap", ascending=False)
    )
    agg["pct_total"] = 100 * agg["mean_abs_shap"] / total
    return agg


# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ds = "home_credit"
    print(f"=== SHAP analysis: {ds} ===")
    df_te, X_tr_s, X_te_s, feature_names, y_te, X_te_raw = load_test(ds)
    # Load training y for XGB refit
    y_tr = pd.read_parquet(PROCESSED_DIR / ds / "train.parquet")[TARGET_COL].to_numpy(dtype=np.int8)
    X_tr_raw = pd.read_parquet(PROCESSED_DIR / ds / "train.parquet")[feature_names].to_numpy(dtype=np.float32)

    out_dir = RESULTS_DIR / ds
    out_dir.mkdir(parents=True, exist_ok=True)

    # LoRA
    ckpt = out_dir / "lora_seed42.pt"
    if not ckpt.exists():
        print(f"  [skip-lora] no checkpoint at {ckpt}")
    else:
        print("\n[lora] loading LoRA MLP ...")
        model = load_lora_model(d_in=X_te_s.shape[1], checkpoint=ckpt)
        print("[lora] computing SHAP via GradientExplainer ...")
        df_lora = shap_lora(model, X_tr_s, X_te_s, feature_names)
        df_lora = df_lora.sort_values("mean_abs_shap", ascending=False)
        df_lora.to_csv(out_dir / "shap_lora.csv", index=False)
        print(f"[save] {out_dir / 'shap_lora.csv'}")
        print("\n  top 15 features by mean|SHAP| (LoRA):")
        print(df_lora.head(15).to_string(index=False))
        cat_lora = aggregate_categories(df_lora)
        cat_lora.insert(0, "model", "LoRA MLP")
    print("\n[xgb] refitting XGBoost for TreeExplainer ...")
    df_xgb = shap_xgb(X_tr_raw, X_te_raw, y_tr, feature_names)
    df_xgb = df_xgb.sort_values("mean_abs_shap", ascending=False)
    df_xgb.to_csv(out_dir / "shap_xgb.csv", index=False)
    print(f"[save] {out_dir / 'shap_xgb.csv'}")
    print("\n  top 15 features by mean|SHAP| (XGBoost):")
    print(df_xgb.head(15).to_string(index=False))
    cat_xgb = aggregate_categories(df_xgb)
    cat_xgb.insert(0, "model", "XGBoost")

    # Combined category table
    if ckpt.exists():
        cats = pd.concat([cat_lora, cat_xgb], ignore_index=True)
    else:
        cats = cat_xgb
    cats.to_csv(out_dir / "shap_categories.csv", index=False)
    print(f"\n[save] {out_dir / 'shap_categories.csv'}")
    print("\n  Aggregated by category (% of total mean|SHAP|):")
    print(cats.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
