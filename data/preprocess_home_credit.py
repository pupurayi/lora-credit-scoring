"""
Preprocess Home Credit Default Risk for credit-scoring experiments.

Pipeline:
  1. Load application_train.csv (the main application table).
  2. Drop features with more than 40% missing values.
  3. Median-impute remaining numerical columns; mode-impute categoricals.
  4. One-hot encode categorical features.
  5. Stratified 70/15/15 split on TARGET at fixed seed.
  6. Write parquet files to data/processed/home_credit/ with metadata.

Usage:
    python data/preprocess_home_credit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    RAW_DIR,
    PROCESSED_DIR,
    SEED,
    ensure_dir,
    report_class_balance,
    set_seeds,
    stratified_3way_split,
)

# Tuneable knobs
MISSING_THRESHOLD = 0.40   # drop columns with more missingness than this
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"


def load_raw() -> pd.DataFrame:
    csv_path = RAW_DIR / "home_credit" / "application_train.csv"
    print(f"[load] {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  raw shape: {df.shape}")
    return df


def drop_high_missing(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    miss = df.isna().mean()
    dropped = miss[miss > threshold].index.tolist()
    # Never drop target or ID
    dropped = [c for c in dropped if c not in (TARGET_COL, ID_COL)]
    print(f"[drop] {len(dropped)} columns with >{threshold:.0%} missing")
    return df.drop(columns=dropped), dropped


def impute(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET_COL, ID_COL)]

    print(f"[impute] numerical (median): {len(num_cols)} cols  "
          f"categorical (mode): {len(cat_cols)} cols")

    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "MISSING"
            df[c] = df[c].fillna(fill)

    assert df.isna().sum().sum() == 0, "Imputation incomplete"
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"[encode] one-hot encoding {len(cat_cols)} categorical columns")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    # Ensure all dtypes are numeric for downstream ML
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)
    return df


def main() -> int:
    set_seeds(SEED)
    t0 = time.time()

    df = load_raw()
    raw_shape = tuple(df.shape)  # capture BEFORE we modify df
    df, dropped_high_missing = drop_high_missing(df, MISSING_THRESHOLD)
    post_drop_shape = tuple(df.shape)
    df = impute(df)

    # Keep TARGET and ID separate during encoding to avoid one-hotting them
    y = df[TARGET_COL].copy()
    ids = df[ID_COL].copy()
    df_features = df.drop(columns=[TARGET_COL, ID_COL])
    df_features = encode_categoricals(df_features)

    # Reattach
    df_out = pd.concat([ids, df_features, y], axis=1)
    print(f"[shape] post-encode: {df_out.shape}")
    n_features = df_out.shape[1] - 2  # excl. ID, TARGET
    print(f"  features = {n_features}")
    print(f"  target rate = {y.mean()*100:.2f}%  "
          f"(positive:negative = 1:{(1-y.mean())/y.mean():.2f})")

    # Stratified 70/15/15 split
    train, val, test = stratified_3way_split(df_out, TARGET_COL, seed=SEED)
    print("[split] class balance per split:")
    report_class_balance("train", train, TARGET_COL)
    report_class_balance("val",   val,   TARGET_COL)
    report_class_balance("test",  test,  TARGET_COL)

    # Save
    out_dir = ensure_dir(PROCESSED_DIR / "home_credit")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        print(f"[save] {path}  ({len(part):,d} rows, {path.stat().st_size/1e6:.1f} MB)")

    # Metadata - critical for reproducibility
    meta = {
        "seed": SEED,
        "missing_threshold": MISSING_THRESHOLD,
        "raw_shape": list(raw_shape),
        "post_drop_shape": list(post_drop_shape),
        "processed_shape": [int(x) for x in df_out.shape],
        "n_features_after_encoding": int(n_features),
        "target_positive_rate": float(y.mean()),
        "class_imbalance_ratio_neg_to_pos": float((1 - y.mean()) / y.mean()),
        "dropped_high_missing_columns": dropped_high_missing,
        "train_n": int(len(train)),
        "val_n": int(len(val)),
        "test_n": int(len(test)),
        "wall_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[save] {out_dir / 'metadata.json'}")
    print(f"[done] {meta['wall_seconds']:.1f}s elapsed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
