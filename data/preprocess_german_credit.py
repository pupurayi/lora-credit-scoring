"""
Preprocess UCI German Credit (Statlog) for credit-scoring experiments.

Format reference: UCI german.doc
  - 20 attributes, columns separated by single space
  - mix of categorical (encoded "A11", "A12" etc.) and numerical
  - last column is label: 1 = good credit, 2 = bad credit
  - We remap: bad (2) -> 1, good (1) -> 0 so that positive class = "default"
    (matches Home Credit convention).

Run from the code/ folder:
    python data/preprocess_german_credit.py

Outputs:
    code/data/processed/german/train.parquet
    code/data/processed/german/val.parquet
    code/data/processed/german/test.parquet
    code/data/processed/german/metadata.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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

# Column names per UCI german.doc
COLUMNS = [
    "checking_status",        # A11..A14 (categorical)
    "duration_months",        # numerical
    "credit_history",         # A30..A34
    "purpose",                # A40..A410
    "credit_amount",          # numerical
    "savings_status",         # A61..A65
    "employment_since",       # A71..A75
    "installment_rate_pct",   # numerical
    "personal_status_sex",    # A91..A95
    "other_debtors",          # A101..A103
    "residence_since",        # numerical
    "property",               # A121..A124
    "age_years",              # numerical
    "other_installment_plans",# A141..A143
    "housing",                # A151..A153
    "existing_credits",       # numerical
    "job",                    # A171..A174
    "n_dependents",           # numerical
    "telephone",              # A191..A192
    "foreign_worker",         # A201..A202
    "label_uci",              # 1 = good, 2 = bad
]

CATEGORICAL = [c for c in COLUMNS if c not in {
    "duration_months", "credit_amount", "installment_rate_pct",
    "residence_since", "age_years", "existing_credits", "n_dependents",
    "label_uci",
}]

TARGET_COL = "TARGET"  # post-remap: 1 = default (bad), 0 = non-default


def load_raw() -> pd.DataFrame:
    path = RAW_DIR / "german" / "german.data"
    print(f"[load] {path}")
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMNS)
    print(f"  raw shape: {df.shape}")
    return df


def main() -> int:
    set_seeds(SEED)
    t0 = time.time()

    df = load_raw()

    # Remap label: UCI 1=good -> 0, UCI 2=bad -> 1 (= default)
    df[TARGET_COL] = (df["label_uci"] == 2).astype(np.int8)
    df = df.drop(columns=["label_uci"])

    # One-hot encode categoricals
    print(f"[encode] one-hot encoding {len(CATEGORICAL)} categorical columns")
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=False)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)

    n_features = df.shape[1] - 1
    pos_rate = df[TARGET_COL].mean()
    print(f"[shape] post-encode: {df.shape}  features={n_features}")
    print(f"  target rate = {pos_rate*100:.2f}%  "
          f"(non-default:default ratio = {(1-pos_rate)/pos_rate:.2f})")

    train, val, test = stratified_3way_split(df, TARGET_COL, seed=SEED)
    print("[split] class balance per split:")
    report_class_balance("train", train, TARGET_COL)
    report_class_balance("val",   val,   TARGET_COL)
    report_class_balance("test",  test,  TARGET_COL)

    out_dir = ensure_dir(PROCESSED_DIR / "german")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        print(f"[save] {path}  ({len(part):,d} rows, {path.stat().st_size/1e3:.1f} KB)")

    meta = {
        "seed": SEED,
        "raw_shape": [1000, 21],
        "processed_shape": [int(x) for x in df.shape],
        "n_features_after_encoding": int(n_features),
        "target_positive_rate": float(pos_rate),
        "class_imbalance_ratio_neg_to_pos": float((1 - pos_rate) / pos_rate),
        "label_mapping": {"original_1_good": 0, "original_2_bad": 1},
        "train_n": int(len(train)),
        "val_n": int(len(val)),
        "test_n": int(len(test)),
        "wall_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[save] {out_dir / 'metadata.json'}")
    print(f"[done] {meta['wall_seconds']:.2f}s elapsed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
