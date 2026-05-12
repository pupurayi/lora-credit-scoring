"""Shared helpers for the preprocessing scripts. Fixed seed for reproducibility."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42  # fixed everywhere; do not change after first run

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
RAW_DIR = CODE_DIR / "data" / "raw"
PROCESSED_DIR = CODE_DIR / "data" / "processed"


def stratified_3way_split(
    df: pd.DataFrame,
    target_col: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 split preserving class proportions.
    Returns (train, val, test).
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-9

    df_train, df_temp = train_test_split(
        df,
        test_size=(1 - train_size),
        stratify=df[target_col],
        random_state=seed,
    )
    val_frac_of_temp = val_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1 - val_frac_of_temp),
        stratify=df_temp[target_col],
        random_state=seed,
    )
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def report_class_balance(name: str, df: pd.DataFrame, target_col: str) -> None:
    pct = df[target_col].mean() * 100
    print(f"  {name:>8s}: n={len(df):>7,d}  positive_rate={pct:5.2f}%")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seeds(seed: int = SEED) -> None:
    """Set numpy seed. Torch seed set in model scripts."""
    np.random.seed(seed)
