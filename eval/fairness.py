"""
Fairness analysis on Home Credit using the LoRA MLP test predictions.

Protected attributes available in Home Credit:
  - CODE_GENDER (F vs M; XNA discarded)
  - DAYS_BIRTH-derived age bands (<=30, 30-60, >60 years)

For each protected-vs-reference pair we compute the approval rate
(predicted non-default at threshold 0.5), the disparate-impact ratio, a
chi-square test on the 2x2 approval-by-group contingency, and the
four-fifths rule pass/fail at DI >= 0.80.

Usage:
    python eval/fairness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "code" / "results"

DATASET = "home_credit"
MODEL_KEY = "lora"   # use LoRA predictions
THRESHOLD = 0.5      # standard decision threshold


def load_preds_and_attrs():
    df_te = pd.read_parquet(PROCESSED_DIR / DATASET / "test.parquet")
    npz = np.load(RESULTS_DIR / DATASET / f"{MODEL_KEY}_seed42_preds.npz")
    proba = npz["proba_test"]
    y = npz["y_test"]
    assert len(proba) == len(df_te), \
        f"mismatch: parquet has {len(df_te)} rows, npz has {len(proba)}"

    # Approval: predicted non-default. Higher proba = higher predicted default
    # risk, so approve iff proba < THRESHOLD.
    approved = (proba < THRESHOLD).astype(np.int8)

    # Gender from one-hot columns
    has_gender = {"CODE_GENDER_F", "CODE_GENDER_M"}.issubset(df_te.columns)
    if has_gender:
        gender = np.where(df_te["CODE_GENDER_F"].to_numpy() == 1, "F",
                  np.where(df_te["CODE_GENDER_M"].to_numpy() == 1, "M", "X"))
    else:
        gender = None

    # Age from DAYS_BIRTH (negative integer days)
    if "DAYS_BIRTH" in df_te.columns:
        age = -df_te["DAYS_BIRTH"].to_numpy() / 365.25
        # But DAYS_BIRTH was standardized in the model input - in the parquet
        # we store the RAW preprocessed value, which is still the original
        # signed-day integer (negative). Confirm by checking range.
        if age.min() < 0:  # we got standardized values
            # If standardized, we cannot recover real age. Use rank quantiles.
            ranks = stats.rankdata(df_te["DAYS_BIRTH"]) / len(df_te)
            # DAYS_BIRTH is more negative for OLDER applicants.
            # Lower rank = more negative DAYS_BIRTH = older.
            age_band = np.where(ranks <= 1/3, "old",
                       np.where(ranks <= 2/3, "mid", "young"))
            age_real = None
        else:
            age_band = np.where(age <= 30, "young",
                       np.where(age <= 60, "mid", "old"))
            age_real = age
    else:
        age_band = None
        age_real = None

    return y, proba, approved, gender, age_band, age_real


def two_group_fairness(approved: np.ndarray, group: np.ndarray,
                       protected: str, reference: str) -> dict:
    mask_p = group == protected
    mask_r = group == reference
    n_p = int(mask_p.sum())
    n_r = int(mask_r.sum())
    if n_p == 0 or n_r == 0:
        return {"protected": protected, "reference": reference,
                "n_protected": n_p, "n_reference": n_r,
                "note": "insufficient samples in one or both groups"}
    rate_p = float(approved[mask_p].mean())
    rate_r = float(approved[mask_r].mean())
    di = rate_p / rate_r if rate_r > 0 else float("nan")

    # Contingency: approved/declined x protected/reference
    table = np.array([
        [int(approved[mask_p].sum()),     int((1 - approved[mask_p]).sum())],
        [int(approved[mask_r].sum()),     int((1 - approved[mask_r]).sum())],
    ])
    chi2, p, dof, _ = stats.chi2_contingency(table)

    return {
        "protected": protected,
        "reference": reference,
        "n_protected": n_p,
        "n_reference": n_r,
        "approval_rate_protected": round(rate_p, 4),
        "approval_rate_reference": round(rate_r, 4),
        "disparate_impact": round(di, 4),
        "passes_four_fifths_rule": bool(di >= 0.80),
        "chi2": round(float(chi2), 3),
        "p_value": float(p),
        "degrees_of_freedom": int(dof),
    }


def main() -> int:
    print(f"=== Fairness analysis: {DATASET} / {MODEL_KEY} model ===")
    y, proba, approved, gender, age_band, age_real = load_preds_and_attrs()
    n = len(y)
    overall_approval = float(approved.mean())
    overall_default = float(y.mean())
    print(f"  n_test={n:,}  overall approval rate={overall_approval:.3f}  "
          f"true default rate={overall_default:.3f}\n")

    results = []
    if gender is not None:
        print("[gender] F vs M")
        r = two_group_fairness(approved, gender, "F", "M")
        print(json.dumps(r, indent=2))
        r["attribute"] = "gender"
        results.append(r)

    if age_band is not None:
        # We construct age groups depending on whether DAYS_BIRTH was standardised.
        bands = pd.unique(age_band).tolist()
        print(f"\n[age] groups present: {bands}")
        # Pairwise: young vs mid, young vs old, old vs mid
        # (reference choice: 'mid' as the largest demographic)
        for p_, ref_ in [("young", "mid"), ("old", "mid"), ("young", "old")]:
            if p_ in bands and ref_ in bands:
                r = two_group_fairness(approved, age_band, p_, ref_)
                print(json.dumps(r, indent=2))
                r["attribute"] = "age_band"
                results.append(r)

    # Save
    out = RESULTS_DIR / DATASET / "fairness.json"
    payload = {
        "dataset": DATASET,
        "model": MODEL_KEY,
        "threshold": THRESHOLD,
        "n_test": n,
        "overall_approval_rate": overall_approval,
        "overall_default_rate": overall_default,
        "note_on_age": "DAYS_BIRTH was retained in standardized form in the "
                       "processed parquet; if so, age bands are derived from "
                       "rank quantiles (young = top-tercile DAYS_BIRTH = "
                       "youngest applicants; old = bottom-tercile = oldest).",
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))

    # CSV form for the paper
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / DATASET / "fairness_table.csv", index=False)
    print(f"\n[save] {out}")
    print(f"[save] {RESULTS_DIR / DATASET / 'fairness_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
