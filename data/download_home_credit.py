r"""
Download the Home Credit Default Risk dataset from Kaggle.

Prerequisites:
  1. Kaggle account: https://www.kaggle.com/account/login
  2. Accept the competition rules:
     https://www.kaggle.com/c/home-credit-default-risk/rules
  3. Generate API token (kaggle.json) and place it at:
     - Linux/Mac: ~/.kaggle/kaggle.json    (chmod 600)
     - Windows:   %USERPROFILE%\.kaggle\kaggle.json
  4. pip install kaggle  (included in requirements.txt)

Output: code/data/raw/home_credit/*.csv
Size on disk: ~2.7 GB extracted (~700 MB zip).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "home-credit-default-risk"
EXPECTED_FILES = {
    "application_train.csv": 307_512,   # row count incl. header (307,511 data rows + 1 header)
    "application_test.csv":   48_745,
    "bureau.csv":            1_716_429,
    "bureau_balance.csv":   27_299_926,
    "credit_card_balance.csv": 3_840_313,
    "installments_payments.csv": 13_605_402,
    "POS_CASH_balance.csv":  10_001_359,
    "previous_application.csv": 1_670_215,
}


def have_kaggle_token() -> bool:
    home = Path.home()
    candidates = [home / ".kaggle" / "kaggle.json"]
    if os.name == "nt":
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            candidates.append(Path(userprofile) / ".kaggle" / "kaggle.json")
    return any(p.exists() for p in candidates)


def main(out_dir: Path) -> int:
    if not have_kaggle_token():
        print("[error] kaggle.json not found.", file=sys.stderr)
        print("        Place it in ~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\kaggle.json).",
              file=sys.stderr)
        print("        Generate from: https://www.kaggle.com/settings/account -> 'Create New Token'",
              file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{COMPETITION}.zip"

    if not zip_path.exists():
        print(f"[download] kaggle competitions download -c {COMPETITION}")
        # Kaggle 2.x has no kaggle.__main__; the CLI is kaggle.cli (see PyPI kaggle>=2).
        kaggle_mod = (
            "kaggle"
            if importlib.util.find_spec("kaggle.__main__") is not None
            else "kaggle.cli"
        )
        cmd = [sys.executable, "-m", kaggle_mod, "competitions", "download",
               "-c", COMPETITION, "-p", str(out_dir)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            print("\n[hint] If you see '403 Forbidden', visit "
                  "https://www.kaggle.com/c/home-credit-default-risk/rules "
                  "and click 'I Understand and Accept' first.", file=sys.stderr)
            return r.returncode

    # Extract
    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    # Sanity check: row counts
    print("[check] verifying row counts...")
    all_ok = True
    for fname, expected in EXPECTED_FILES.items():
        f = out_dir / fname
        if not f.exists():
            print(f"  [missing] {fname}")
            all_ok = False
            continue
        n = sum(1 for _ in f.open(encoding="utf-8"))
        ok = (n == expected)
        marker = "ok" if ok else "MISMATCH"
        print(f"  [{marker}] {fname:30s} got {n:>12,d}  expected {expected:>12,d}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("[warning] Some files have unexpected row counts. "
              "Proceed cautiously - upstream dataset may have been updated.",
              file=sys.stderr)
        return 1

    print(f"[ok] Home Credit dataset ready in {out_dir}")
    return 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    sys.exit(main(root / "code" / "data" / "raw" / "home_credit"))
