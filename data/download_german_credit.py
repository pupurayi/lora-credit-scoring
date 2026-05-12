"""
Download the UCI German Credit (Statlog) dataset.

The dataset has two forms in the UCI repo:
  - german.data         : 20 attributes, mix of categorical (encoded as Aii)
                          and numerical, plus the label (1 = good, 2 = bad)
  - german.data-numeric : 24 numerical features, already standardised

We download both. The categorical version is what the paper uses
(1,000 rows x 20 attributes, then one-hot encoded).

Output: code/data/raw/german/{german.data, german.doc, german.data-numeric}
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/"
FILES = ["german.data", "german.data-numeric", "german.doc"]


def main(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        target = out_dir / fname
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] {fname} already present")
            continue
        url = UCI_BASE + fname
        print(f"[download] {url}")
        try:
            urlretrieve(url, target)
        except Exception as e:  # noqa: BLE001
            print(f"[error] Failed to download {fname}: {e}", file=sys.stderr)
            return 1

    # Sanity: german.data must have 1000 rows
    n = sum(1 for _ in (out_dir / "german.data").open())
    print(f"[check] german.data row count = {n} (expected 1000)")
    if n != 1000:
        print("[error] Row count mismatch - aborting.", file=sys.stderr)
        return 1

    print(f"[ok] German Credit dataset ready in {out_dir}")
    return 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    sys.exit(main(root / "code" / "data" / "raw" / "german"))
