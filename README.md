# LoRA Credit Scoring — Reproducibility Package

Companion code for the paper:

> **Parameter-Efficient LoRA Fine-Tuning for Tabular Credit Scoring: A
> Rigorous Evaluation on Public Benchmarks.**
> P. P. Chinyavada and A. Ndlovu. *Expert Systems with Applications*, under review.

This repository reproduces every numerical claim in the paper end-to-end
from a clean clone. Two public datasets are used; both are downloaded by
scripts in this repo.

## Repository layout

```
.
├── data/
│   ├── download_german_credit.py     # UCI German Credit (1,000 rows)
│   ├── download_home_credit.py       # Home Credit Default Risk (Kaggle)
│   ├── preprocess_german_credit.py   # → data/processed/german/*.parquet
│   └── preprocess_home_credit.py     # → data/processed/home_credit/*.parquet
├── models/
│   ├── mlp.py                        # 3-layer MLP and LoRA wrapper
│   ├── train_baselines.py            # LR, RF, XGBoost (5 seeds)
│   └── train_mlp.py                  # MLP full FT and LoRA (5 seeds)
├── eval/
│   ├── statistical_tests.py          # DeLong, McNemar, Friedman
│   ├── shap_analysis.py              # SHAP for LoRA MLP and XGBoost
│   └── fairness.py                   # Disparate impact analysis
├── figures/
│   └── generate_figures.py           # Builds the two paper figures
├── results/                          # Output: per-seed JSONs + predictions
├── requirements.txt
├── requirements-optional.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.11 or 3.12 (3.13 / 3.14 may need a manual setuptools install)
- ~5 GB free disk (Home Credit unzipped is ~2.7 GB)
- A Kaggle account with an API token (for Home Credit download)

## Setup (one time)

```bash
# 1. Clone
git clone https://github.com/pupurayi/lora-credit-scoring.git
cd lora-credit-scoring

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

# 3. Dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Kaggle API token (one time)
#    a. Visit https://www.kaggle.com/settings → "Create New Token"
#    b. Place kaggle.json at ~/.kaggle/kaggle.json (Linux/Mac)
#       or %USERPROFILE%\.kaggle\kaggle.json (Windows)
#    c. Accept the competition rules:
#       https://www.kaggle.com/c/home-credit-default-risk/rules
```

## Reproduce the paper

Each command below runs to completion in tens of seconds to ~30 minutes
on a consumer laptop. Random seeds are fixed (42–46). Re-running any
script overwrites only its own outputs in `results/`.

```bash
# 1. Download both datasets
python data/download_german_credit.py
python data/download_home_credit.py

# 2. Preprocess
python data/preprocess_german_credit.py
python data/preprocess_home_credit.py

# 3. Classical baselines (LR, RF, XGBoost, 5 seeds each)
python models/train_baselines.py --dataset all

# 4. MLP under full fine-tuning, then LoRA adaptation
python models/train_mlp.py --dataset german      --mode full
python models/train_mlp.py --dataset home_credit --mode full
python models/train_mlp.py --dataset german      --mode lora \
    --checkpoint results/german/full_seed42.pt
python models/train_mlp.py --dataset home_credit --mode lora \
    --checkpoint results/home_credit/full_seed42.pt

# 5. Statistical tests, SHAP, fairness
python eval/statistical_tests.py
python eval/shap_analysis.py
python eval/fairness.py

# 6. Build the two paper figures
python figures/generate_figures.py
```

After step 6 the `figures/` directory contains the same PDFs used in the
manuscript, and `results/` contains every per-seed JSON and saved
prediction archive cited in the paper.

## What each table / figure corresponds to

| Paper artefact | Reproduced by | Output file(s) |
|----------------|---------------|----------------|
| Table 2 (AUC across models) | steps 3 + 4 | `results/*/baseline_summary.csv`, `results/*/mlp_summary.csv` |
| Table 3 (efficiency) | step 4 | `results/home_credit/full_seed*.json`, `lora_seed*.json` |
| Table 4 (statistical tests) | step 5 | `results/home_credit/delong_table.csv`, `mcnemar_table.csv`, `results/friedman.json` |
| Table 5 (SHAP categories) | step 5 | `results/home_credit/shap_categories.csv` |
| Table 6 (fairness) | step 5 | `results/home_credit/fairness_table.csv` |
| Figure 1 (Pareto plot) | step 6 | `figures/fig1_pareto_auc_vs_params.pdf` |
| Figure 2 (SHAP categories) | step 6 | `figures/fig2_shap_categories.pdf` |

## Expected runtime (CPU laptop)

| Step | Wall time |
|------|-----------|
| Downloads (1) | 5–15 min (network bound) |
| Preprocessing (2) | <30 s |
| Classical baselines (3) | 5–10 min |
| MLP full + LoRA (4) | 25–35 min |
| Tests + SHAP + fairness (5) | 10–15 min |
| Figures (6) | <10 s |

Total end-to-end: roughly 45–75 minutes after downloads.

## Datasets

- **Home Credit Default Risk** — Kaggle competition by Home Credit Group, 2018. <https://www.kaggle.com/c/home-credit-default-risk>
- **UCI German Credit (Statlog)** — UCI Machine Learning Repository. <https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data>

Both datasets are used under their respective public licences. No
proprietary or third-party data is included in this repository.

## Citing

```bibtex
@article{chinyavada2026loracredit,
  author  = {Chinyavada, Pupurayi Paula and Ndlovu, Arthur},
  title   = {Parameter-Efficient {LoRA} Fine-Tuning for Tabular Credit
             Scoring: A Rigorous Evaluation on Public Benchmarks},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Under review}
}
```

## Licence

Code in this repository is released under the MIT Licence (see `LICENSE`).
The datasets are governed by their own licences linked above.

## Contact

Pupurayi Paula Chinyavada — <h240799q@hit.ac.zw>
Harare Institute of Technology, Zimbabwe
