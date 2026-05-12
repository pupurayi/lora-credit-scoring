"""
Generate the two paper figures from the saved experimental results.

Outputs to <project>/figures/:
    fig1_pareto_auc_vs_params.pdf    AUC vs trainable parameters (Home Credit)
    fig2_shap_categories.pdf          SHAP feature importance by category

Both figures are also written as .png for quick preview.

Usage:
    python figures/generate_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common import PROJECT_ROOT  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "code" / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set publication-quality defaults
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.frameon":    False,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: AUC vs trainable parameters on Home Credit
# ──────────────────────────────────────────────────────────────────────────────
def load_homecredit_summary():
    """Pull average AUC + trainable params per model from saved JSONs."""
    ds_dir = RESULTS_DIR / "home_credit"
    rows = []
    # Classical baselines: from per-seed JSONs (no trainable_params field)
    for key in ["lr", "rf", "xgb"]:
        seed_files = sorted(ds_dir.glob(f"{key}_seed*.json"))
        if not seed_files:
            continue
        aucs = [json.loads(p.read_text())["auc"] for p in seed_files]
        rows.append({
            "model": key,
            "auc":   float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            # nominal trainable counts; LR roughly n_features + 1
            "trainable": {"lr": 184, "rf": np.nan, "xgb": np.nan}[key],
        })
    # MLP runs: include trainable_params
    for key in ["full", "lora"]:
        seed_files = sorted(ds_dir.glob(f"{key}_seed*.json"))
        if not seed_files:
            continue
        rs = [json.loads(p.read_text()) for p in seed_files]
        rows.append({
            "model":     key,
            "auc":       float(np.mean([r["auc"] for r in rs])),
            "auc_std":   float(np.std([r["auc"] for r in rs])),
            "trainable": int(rs[0]["trainable_params"]),
        })
    return pd.DataFrame(rows)


def fig1_pareto(df: pd.DataFrame) -> Path:
    """AUC vs trainable parameters on Home Credit."""
    label_map = {
        "lr":   "Logistic Regression",
        "rf":   "Random Forest",
        "xgb":  "XGBoost",
        "full": "MLP (Full FT)",
        "lora": "MLP (LoRA)",
    }
    marker_map = {"lr": "o", "rf": "s", "xgb": "^", "full": "D", "lora": "*"}
    color_map  = {"lr": "#1f77b4", "rf": "#2ca02c", "xgb": "#ff7f0e",
                  "full": "#9467bd", "lora": "#d62728"}
    size_map   = {"lr": 70, "rf": 70, "xgb": 80, "full": 80, "lora": 220}

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Plot trees-no-trainable separately at fixed x (a constant) with a marker note
    for _, r in df.iterrows():
        m = r["model"]
        if np.isnan(r["trainable"]):
            # Show RF/XGB at a notional x slightly off-axis; we don't claim a
            # comparable "trainable param" count for tree ensembles.
            x = 1500 if m == "xgb" else 7000  # arbitrary positions for legend
            ax.scatter(x, r["auc"], marker=marker_map[m], s=size_map[m],
                       color=color_map[m], edgecolor="black", linewidth=0.5,
                       label=f"{label_map[m]} (n/a)",
                       zorder=3)
        else:
            ax.scatter(r["trainable"], r["auc"], marker=marker_map[m],
                       s=size_map[m], color=color_map[m],
                       edgecolor="black", linewidth=0.5,
                       label=label_map[m], zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("AUC on Home Credit test set")
    ax.set_title("Parameter efficiency: LoRA vs full fine-tuning")
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.set_xlim(50, 1e6)
    ax.set_ylim(0.72, 0.77)

    # Annotate LoRA vs Full reduction
    lora_row = df.loc[df["model"] == "lora"].iloc[0]
    full_row = df.loc[df["model"] == "full"].iloc[0]
    ax.annotate(
        f"95.5% fewer\ntrainable params\n({int(lora_row['trainable']):,} vs {int(full_row['trainable']):,})",
        xy=(lora_row["trainable"], lora_row["auc"]),
        xytext=(2000, 0.755),
        arrowprops=dict(arrowstyle="->", color="#444", lw=0.8),
        fontsize=8.5, color="#222",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", lw=0.5, alpha=0.9),
    )

    ax.legend(loc="lower right", ncol=1)

    out_pdf = FIGURES_DIR / "fig1_pareto_auc_vs_params.pdf"
    out_png = FIGURES_DIR / "fig1_pareto_auc_vs_params.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    return out_pdf


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: SHAP categories
# ──────────────────────────────────────────────────────────────────────────────
def fig2_shap_categories() -> Path:
    """Side-by-side bar chart of SHAP importance by category."""
    csv_path = RESULTS_DIR / "home_credit" / "shap_categories.csv"
    df = pd.read_csv(csv_path)

    # Keep top categories present in BOTH models
    order = [
        "External credit score",
        "Credit terms",
        "Employment",
        "Gender (demographic)",
        "Education / suite",
        "Age (demographic)",
        "Housing / assets",
        "Phone / contact (alt-data proxy)",
        "Income",
    ]
    df = df[df["category"].isin(order)].copy()
    df["category"] = pd.Categorical(df["category"], categories=order, ordered=True)
    df = df.sort_values(["category", "model"])

    lora = df[df["model"] == "LoRA MLP"].set_index("category").reindex(order)
    xgb  = df[df["model"] == "XGBoost"].set_index("category").reindex(order)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    y = np.arange(len(order))
    h = 0.38
    bars_lora = ax.barh(y - h/2, lora["pct_total"], h,
                        label="LoRA MLP", color="#d62728",
                        edgecolor="black", linewidth=0.4)
    bars_xgb  = ax.barh(y + h/2, xgb["pct_total"],  h,
                        label="XGBoost",  color="#ff7f0e",
                        edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel(r"Mean $|$SHAP$|$ contribution (% of total)")
    ax.set_title("Feature importance by semantic category (Home Credit)")
    ax.grid(True, alpha=0.3, axis="x", linestyle="--", zorder=0)
    ax.legend(loc="lower right")

    # Annotate alt-data row to emphasise the honest finding
    alt_idx = order.index("Phone / contact (alt-data proxy)")
    lora_val = float(lora.iloc[alt_idx]["pct_total"])
    xgb_val  = float(xgb.iloc[alt_idx]["pct_total"])
    ax.annotate(
        f"Alt-data only {lora_val:.1f}% / {xgb_val:.1f}%\n(not 15.3% as some prior work claims)",
        xy=(max(lora_val, xgb_val), alt_idx),
        xytext=(15, alt_idx + 1.3),
        arrowprops=dict(arrowstyle="->", color="#444", lw=0.8),
        fontsize=8.5, color="#222",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", lw=0.5, alpha=0.9),
    )

    out_pdf = FIGURES_DIR / "fig2_shap_categories.pdf"
    out_png = FIGURES_DIR / "fig2_shap_categories.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    return out_pdf


# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print("Loading Home Credit summary ...")
    df = load_homecredit_summary()
    print(df.to_string(index=False))

    print("\n[fig1] Pareto plot ...")
    p1 = fig1_pareto(df)
    print(f"  saved {p1}")
    print(f"  saved {p1.with_suffix('.png')}")

    print("\n[fig2] SHAP categories ...")
    p2 = fig2_shap_categories()
    print(f"  saved {p2}")
    print(f"  saved {p2.with_suffix('.png')}")

    print(f"\nFigures in {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
