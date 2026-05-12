"""
Train the tabular MLP - either with full fine-tuning (all parameters) or
with LoRA adapters (only adapters trainable).

Modes:
    --mode full   Train all parameters from random initialisation.
    --mode lora   Load a pre-trained .pt, freeze it, inject LoRA adapters
                  (r=8, alpha=16) into the two largest Linear layers, train
                  only adapters and the classifier head.

Usage:
    python models/train_mlp.py --dataset home_credit --mode full
    python models/train_mlp.py --dataset home_credit --mode lora \\
        --checkpoint results/home_credit/full_seed42.pt
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp import (  # noqa: E402
    MLPConfig, TabularMLP, apply_lora,
    count_trainable, count_total,
)

SEEDS = [42, 43, 44, 45, 46]
TARGET_COL = "TARGET"
ID_COL_CANDIDATES = {"SK_ID_CURR", "ID"}

RESULTS_DIR = PROJECT_ROOT / "code" / "results"


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def load_split(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = PROCESSED_DIR / dataset / f"{split}.parquet"
    df = pd.read_parquet(path)
    y = df[TARGET_COL].to_numpy().astype(np.float32)
    drop = [TARGET_COL] + [c for c in df.columns if c in ID_COL_CANDIDATES]
    X = df.drop(columns=drop).to_numpy(dtype=np.float32)
    return X, y


def standardize(Xtr, Xva, Xte):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


def make_loaders(Xtr, ytr, Xva, yva, batch=512, seed=42):
    g = torch.Generator(); g.manual_seed(seed)
    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    val_ds   = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, generator=g)
    val_ld   = DataLoader(val_ds, batch_size=4096, shuffle=False)
    return train_ld, val_ld


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def ks_statistic(y_true, y_score):
    pos = np.sort(y_score[y_true == 1])
    neg = np.sort(y_score[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    all_vals = np.unique(np.concatenate([pos, neg]))
    cdf_pos = np.searchsorted(pos, all_vals, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, all_vals, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def eval_model(model: nn.Module, X: np.ndarray, y: np.ndarray,
               device, batch=4096) -> tuple[np.ndarray, dict]:
    model.eval()
    proba_chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            logits = model(xb).squeeze(-1)
            proba_chunks.append(torch.sigmoid(logits).cpu().numpy())
    proba = np.concatenate(proba_chunks).astype(np.float32)
    y_pred = (proba >= 0.5).astype(np.int8)
    yt = y.astype(np.int8)
    m = {
        "auc": float(roc_auc_score(yt, proba)),
        "precision": float(precision_score(yt, y_pred, zero_division=0)),
        "recall":    float(recall_score(yt, y_pred, zero_division=0)),
        "f1":        float(f1_score(yt, y_pred, zero_division=0)),
        "ks":        ks_statistic(yt, proba),
        "threshold": 0.5,
        "n":         int(len(y)),
        "n_pos":     int(yt.sum()),
    }
    return proba, m


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_one_seed(dataset: str, mode: str, seed: int,
                   checkpoint: Path | None,
                   max_epochs: int = 80, patience: int = 10,
                   batch: int = 512, lr: float = 1e-3, weight_decay: float = 0.01,
                   ) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr = load_split(dataset, "train")
    Xva, yva = load_split(dataset, "val")
    Xte, yte = load_split(dataset, "test")
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)
    print(f"  shapes: train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}  device={device}")

    pos_rate = float(ytr.mean())
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate], device=device,
                              dtype=torch.float32)

    cfg = MLPConfig(d_in=Xtr.shape[1])
    model = TabularMLP(cfg).to(device)

    # --- mode-specific setup ---
    if mode == "full":
        total_params = count_total(model)
        trainable_params = count_trainable(model)
    elif mode == "lora":
        assert checkpoint is not None and checkpoint.exists(), \
            f"LoRA mode requires --checkpoint; got {checkpoint}"
        sd = torch.load(checkpoint, map_location=device, weights_only=True)
        # Filter only matching keys (the head/body should match for same-arch base)
        model.load_state_dict(sd, strict=True)
        trainable_params = apply_lora(model, r=8, alpha=16, which=(0, 1))
        model.to(device)
        total_params = count_total(model)
    else:
        raise ValueError(f"unknown mode {mode}")
    print(f"  mode={mode}  total={total_params:,}  trainable={trainable_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")

    train_ld, val_ld = make_loaders(Xtr, ytr, Xva, yva, batch=batch, seed=seed)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -1.0
    best_state = None
    epochs_since_improve = 0
    history = []
    t0 = time.time()
    tracemalloc.start()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_ld:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            logit = model(xb).squeeze(-1)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            n_batches += 1
        train_loss /= max(n_batches, 1)

        _, m_val = eval_model(model, Xva, yva, device)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_auc": round(m_val["auc"], 5)})
        improved = m_val["auc"] > best_val_auc + 1e-5
        if improved:
            best_val_auc = m_val["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        marker = " *" if improved else ""
        print(f"    epoch {epoch:3d}: loss={train_loss:.4f}  val_auc={m_val['auc']:.4f}{marker}")
        if epochs_since_improve >= patience:
            print(f"    [early-stop] no improvement for {patience} epochs")
            break

    train_seconds = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Reload best
    if best_state is not None:
        model.load_state_dict(best_state)
    proba_te, m_te = eval_model(model, Xte, yte, device)
    _, m_va = eval_model(model, Xva, yva, device)

    out = {
        "dataset": dataset,
        "mode": mode,
        "seed": int(seed),
        "device": str(device),
        "checkpoint_in": str(checkpoint) if checkpoint else None,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_fraction": float(trainable_params) / float(total_params),
        "train_seconds": round(train_seconds, 3),
        "peak_memory_mb": round(peak_mem / 1e6, 2),
        "epochs_run": int(history[-1]["epoch"]),
        "best_val_auc": float(best_val_auc),
        "val_auc": float(m_va["auc"]),
        **m_te,
        "history": history,
    }

    # Save outputs
    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{mode}_seed{seed}.json"
    json_path.write_text(json.dumps(out, indent=2))
    npz_path = out_dir / f"{mode}_seed{seed}_preds.npz"
    np.savez_compressed(npz_path, y_test=yte.astype(np.int8), proba_test=proba_te)
    pt_path = out_dir / f"{mode}_seed{seed}.pt"
    torch.save(model.state_dict(), pt_path)

    print(f"  -> {json_path.name}  test_AUC={m_te['auc']:.4f}  "
          f"trainable={trainable_params:,}  time={train_seconds:.1f}s")

    del model, opt, train_ld, val_ld
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out


def summarise(dataset: str) -> None:
    rows = []
    for js in sorted((RESULTS_DIR / dataset).glob("*_seed*.json")):
        d = json.loads(js.read_text())
        # Only include MLP-mode rows (full / lora) here
        if d.get("mode") not in ("full", "lora"):
            continue
        rows.append(d)
    if not rows:
        return
    df = pd.DataFrame(rows)
    agg = df.groupby("mode").agg(
        auc_mean=("auc", "mean"),
        auc_std =("auc", "std"),
        f1_mean =("f1",  "mean"),
        ks_mean =("ks",  "mean"),
        train_mean=("train_seconds", "mean"),
        trainable_params=("trainable_params", "mean"),
        total_params=("total_params", "mean"),
    ).reset_index()
    out_path = RESULTS_DIR / dataset / "mlp_summary.csv"
    agg.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n[mlp summary] {out_path}")
    print(agg.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["home_credit", "german"], required=True)
    ap.add_argument("--mode", choices=["full", "lora"], required=True)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Required for --mode lora: path to a pre-trained .pt")
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    if args.mode == "lora" and args.checkpoint is None:
        ap.error("--mode lora requires --checkpoint")

    for seed in args.seeds:
        print(f"\n=== {args.mode.upper()} MLP on {args.dataset}  seed={seed} ===")
        train_one_seed(
            dataset=args.dataset,
            mode=args.mode,
            seed=seed,
            checkpoint=args.checkpoint,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch=args.batch,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    summarise(args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
