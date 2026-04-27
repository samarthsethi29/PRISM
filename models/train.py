import argparse
import os
import sys
import time
import logging
import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gnn_model import MultiTargetGNN, WeightedMultiTargetLoss
from utils.molecular_dataset import MolecularDataset, SPLIT_MAP_INV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def evaluate(model, loader, loss_fn, device, target_names):
    model.eval()
    total_loss = 0.0
    all_logits = {n: [] for n in target_names}
    all_labels = {n: [] for n in target_names}

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model(batch)
            # always keep as 1D tensor regardless of batch size
            y = batch.y.view(-1)
            labels = {n: y for n in target_names}

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            for n in target_names:
                preds = torch.sigmoid(logits[n]).view(-1).cpu().tolist()
                trues = labels[n].cpu().tolist()
                # tolist() on a 0-dim tensor returns a float, wrap it
                if isinstance(preds, float):
                    preds = [preds]
                if isinstance(trues, float):
                    trues = [trues]
                all_logits[n].extend(preds)
                all_labels[n].extend(trues)

    metrics = {"loss": total_loss / max(len(loader), 1)}
    for n in target_names:
        yt = np.array(all_labels[n])
        ys = np.array(all_logits[n])
        yp = (ys >= 0.5).astype(int)
        if len(np.unique(yt)) < 2:
            metrics[f"{n}_auc"]   = float("nan")
            metrics[f"{n}_auprc"] = float("nan")
            metrics[f"{n}_mcc"]   = float("nan")
        else:
            metrics[f"{n}_auc"]   = roc_auc_score(yt, ys)
            metrics[f"{n}_auprc"] = average_precision_score(yt, ys)
            metrics[f"{n}_mcc"]   = matthews_corrcoef(yt, yp)
    return metrics


def train(args):
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    df           = pd.read_csv(args.csv)
    target_names = ["TR"]
    train_df     = df[df["split"] == "train"]
    n_pos        = int(train_df["label"].sum())
    n_neg        = len(train_df) - n_pos
    pos_weights  = {"TR": float(n_neg / max(n_pos, 1))}
    log.info(f"pos_weight[TR] = {pos_weights['TR']:.2f}  ({n_pos} actives, {n_neg} inactives)")

    log.info("Loading dataset ...")
    full_ds = MolecularDataset(
        root     = os.path.join(os.path.dirname(args.csv), "processed"),
        csv_path = args.csv,
    )
    log.info(f"Total graphs: {len(full_ds)}")

    split_map = {"train": [], "val": [], "test": []}
    for i in range(len(full_ds)):
        g   = full_ds.get(i)
        key = SPLIT_MAP_INV.get(int(g.split_idx.item()), "train")
        split_map[key].append(i)

    log.info(f"Split sizes → train:{len(split_map['train'])} "
             f"val:{len(split_map['val'])} test:{len(split_map['test'])}")

    train_ds = full_ds.index_select(split_map["train"])
    val_ds   = full_ds.index_select(split_map["val"])
    test_ds  = full_ds.index_select(split_map["test"])

    # drop_last=True prevents single-sample batches which cause shape issues
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, drop_last=False)

    model = MultiTargetGNN(
        hidden_dim   = args.hidden_dim,
        num_layers   = args.num_layers,
        dropout      = args.dropout,
        target_names = target_names,
    ).to(device)
    log.info(f"Parameters: {model.count_parameters():,}")

    loss_fn   = WeightedMultiTargetLoss(pos_weights, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auprc = -1.0
    history        = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            batch  = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            labels = {"TR": batch.y.view(-1)}
            loss   = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        val_metrics   = evaluate(model, val_loader,  loss_fn, device, target_names)
        val_auprc     = float(val_metrics.get("TR_auprc", 0.0) or 0.0)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "args":        vars(args),
            }, "outputs/best_model.pt")

        row = {
            "epoch":        epoch,
            "train_loss":   round(train_loss, 4),
            "val_loss":     round(float(val_metrics["loss"]), 4),
            "val_TR_auc":   round(float(val_metrics.get("TR_auc",   0) or 0), 4),
            "val_TR_auprc": round(float(val_metrics.get("TR_auprc", 0) or 0), 4),
            "time_s":       round(time.time() - t0, 1),
        }
        history.append(row)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_AUC={val_metrics.get('TR_auc', float('nan')):.4f} | "
                f"val_AUPRC={val_auprc:.4f} | "
                f"best={best_val_auprc:.4f}"
            )

    # ── Test ─────────────────────────────────────────────────────────────────
    ck = torch.load("outputs/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    test_metrics = evaluate(model, test_loader, loss_fn, device, target_names)

    log.info("── Test results ─────────────────────────────────────────────")
    for k, v in test_metrics.items():
        log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open("outputs/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    pd.DataFrame(history).to_csv("outputs/training_log.csv", index=False)

    # ── Plot ─────────────────────────────────────────────────────────────────
    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(hist_df["epoch"], hist_df["train_loss"], label="Train", color="#1D9E75")
    ax.plot(hist_df["epoch"], hist_df["val_loss"],   label="Val",   color="#D85A30", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss curves")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(hist_df["epoch"], hist_df["val_TR_auc"],   label="AUC",   color="#1D9E75")
    ax.plot(hist_df["epoch"], hist_df["val_TR_auprc"], label="AUPRC", color="#1D9E75", linestyle="--")
    ax.axhline(0.75, color="gray", linestyle=":", alpha=0.5, label="Target=0.75")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Validation metrics")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("✓ Done. Outputs saved to outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="data/labelled_dataset.csv")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    train(args)