"""
Optuna hyperparameter sweep.
Run AFTER confirming training loop works (models/train.py).

Usage:
    python models/hparam_sweep.py --csv data/labelled_dataset.csv --n_trials 20

Saves: outputs/best_hparams.json
"""

import argparse
import os
import sys
import json
import logging

import optuna
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.gnn_model import MultiTargetGNN, WeightedMultiTargetLoss
from utils.molecular_dataset import MolecularDataset

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, full_ds, split_map, pos_weights, device):
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 3, 5)
    dropout    = trial.suggest_float("dropout", 0.1, 0.3, step=0.1)
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    target_names = ["TR"]

    train_ds = full_ds.index_select(split_map["train"])
    val_ds   = full_ds.index_select(split_map["val"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = MultiTargetGNN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        target_names=target_names,
    ).to(device)

    loss_fn   = WeightedMultiTargetLoss(pos_weights, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Train for 30 epochs per trial
    for epoch in range(30):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            labels = {"TR": batch.y.squeeze()}
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Pruning (Optuna can stop bad trials early)
        if epoch % 10 == 9:
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out   = torch.sigmoid(model(batch)["TR"]).squeeze().cpu().numpy()
                    lbl   = batch.y.squeeze().cpu().numpy()
                    all_logits.extend(out.tolist())
                    all_labels.extend(lbl.tolist())
            y_true  = np.array(all_labels)
            y_score = np.array(all_logits)
            if len(np.unique(y_true)) > 1:
                auprc = average_precision_score(y_true, y_score)
                trial.report(auprc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    # Final val AUPRC
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out   = torch.sigmoid(model(batch)["TR"]).squeeze().cpu().numpy()
            lbl   = batch.y.squeeze().cpu().numpy()
            all_logits.extend(out.tolist())
            all_labels.extend(lbl.tolist())

    y_true  = np.array(all_labels)
    y_score = np.array(all_logits)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_score)


def main(args):
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.csv)
    train_df = df[df["split"] == "train"]
    n_pos = train_df["label"].sum()
    n_neg = len(train_df) - n_pos
    pos_weights = {"TR": float(n_neg / max(n_pos, 1))}

    full_ds = MolecularDataset(root="data/processed", csv_path=args.csv)
    split_map = {"train": [], "val": [], "test": []}
    for i in range(len(full_ds)):
        split_map[full_ds.get(i).split_label].append(i)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, full_ds, split_map, pos_weights, device),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n── Best trial ──────────────────────────────────────────────────")
    print(f"  AUPRC: {best.value:.4f}")
    print(f"  Params: {best.params}")

    with open("outputs/best_hparams.json", "w") as f:
        json.dump({"auprc": best.value, "params": best.params}, f, indent=2)
    print("✓ Saved outputs/best_hparams.json")
    print("\nNow retrain with best params:")
    p = best.params
    print(f"  python models/train.py "
          f"--hidden_dim {p['hidden_dim']} "
          f"--num_layers {p['num_layers']} "
          f"--dropout {p['dropout']} "
          f"--lr {p['lr']:.5f} "
          f"--batch_size {p['batch_size']} "
          f"--epochs 100")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",      default="data/labelled_dataset.csv")
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()
    main(args)
