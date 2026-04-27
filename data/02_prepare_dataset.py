"""
Step 2: Merge actives + inactives, run scaffold split, save labelled_dataset.csv
Run AFTER 01_fetch_chembl.py.

Output:
  data/labelled_dataset.csv      -- full merged dataset with split column
  data/train.csv / val.csv / test.csv
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import logging, random

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15 (remainder)


# ── Scaffold split (Bemis-Murcko) ────────────────────────────────────────────
# This is CRITICAL. A random split leaks similar scaffolds between train/test
# and gives falsely optimistic AUC. Scaffold split tests true generalisation.

def get_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        return scaffold
    except Exception:
        return None


def scaffold_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float   = VAL_FRAC,
    seed: int         = SEED,
) -> pd.DataFrame:
    """
    Assign each molecule to train/val/test such that all molecules sharing
    the same Murcko scaffold go to the same split.
    """
    log.info("Computing Murcko scaffolds ...")
    df = df.copy()
    df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)
    df.loc[df["scaffold"].isna(), "scaffold"] = "NO_SCAFFOLD"

    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in zip(df.index, df["scaffold"]):
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffolds by size descending (large scaffolds assigned first)
    scaffold_sets = sorted(
        scaffold_to_indices.values(), key=len, reverse=True
    )

    n = len(df)
    train_cutoff = int(n * train_frac)
    val_cutoff   = int(n * (train_frac + val_frac))

    rng = random.Random(seed)
    rng.shuffle(scaffold_sets)

    train_idx, val_idx, test_idx = [], [], []
    for indices in scaffold_sets:
        if len(train_idx) < train_cutoff:
            train_idx.extend(indices)
        elif len(val_idx) < (val_cutoff - train_cutoff):
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    df["split"] = "test"
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx,   "split"] = "val"

    counts = df["split"].value_counts()
    log.info(f"Split → train:{counts.get('train',0)} val:{counts.get('val',0)} test:{counts.get('test',0)}")
    return df


def main():
    actives   = pd.read_csv("data/raw_actives.csv")
    inactives = pd.read_csv("data/raw_inactives.csv")

    # Combine and keep only what we need
    cols = ["molecule_chembl_id", "canonical_smiles", "inchikey",
            "standard_value", "target", "label"]
    combined = pd.concat([
        actives[cols],
        inactives[cols],
    ], ignore_index=True)

    log.info(f"Combined: {len(combined)} molecules ({combined['label'].sum():.0f} active)")

    # Scaffold split
    combined = scaffold_split(combined)

    # Save full dataset
    combined.to_csv("data/labelled_dataset.csv", index=False)
    log.info("✓ Saved data/labelled_dataset.csv")

    # Save individual splits
    for split in ["train", "val", "test"]:
        subset = combined[combined["split"] == split]
        subset.to_csv(f"data/{split}.csv", index=False)
        pos_rate = subset["label"].mean()
        log.info(f"  {split}: {len(subset)} molecules, {pos_rate:.1%} positive")

    # Summary
    print("\n── Split summary ────────────────────────────────────────────────")
    print(combined.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
