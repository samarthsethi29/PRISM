import os
import sys
import logging
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit.Chem.inchi import MolToInchiKey
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Target essentiality & selectivity weights ─────────────────────────────────
TARGET_WEIGHTS = {
    "TR": {
        "essentiality": 0.95,
        "selectivity": 0.90,
    },
    # Add PTR1 and CYP51 when you calculate them
    # "PTR1": {"essentiality": 0.85, "selectivity": 0.75},
    # "CYP51": {"essentiality": 0.90, "selectivity": 0.80},
}

# ── Filters: Original + Lipinski (tuned to get ~10k-15k final) ───────────────
COCONUT_FILTERS = {
    "mw_min": 180,
    "mw_max": 580,        # Slightly stricter than original 700
    "logp_max": 5.0,      # Same as original
    "hbd_max": 5,         # Lipinski
    "hba_max": 10,        # Lipinski
    "rotatable_max": 10,  # Lipinski-like
    "min_n": 1,           # Original requirement
    "max_rings": 6,       # Stricter than original 8
    "min_heavy_atoms": 15,
}

def passes_coconut_filter(mol) -> bool:
    """Combined original + Lipinski filters"""
    if mol is None:
        return False
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_heavy = mol.GetNumHeavyAtoms()

        return (
            COCONUT_FILTERS["mw_min"] <= mw <= COCONUT_FILTERS["mw_max"]
            and logp <= COCONUT_FILTERS["logp_max"]
            and hbd <= COCONUT_FILTERS["hbd_max"]
            and hba <= COCONUT_FILTERS["hba_max"]
            and n_rot <= COCONUT_FILTERS["rotatable_max"]
            and n_n >= COCONUT_FILTERS["min_n"]
            and n_rings <= COCONUT_FILTERS["max_rings"]
            and n_heavy >= COCONUT_FILTERS["min_heavy_atoms"]
        )
    except Exception:
        return False


def build_pains_filter():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


def prismcology_score(predictions: dict[str, float]) -> float:
    """Calculate Polypharmacology Score (PS)"""
    probs = np.array(list(predictions.values()))
    geo_mean = np.exp(np.log(np.clip(probs, 1e-9, 1.0)).mean())
    ess_sel_sum = sum(
        TARGET_WEIGHTS.get(n, {}).get("essentiality", 1.0) *
        TARGET_WEIGHTS.get(n, {}).get("selectivity", 1.0)
        for n in predictions.keys()
    )
    return geo_mean * ess_sel_sum


def process_batch(batch_df: pd.DataFrame, pains_catalog) -> pd.DataFrame:
    """Process one batch safely"""
    if batch_df.empty:
        return pd.DataFrame()

    # SMILES → Mol objects
    mols = [Chem.MolFromSmiles(smi) for smi in batch_df["smiles"]]

    # Apply physicochemical + Lipinski filter
    filter_mask = [passes_coconut_filter(m) for m in mols]
    filtered_df = batch_df[filter_mask].copy().reset_index(drop=True)
    filtered_mols = [m for m, keep in zip(mols, filter_mask) if keep]

    if filtered_df.empty:
        return pd.DataFrame()

    filtered_df["mol_obj"] = filtered_mols

    # PAINS filter
    is_pains = []
    pains_reason = []
    for mol in tqdm(filtered_mols, desc="PAINS check", leave=False):
        if mol is None:
            is_pains.append(True)
            pains_reason.append("INVALID")
            continue
        entry = pains_catalog.GetFirstMatch(mol)
        if entry:
            is_pains.append(True)
            pains_reason.append(entry.GetDescription())
        else:
            is_pains.append(False)
            pains_reason.append("")

    filtered_df["is_pains"] = is_pains
    filtered_df["pains_reason"] = pains_reason

    # Remove PAINS compounds
    clean = filtered_df[~filtered_df["is_pains"]].copy()

    # InChIKey for uniqueness
    clean["inchikey"] = clean["mol_obj"].apply(
        lambda m: MolToInchiKey(m) if m is not None else None
    )
    clean = clean.dropna(subset=["inchikey"]).drop_duplicates(subset="inchikey")

    # Drop temporary mol_obj column
    if "mol_obj" in clean.columns:
        clean = clean.drop(columns=["mol_obj"])

    return clean


if __name__ == "__main__":
    COCONUT_PATH = "data/COCONUT_DB.csv"
    OUTPUT_PATH = "data/coconut_filtered.csv"
    BATCH_SIZE = 4000          # Safe batch size for MacBook Air

    if not os.path.exists(COCONUT_PATH):
        log.error(f"File {COCONUT_PATH} not found. Please place your COCONUT CSV here.")
        sys.exit(1)

    log.info(f"Starting full processing of COCONUT (~738k molecules)")
    log.info("Filters: Original + Lipinski rules (targeting ~10k-15k final compounds)")

    pains_catalog = build_pains_filter()
    header_written = False
    total_passed = 0

    # Read CSV in chunks
    chunk_iter = pd.read_csv(COCONUT_PATH, chunksize=BATCH_SIZE)

    for chunk_num, chunk in enumerate(tqdm(chunk_iter, desc="Processing chunks")):
        # Find SMILES column (flexible)
        smiles_col = next((c for c in chunk.columns if "smiles" in c.lower()), None)
        if smiles_col is None:
            log.error(f"No SMILES column found. Columns: {list(chunk.columns)}")
            sys.exit(1)

        chunk = chunk.rename(columns={smiles_col: "smiles"}).copy()

        processed = process_batch(chunk, pains_catalog)

        if not processed.empty:
            mode = 'w' if not header_written else 'a'
            processed.to_csv(OUTPUT_PATH, mode=mode, header=not header_written, index=False)
            header_written = True
            total_passed += len(processed)

        if chunk_num % 25 == 0 and total_passed > 0:
            log.info(f"Progress: {total_passed:,} molecules passed filters so far")

    # Final summary
    if os.path.exists(OUTPUT_PATH):
        final_df = pd.read_csv(OUTPUT_PATH)
        final_count = len(final_df)
        log.info(f"✓ FINISHED PROCESSING!")
        log.info(f"Final count: {final_count:,} molecules saved to {OUTPUT_PATH}")
        log.info(f"Reduction rate: {final_count / 738000:.2%} of original COCONUT")
    else:
        log.warning("No molecules passed all filters.")

    # PS sanity check
    ps = prismcology_score({"TR": 0.85})
    print(f"\nPS sanity check: P(TR)=0.85 → PS={ps:.4f} ✓")
def top_scaffolds(df, smiles_col="smiles", n=10):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffolds = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            scaffolds.append(None)
            continue
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds.append(s)
        except Exception:
            scaffolds.append(None)
    df = df.copy()
    df["scaffold"] = scaffolds
    counts = (
        df.groupby("scaffold")["PS"]
        .agg(["count", "mean", "max"])
        .rename(columns={"count": "n_analogs", "mean": "mean_PS", "max": "max_PS"})
        .sort_values("n_analogs", ascending=False)
        .head(n)
        .reset_index()
    )
    return counts
