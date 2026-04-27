"""
Step 6: UMAP chemical space map + prismcology heatmap.

Usage:
    python utils/visualise.py

Inputs:
    outputs/embeddings.npy
    outputs/embedding_ids.csv
    outputs/coconut_scored.csv   (optional, for full chemical space)
    outputs/top500_candidates.csv

Outputs:
    outputs/umap_coords.npy
    outputs/figure_umap_heatmap.png    ← main paper figure
    outputs/figure_top10_structures.png ← structure grid (requires rdkit drawing)
"""

import os, sys, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ── UMAP ──────────────────────────────────────────────────────────────────────

def run_umap(embeddings: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1) -> np.ndarray:
    """
    Project high-dim embeddings to 2D with UMAP.
    ~15-30 min for 10k molecules on CPU. Use GPU (cuML) if available.
    """
    try:
        from cuml.manifold import UMAP as cuUMAP
        log.info("Using GPU UMAP (cuML) ...")
        reducer = cuUMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         n_components=2, metric="cosine", random_state=42)
    except ImportError:
        log.info("Using CPU UMAP (umap-learn) ...")
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            n_components=2, metric="cosine", random_state=42,
                            verbose=True)

    coords = reducer.fit_transform(embeddings)
    return coords


# ── Main figures ──────────────────────────────────────────────────────────────

def plot_umap_heatmap(
    coords: np.ndarray,
    ps_scores: np.ndarray,
    top500_mask: np.ndarray,
    save_path: str = "outputs/figure_umap_heatmap.png",
):
    """
    Figure 1: UMAP chemical space coloured by PS score.
    Top-500 compounds highlighted with a border.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: all screened compounds (grey, small)
    background_mask = ~top500_mask
    ax.scatter(
        coords[background_mask, 0],
        coords[background_mask, 1],
        c="lightgrey",
        s=2,
        alpha=0.3,
        linewidths=0,
        label="All screened",
        rasterized=True,
    )

    # Foreground: top-500 coloured by PS
    norm = Normalize(vmin=ps_scores[top500_mask].min(),
                     vmax=ps_scores[top500_mask].max())
    sc = ax.scatter(
        coords[top500_mask, 0],
        coords[top500_mask, 1],
        c=ps_scores[top500_mask],
        cmap="viridis",
        s=18,
        alpha=0.9,
        linewidths=0.3,
        edgecolors="white",
        norm=norm,
        label="Top-500 candidates",
        zorder=3,
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Polypharmacology Score (PS)", fontsize=11)

    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.set_title(
        "Chemical space of screened COCONUT natural products\n"
        "coloured by Polypharmacology Score (TR, Phase 1)",
        fontsize=12,
        fontweight="normal",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"✓ Saved {save_path}")


def plot_top10_structures(
    top_df: pd.DataFrame,
    save_path: str = "outputs/figure_top10_structures.png",
    n: int = 10,
):
    """Draw the top-N structures with their PS scores."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        from PIL import Image
        import io
    except ImportError:
        log.warning("RDKit drawing or PIL not available. Skipping structure figure.")
        return

    top = top_df.head(n)
    mols = []
    legends = []
    for _, row in top.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            mols.append(mol)
            name = row.get("name", row.get("coconut_id", f"rank_{row['rank']}"))
            legends.append(f"{name}\nPS={row['PS']:.3f}  P(TR)={row['P_TR']:.3f}")

    if not mols:
        log.warning("No valid molecules for structure figure.")
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(300, 250),
        legends=legends,
        returnPNG=False,
    )
    img.save(save_path)
    log.info(f"✓ Saved {save_path}")


def plot_ps_distribution(
    scored_df: pd.DataFrame,
    save_path: str = "outputs/figure_ps_distribution.png",
):
    """PS score distribution with top-500 threshold marked."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(scored_df["PS"], bins=80, color="#1D9E75", alpha=0.7, edgecolor="none")

    threshold = scored_df["PS"].nlargest(500).min()
    ax.axvline(threshold, color="#D85A30", linestyle="--", linewidth=1.5,
               label=f"Top-500 threshold (PS={threshold:.3f})")

    ax.set_xlabel("Polypharmacology Score (PS)", fontsize=12)
    ax.set_ylabel("Number of compounds", fontsize=12)
    ax.set_title("PS distribution across screened COCONUT compounds", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"✓ Saved {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load embeddings
    emb_path = "outputs/embeddings.npy"
    id_path  = "outputs/embedding_ids.csv"
    if not os.path.exists(emb_path):
        log.error(f"Embeddings not found at {emb_path}. Run screen_coconut.py first.")
        sys.exit(1)

    embeddings = np.load(emb_path)
    id_df      = pd.read_csv(id_path)
    scored_df  = pd.read_csv("outputs/coconut_scored.csv")
    top500_df  = pd.read_csv("outputs/top500_candidates.csv")

    log.info(f"Embeddings: {embeddings.shape}")
    log.info(f"Scored molecules: {len(scored_df)}")

    # UMAP
    umap_path = "outputs/umap_coords.npy"
    if os.path.exists(umap_path):
        log.info("Loading cached UMAP coordinates ...")
        coords = np.load(umap_path)
    else:
        log.info("Running UMAP (this may take 15–30 min on CPU) ...")
        coords = run_umap(embeddings)
        np.save(umap_path, coords)
        log.info(f"✓ Saved UMAP coords: {umap_path}")

    # Align id_df with embeddings
    ps_scores = id_df["PS"].values

    # Mark which rows are in top-500
    top500_smiles = set(top500_df["smiles"].tolist())
    top500_mask   = np.array([s in top500_smiles for s in id_df["smiles"]])

    log.info(f"Top-500 mask: {top500_mask.sum()} / {len(top500_mask)}")

    # Generate figures
    plot_umap_heatmap(coords, ps_scores, top500_mask)
    plot_ps_distribution(scored_df)
    plot_top10_structures(top500_df)

    log.info("\n── All figures saved to outputs/ ─────────────────────────────")
    for f in os.listdir("outputs"):
        if f.endswith(".png"):
            log.info(f"  {f}")


if __name__ == "__main__":
    main()
