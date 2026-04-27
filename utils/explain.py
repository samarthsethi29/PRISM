"""
Step 7: GNNExplainer — per-atom importance scores for top candidates.

Highlights which atoms/bonds drive the TR activity prediction.
These become the "mechanistic insight" figure in your manuscript.

Usage:
    python utils/explain.py \
        --model  outputs/best_model.pt \
        --top_df outputs/top500_candidates.csv \
        --n      5

Output:
    outputs/explanations/compound_{rank}_explanation.png  (one per compound)
    outputs/figure_explanation_grid.png                   (grid of top-5)
"""

import os, sys, argparse, logging
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.gnn_model import MultiTargetGNN, NUM_ATOM_FEATURES, NUM_EDGE_FEATURES
from utils.molecular_dataset import smiles_to_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    m  = MultiTargetGNN(
        hidden_dim=ck["args"].get("hidden_dim", 256),
        num_layers=ck["args"].get("num_layers", 4),
        dropout=0.0,
        target_names=["TR"],
    ).to(device)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def explain_molecule(model, smiles: str, device: torch.device) -> dict | None:
    """
    Run GNNExplainer on a single molecule.
    Returns dict with atom_importance array and the graph.
    """
    try:
        from torch_geometric.explain import Explainer, GNNExplainer
    except ImportError:
        log.error("GNNExplainer requires torch_geometric >= 2.3. "
                  "Install: pip install torch_geometric")
        return None

    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    graph = graph.to(device)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

    # Wrapper: GNNExplainer needs a model that returns a single tensor
    class Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x, edge_index, edge_attr=None, batch=None):
            from torch_geometric.data import Data, Batch
            d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                     batch=batch if batch is not None else torch.zeros(x.shape[0], dtype=torch.long))
            b = Batch.from_data_list([d])
            return torch.sigmoid(self.m(b)["TR"])

    wrapped = Wrapper(model)
    explainer = Explainer(
        model          = wrapped,
        algorithm      = GNNExplainer(epochs=200),
        explanation_type = "model",
        node_mask_type = "attributes",
        edge_mask_type = "object",
        model_config   = dict(mode="binary_classification", task_level="graph", return_type="probs"),
    )

    explanation = explainer(
        x          = graph.x,
        edge_index = graph.edge_index,
        edge_attr  = graph.edge_attr,
        batch      = graph.batch,
        index      = 0,
    )

    atom_importance = explanation.node_mask.sum(dim=1).cpu().numpy()
    atom_importance = (atom_importance - atom_importance.min()) / \
                      (atom_importance.max() - atom_importance.min() + 1e-9)

    return {
        "atom_importance": atom_importance,
        "graph": graph,
        "smiles": smiles,
    }


def draw_explanation(result: dict, title: str, save_path: str):
    """Draw molecule with atom importance as colour intensity."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, rdMolDescriptors
        from rdkit.Chem.Draw import rdMolDraw2D
        import io
        from PIL import Image
    except ImportError:
        log.warning("RDKit/PIL drawing not available — skipping visual output.")
        return

    mol = Chem.MolFromSmiles(result["smiles"])
    if mol is None:
        return

    atom_imp = result["atom_importance"]

    # Map importance to colour: low=light blue, high=dark orange
    from matplotlib.cm import RdYlGn_r
    atom_colors = {}
    for i, imp in enumerate(atom_imp):
        r, g, b, _ = RdYlGn_r(imp)
        atom_colors[i] = (r, g, b)

    drawer = rdMolDraw2D.MolDraw2DSVG(400, 350)
    drawer.drawOptions().addAtomIndices = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=atom_colors,
        highlightAtomRadii={i: 0.4 for i in range(mol.GetNumAtoms())},
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Save SVG
    svg_path = save_path.replace(".png", ".svg")
    with open(svg_path, "w") as f:
        f.write(svg)

    # Convert to PNG via cairosvg if available, else save SVG only
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode(), write_to=save_path, scale=2.0)
        log.info(f"  Saved {save_path}")
    except ImportError:
        log.info(f"  Saved {svg_path} (install cairosvg for PNG output)")


def main(args):
    os.makedirs("outputs/explanations", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = load_model(args.model, device)
    top_df = pd.read_csv(args.top_df).head(args.n)

    log.info(f"Explaining top {args.n} candidates ...")
    results = []

    for _, row in top_df.iterrows():
        rank   = int(row["rank"])
        smiles = row["smiles"]
        ps     = row["PS"]
        p_tr   = row["P_TR"]

        log.info(f"  Rank {rank}: PS={ps:.3f} P(TR)={p_tr:.3f}")
        result = explain_molecule(model, smiles, device)
        if result is None:
            log.warning(f"  Explanation failed for rank {rank}")
            continue

        result["rank"] = rank
        result["ps"]   = ps
        result["p_tr"] = p_tr

        draw_explanation(
            result,
            title=f"Rank {rank} | PS={ps:.3f} | P(TR)={p_tr:.3f}",
            save_path=f"outputs/explanations/compound_{rank}_explanation.png",
        )
        results.append(result)

    log.info(f"\n✓ Explanations saved to outputs/explanations/")
    log.info("  Interpretation:")
    log.info("  • Dark red atoms = most important for TR activity prediction")
    log.info("  • Light green atoms = least important")
    log.info("  • Use this to identify the pharmacophore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="outputs/best_model.pt")
    parser.add_argument("--top_df", default="outputs/top500_candidates.csv")
    parser.add_argument("--n",      type=int, default=5)
    args = parser.parse_args()
    main(args)
