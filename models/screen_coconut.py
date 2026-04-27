import os, sys, argparse, logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gnn_model import MultiTargetGNN
from utils.molecular_dataset import smiles_to_graph
from utils.prismcology import prismcology_score, top_scaffolds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BATCH_SIZE = 256

def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    model = MultiTargetGNN(
        hidden_dim   = args.get("hidden_dim", 256),
        num_layers   = args.get("num_layers", 4),
        dropout      = 0.0,
        target_names = ["TR"],
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    log.info(f"Loaded model — best val AUPRC: {ck.get('val_metrics', {}).get('TR_auprc', 'N/A')}")
    return model


def main(args):
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    df = pd.read_csv(args.coconut).reset_index(drop=True)
    log.info(f"Loaded {len(df)} COCONUT molecules")

    model = load_model(args.model, device)

    all_probs  = []
    all_embs   = []
    valid_rows = []

    # Process in batches using positional index
    graphs_buf = []
    rows_buf   = []

    def flush(graphs_buf, rows_buf):
        if not graphs_buf:
            return
        batch = Batch.from_data_list(graphs_buf).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.sigmoid(logits["TR"]).view(-1).cpu().numpy()
            embs   = model.get_embedding(batch).cpu().numpy()
        for i, row_idx in enumerate(rows_buf):
            all_probs.append(float(probs[i]))
            all_embs.append(embs[i])
            valid_rows.append(row_idx)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Screening"):
        g = smiles_to_graph(str(row["smiles"]))
        if g is None:
            continue
        graphs_buf.append(g)
        rows_buf.append(i)
        if len(graphs_buf) == BATCH_SIZE:
            flush(graphs_buf, rows_buf)
            graphs_buf, rows_buf = [], []

    flush(graphs_buf, rows_buf)  # final partial batch

    log.info(f"Scored {len(valid_rows)} molecules successfully")

    # Build results dataframe
    results = df.iloc[valid_rows].copy().reset_index(drop=True)
    results["P_TR"] = all_probs
    results["PS"]   = results["P_TR"].apply(lambda p: prismcology_score({"TR": p}))
    results = results.sort_values("PS", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1

    log.info(f"PS range: {results['PS'].min():.4f} – {results['PS'].max():.4f}")
    log.info(f"Top 5 PS: {results['PS'].head(5).values}")

    # Save outputs
    results.to_csv(args.output, index=False)
    log.info(f"✓ Saved {args.output}")

    top500 = results.head(500).copy()
    top500.to_csv("outputs/top500_candidates.csv", index=False)
    log.info("✓ Saved outputs/top500_candidates.csv")

    top_scaf = top_scaffolds(top500, smiles_col="smiles", n=10)
    top_scaf.to_csv("outputs/top_scaffolds.csv", index=False)
    log.info("✓ Saved outputs/top_scaffolds.csv")
    print(top_scaf.to_string(index=False))

    # Save embeddings aligned to results order
    emb_matrix = np.array([all_embs[i] for i in results.index])
    np.save("outputs/embeddings.npy", emb_matrix)

    id_df = results[["smiles", "identifier", "P_TR", "PS", "rank"]].copy() \
        if "identifier" in results.columns \
        else results[["smiles", "P_TR", "PS", "rank"]].copy()
    id_df.to_csv("outputs/embedding_ids.csv", index=False)

    log.info(f"✓ Embeddings: {emb_matrix.shape} → outputs/embeddings.npy")
    print(f"\nTotal screened: {len(results):,}")
    print(f"Top-500 saved:  outputs/top500_candidates.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="outputs/best_model.pt")
    parser.add_argument("--coconut", default="data/coconut_filtered.csv")
    parser.add_argument("--output",  default="outputs/coconut_scored.csv")
    args = parser.parse_args()
    main(args)