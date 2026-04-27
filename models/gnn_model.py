"""
Multi-Task GNN for PRISM target activity prediction.

Architecture:
  • 4× GINEConv layers (GIN + edge features) with batch norm
  • Global mean + max pooling → 2×hidden_dim vector
  • Multi-task head: one sigmoid output per target (Phase 1 = TR only)

Designed so adding more targets in batch 2 is a one-line change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch

NUM_ATOM_FEATURES = 7
NUM_EDGE_FEATURES = 8

# Target list — add PTR1/CYP51 here for batch 2
TARGET_NAMES = ["TR"]


class MultiTargetGNN(nn.Module):
    """
    Args:
        hidden_dim:   width of GIN hidden layers (default 256)
        num_layers:   number of GINEConv message-passing layers (default 4)
        dropout:      dropout probability (default 0.2)
        target_names: list of target names (one output head each)
    """

    def __init__(
        self,
        hidden_dim:   int       = 256,
        num_layers:   int       = 4,
        dropout:      float     = 0.2,
        target_names: list[str] = None,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.dropout      = dropout
        self.target_names = target_names or TARGET_NAMES

        # ── Input projection ─────────────────────────────────────────────────
        self.atom_proj = nn.Linear(NUM_ATOM_FEATURES, hidden_dim)
        self.edge_proj = nn.Linear(NUM_EDGE_FEATURES, hidden_dim)

        # ── GINEConv layers ──────────────────────────────────────────────────
        # Each GINEConv needs an MLP for its ε-function
        self.convs  = nn.ModuleList()
        self.bns    = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ── Global pooling → graph embedding ─────────────────────────────────
        # Concatenate mean + max → 2×hidden_dim
        pool_dim = hidden_dim * 2

        # ── Shared MLP before task heads ─────────────────────────────────────
        self.shared_mlp = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Per-target output heads ───────────────────────────────────────────
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, 1)
            for name in self.target_names
        })

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, data: Batch) -> dict[str, torch.Tensor]:
        """
        Returns dict: {target_name: logits (shape [B, 1])}
        Use logits with BCEWithLogitsLoss; apply sigmoid for inference.
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Project to hidden_dim
        x         = F.relu(self.atom_proj(x))
        edge_attr = F.relu(self.edge_proj(edge_attr))

        # Message passing
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling
        x_mean = global_mean_pool(x, batch)   # [B, hidden_dim]
        x_max  = global_max_pool(x, batch)    # [B, hidden_dim]
        x_pool = torch.cat([x_mean, x_max], dim=1)  # [B, 2*hidden_dim]

        # Shared MLP
        h = self.shared_mlp(x_pool)   # [B, hidden_dim]

        # Per-target logits
        return {name: self.heads[name](h) for name in self.target_names}

    def get_embedding(self, data: Batch) -> torch.Tensor:
        """
        Return the 256-dim molecular embedding (after shared MLP).
        Used for UMAP visualisation in Step 5.
        """
        self.eval()
        with torch.no_grad():
            x, edge_index, edge_attr, batch = (
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            x         = F.relu(self.atom_proj(x))
            edge_attr = F.relu(self.edge_proj(edge_attr))
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index, edge_attr)
                x = bn(x)
                x = F.relu(x)
            x_mean = global_mean_pool(x, batch)
            x_max  = global_max_pool(x, batch)
            x_pool = torch.cat([x_mean, x_max], dim=1)
            return self.shared_mlp(x_pool)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Weighted BCE loss ─────────────────────────────────────────────────────────

class WeightedMultiTargetLoss(nn.Module):
    """
    Weighted Binary Cross-Entropy summed across all targets.
    pos_weights: dict {target_name: float}  (num_neg / num_pos)
    """

    def __init__(self, pos_weights: dict[str, float], device: torch.device):
        super().__init__()
        self.criteria = {}
        for name, w in pos_weights.items():
            pw = torch.tensor([w], dtype=torch.float, device=device)
            self.criteria[name] = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total_loss = 0.0
        for name, criterion in self.criteria.items():
            total_loss += criterion(logits[name].view(-1), targets[name].view(-1))
        return total_loss


# ── Quick architecture test ───────────────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    def dummy_molecule(n_atoms=12, n_bonds=13):
        x          = torch.randn(n_atoms, NUM_ATOM_FEATURES)
        src        = torch.randint(0, n_atoms, (n_bonds * 2,))
        dst        = torch.randint(0, n_atoms, (n_bonds * 2,))
        edge_index = torch.stack([src, dst])
        edge_attr  = torch.randn(n_bonds * 2, NUM_EDGE_FEATURES)
        y          = torch.tensor([1.0])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    batch = Batch.from_data_list([dummy_molecule() for _ in range(4)])

    model = MultiTargetGNN(hidden_dim=256, num_layers=4, dropout=0.2)
    print(f"Parameters: {model.count_parameters():,}")

    model.eval()
    with torch.no_grad():
        logits = model(batch)

    for name, out in logits.items():
        probs = torch.sigmoid(out)
        print(f"  {name}: logits {out.shape}, probs {probs.squeeze().tolist()}")

    emb = model.get_embedding(batch)
    print(f"  Embedding shape: {emb.shape}")   # [4, 256]
    print("\n✓ Architecture OK")
