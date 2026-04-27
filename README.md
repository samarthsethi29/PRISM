# PRISM
### Polypharmacology Ranking via Integrated Screening of natural Molecules

> **A multi task Graph Neural Network for resistance proof drug discovery from natural products**

[python 3.10](https://www.python.org/)
[pyg 2022.9+](https://pyg.org/)
[RDKit 2022.9+](https://www.rdkit.org/)

---

## What PRISM does

PRISM uses a **Graph Neural Network (GNN)** to screen natural product molecules against essential disease targets, scoring each molecule on its ability to hit *multiple targets simultaneously*. A pathogen would need to evolve mutations in all targets at once to develop resistance which is statistically near impossible. This is **resistance-proofing by design**.

**Phase 1 : Kinetoplastid parasites (Trypanosoma brucei, T. cruzi, Leishmania spp.):**
- Target: Trypanothione Reductase (TR)  essential enzyme absent in humans
- Screening library: 15,000 drug-like natural products from COCONUT (400,000 total)
- Test ROC-AUC: **1.0000** | Test AUPRC: **1.0000** | Test MCC: **0.8757**
- Top hits: sulfonamide-heterocycles  independently validated by anti-trypanosomal literature
- Screening time: **11 minutes on CPU**

---
## Why Poly-pharmacology is Essential for Modern Medicine

The traditional **"one drug, one target"** model is increasingly inadequate for treating complex diseases. Biological systems are robust and redundant; when a single protein is inhibited, the body or pathogen often finds an alternative pathway to bypass the drug’s effect.

* **Addressing Network Complexity:** Most diseases are polygenic. Poly-pharmacological agents target multiple nodes within a disease network, making it significantly harder for the biological system to compensate.
* **Overcoming Drug Resistance:** In infectious diseases, targeting multiple essential proteins simultaneously lowers the probability of a pathogen developing resistance through a single mutation.
* **Enhanced Efficacy & Synergy:** By hitting multiple targets, these compounds can achieve a greater therapeutic effect at lower concentrations, potentially reducing the risk of dose-related toxicity.
* **Simplified Treatment Regimens:** Creating a single molecule with multiple therapeutic actions reduces the "pill burden" on patients, leading to better compliance and clinical outcomes.

---
## The core idea

```
One molecule  ->  PRISM  ->  P(Target 1) = 0.87
                           P(Target 2) = 0.79    ->  PS Score = 0.823
                           P(Target 3) = 0.81
```

PRISM scores molecules with a novel **Polypharmacology Score (PS)**:

```
PS = geometric_mean(P₁, P₂, ..., Pₙ) × Σᵢ(Eᵢ × Sᵢ)
```

Where **Eᵢ** = target essentiality (from genomic databases) and **Sᵢ** = selectivity vs. human proteome. The geometric mean ensures balanced multi-target activity, a molecule hitting one target at 95% and two others at 5% scores far below one hitting all three at 70%.

---

## Why PRISM is different from existing tools

| Capability | PRISM | AutoDock/Vina | DeepChem | SwissTargetPrediction | ChemProp |
|-----------|-------|--------------|----------|----------------------|----------|
| Integrated multi-target PS score | ✓ | ✗ | ✗ | ✗ | ✗ |
| GINEConv edge-feature-aware GNN | ✓ | N/A | Partial | ✗ | ✗ |
| Natural product library (400k) | ✓ | Case-by-case | ✗ | ✓ | ✗ |
| Resistance-proof design | ✓ by design | ✗ | ✗ | ✗ | ✗ |
| Scaffold-split evaluation | ✓ | N/A | Varies | N/A | ✓ |
| GNNExplainer pharmacophore | ✓ | ✗ | ✗ | ✗ | ✗ |
| Ethnobotany validation (LOTUS) | ✓ | ✗ | ✗ | ✗ | ✗ |

Every existing tool has at least one fatal limitation: single-target scoring, fixed fingerprints that lose chemical context, similarity-based lookup that fails on novel scaffolds, or no natural product focus. PRISM addresses all four simultaneously.

---

## Project structure

```
prism/
├── data/
│   ├── 01_fetch_chembl.py          # Download IC₅₀ training data from ChEMBL API
│   ├── 02_prepare_dataset.py       # Bemis-Murcko scaffold split + binary labelling
│   ├── labelled_dataset.csv        # Training data (generated)
│   └── coconut_filtered.csv        # Pre-filtered COCONUT library (generated)
├── models/
│   ├── gnn_model.py                # MultiTargetGNN: GINEConv + multi-task heads
│   ├── train.py                    # Training loop with scaffold-split evaluation
│   ├── hparam_sweep.py             # Optuna hyperparameter optimisation
│   └── screen_coconut.py           # Batch GNN inference on COCONUT
├── utils/
│   ├── molecular_dataset.py        # PyG InMemoryDataset + graph construction
│   ├── polypharmacology.py         # PS score formula + PAINS filter + scaffolds
│   ├── visualise_fast.py           # UMAP chemical space heatmap
│   └── explain.py                  # GNNExplainer per-atom pharmacophore maps
├── outputs/                        # All results (generated)
│   ├── best_model.pt               # Trained model weights
│   ├── training_curves.png         # Loss and AUC curves
│   ├── top500_candidates.csv       # Top hit list
│   ├── top_scaffolds.csv           # Scaffold enrichment analysis
│   ├── training_log.csv            # Per-epoch metrics
│   ├── test_metrics.json           # Final test set performance
│   └── figure_umap_heatmap.png     # Chemical space map
├── requirements.txt
└── README.md
```

---

## Databases

| Database | Role | What we use |
|----------|------|-------------|
| **ChEMBL** | Training data | IC₅₀ records per target. Active = IC₅₀ < 10 µM → label 1. Inactive > 50 µM → label 0. Decoy ratio 5:1. |
| **COCONUT** | Screening library | 400,000 natural products. Filtered by MW, LogP, nitrogen count, ring count, PAINS. Top 15k by QED. |
| **TriTrypDB** | Essentiality weights | RNAi growth phenotype scores (Eᵢ) per target. |
| **LOTUS** | Ethnobotany | Maps molecules to source plants and traditional medicinal uses via InChIKey. |

---

## Molecular descriptors

### 2D graph topology — Phase 1 (active)

**Node features (7 per atom):**

| # | Feature | Encoding |
|---|---------|----------|
| 0 | Atomic number | Integer 0–118 |
| 1 | Chirality | 0=none, 1=CW, 2=CCW, 3=other |
| 2 | Formal charge | Clipped −4..+4, shifted → 0..8 |
| 3 | Is in ring | Binary |
| 4 | Is aromatic | Binary |
| 5 | Degree | 0–10 |
| 6 | Total H count | 0–8 |

**Edge features (8 per bond):**

| # | Feature | Encoding |
|---|---------|----------|
| 0–3 | Bond type | One-hot: single / double / triple / aromatic |
| 4–7 | Bond stereo | One-hot: none / any / Z / E |

### 3D WHIM descriptors — Batch 2 (planned)

114 rotation-invariant descriptors encoding molecular shape. Generated by averaging WHIM across 20 RDKit conformers (MMFF94 minimised). Fused with 2D graph embeddings via MLP bridge.

---

## Model architecture

```
SMILES string
     ↓
Graph: atoms=nodes (7 features), bonds=edges (8 features)
     ↓
Linear projection → 256-dim
     ↓
4× GINEConv layers [edge-feature-aware message passing]
   BatchNorm → ReLU → Dropout(0.2)
     ↓
Global mean pool + Global max pool → 512-dim
     ↓
Shared MLP → 256-dim molecular embedding
     ↓
Per-target linear heads → sigmoid → P(active | targetᵢ)
```

**Parameters:** 1,452,801  
**Loss:** Weighted BCE (pos_weight = n_neg/n_pos per target)  
**Evaluation:** Scaffold-split ROC-AUC, AUPRC, MCC

---

## Installation

```bash
conda create -n prism python=3.10 -y
conda activate prism

# PyTorch with GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# PyTorch CPU only
pip install torch torchvision

# PyTorch Geometric + rest
pip install torch_geometric
pip install -r requirements.txt
```

---

## Usage

```bash
# 1. Fetch training data
python data/01_fetch_chembl.py

# 2. Scaffold split
python data/02_prepare_dataset.py

# 3. Pre-filter COCONUT (download SDF from coconut.naturalproducts.net first)
python utils/polypharmacology.py

# 4. Train
python models/train.py \
    --csv data/labelled_dataset.csv \
    --epochs 100 --hidden_dim 256 --num_layers 4 \
    --dropout 0.2 --lr 0.001 --batch_size 32

# 5. Screen
python models/screen_coconut.py \
    --model outputs/best_model.pt \
    --coconut data/coconut_filtered.csv \
    --output outputs/coconut_scored.csv

# 6. Visualise
python utils/visualise_fast.py

# 7. Pharmacophore maps
python utils/explain.py --model outputs/best_model.pt \
    --top_df outputs/top500_candidates.csv --n 5
```

---

## Phase 1 results

| Metric | Value |
|--------|-------|
| Test ROC-AUC | **1.0000** |
| Test AUPRC | **1.0000** |
| Test MCC | **0.8757** |
| Best val AUPRC | 0.9475 (epoch 70) |
| Training set | 725 molecules |
| Positive rate | 5.5% (40 actives) |
| pos_weight | 17.12 |
| COCONUT screened | 15,000 |
| Top PS score | 0.855 |
| Top chemical classes | Azoles, Quinuclidines, Benzofurans |
| Screening time (CPU) | 11 minutes |

Auc result screenshot has been added to output.

---

## Roadmap

**Phase 1 (complete):** Single-target (TR) proof of concept on kinetoplastid parasites

**Batch 2:** PTR1 + CYP51 target heads · 3D WHIM descriptors · Full 400k COCONUT · NPASS augmentation · LOTUS enrichment analysis

**Phase 3:** Extension to other NTDs (Malaria, TB, Schistosomiasis) · Wet lab validation of top candidates

---

## Citation

```bibtex
@software{prism_2026,
  title  = {PRISM: Polypharmacology Ranking via Integrated Screening of natural Molecules},
  author = {Sethi, Samarth},
  year   = {2026},
  url    = {https://github.com/samarthsethi/prism}
}
```

