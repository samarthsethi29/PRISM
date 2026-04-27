import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import rdchem
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

BOND_TYPE_MAP = {
    rdchem.BondType.SINGLE:   0,
    rdchem.BondType.DOUBLE:   1,
    rdchem.BondType.TRIPLE:   2,
    rdchem.BondType.AROMATIC: 3,
}
BOND_STEREO_MAP = {
    rdchem.BondStereo.STEREONONE: 0,
    rdchem.BondStereo.STEREOANY:  1,
    rdchem.BondStereo.STEREOZ:    2,
    rdchem.BondStereo.STEREOE:    3,
}
NUM_ATOM_FEATURES = 7
NUM_EDGE_FEATURES = 8
SPLIT_MAP = {"train": 0, "val": 1, "test": 2}
SPLIT_MAP_INV = {0: "train", 1: "val", 2: "test"}


def atom_features(atom):
    chirality_map = {
        rdchem.ChiralType.CHI_UNSPECIFIED:     0,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CW:  1,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
        rdchem.ChiralType.CHI_OTHER:           3,
    }
    return [
        atom.GetAtomicNum(),
        chirality_map.get(atom.GetChiralTag(), 0),
        int(np.clip(atom.GetFormalCharge() + 4, 0, 8)),
        int(atom.IsInRing()),
        int(atom.GetIsAromatic()),
        min(atom.GetDegree(), 10),
        min(atom.GetTotalNumHs(), 8),
    ]


def bond_features(bond):
    bt = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
    bs = BOND_STEREO_MAP.get(bond.GetStereo(), 0)
    return [int(bt == i) for i in range(4)] + [int(bs == i) for i in range(4)]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    if mol.GetNumBonds() == 0:
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        edge_attr  = torch.zeros((1, NUM_EDGE_FEATURES), dtype=torch.float)
    else:
        src, dst, attrs = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            f = bond_features(bond)
            src  += [i, j]
            dst  += [j, i]
            attrs += [f, f]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(attrs, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MolecularDataset(InMemoryDataset):

    def __init__(self, root, csv_path, transform=None, pre_transform=None):
        self.csv_path = csv_path
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["molecular_graphs.pt"]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.csv_path)
        log.info(f"Processing {len(df)} molecules ...")
        data_list = []
        skipped = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
            g = smiles_to_graph(str(row["canonical_smiles"]))
            if g is None:
                skipped += 1
                continue
            g.y         = torch.tensor([float(row["label"])], dtype=torch.float)
            g.split_idx = torch.tensor(
                [SPLIT_MAP.get(str(row.get("split", "train")), 0)],
                dtype=torch.long
            )
            data_list.append(g)
        log.info(f"Built {len(data_list)} graphs, skipped {skipped}")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])