import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
import os

def main():
    os.makedirs("data", exist_ok=True)
    activity = new_client.activity
    molecule = new_client.molecule

    # Target: Trypanothione Reductase (Very Specific)
    target_id = "CHEMBL3038484" 
    
    print(f"Fetching validated inhibitors for TR ({target_id})...")
    res = activity.filter(target_chembl_id=target_id, standard_type="IC50")
    df = pd.DataFrame(list(res))

    # Clean and Label
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors='coerce')
    
    # Strict thresholds: 10uM for Actives, 50uM for Inactives
    actives = df[df["standard_value"] <= 10000].copy()
    actives["label"] = 1
    
    known_inactives = df[df["standard_value"] >= 50000].copy()
    known_inactives["label"] = 0
    
    print(f"Found {len(actives)} true actives.")

    # ── FIXED DECOY FETCH ──
    # We pull exactly 1000 randoms and FORCE them to be Label 0
    print("Fetching 1000 background decoys...")
    decoy_res = molecule.filter(structure__isnull=False).only(["molecule_structures"])[:1000]
    
    decoys_list = []
    for r in decoy_res:
        if r.get("molecule_structures"):
            decoys_list.append({
                "canonical_smiles": r["molecule_structures"]["canonical_smiles"],
                "label": 0,
                "target": "DECOY"
            })
    decoys = pd.DataFrame(decoys_list)

    # Combine
    final_df = pd.concat([actives, known_inactives, decoys], ignore_index=True)

    # Final Deduplication via InChIKey
    print("Deduplicating...")
    def get_key(smi):
        try:
            m = Chem.MolFromSmiles(smi)
            return MolToInchiKey(m) if m else None
        except: return None
    
    final_df["inchikey"] = final_df["canonical_smiles"].apply(get_key)
    final_df = final_df.dropna(subset=["inchikey"]).drop_duplicates(subset="inchikey")

    # Split back for saving
    out_actives = final_df[final_df["label"] == 1]
    out_inactives = final_df[final_df["label"] == 0]

    out_actives.to_csv("data/raw_actives.csv", index=False)
    out_inactives.to_csv("data/raw_inactives.csv", index=False)

    print(f"\nSUCCESS: {len(out_actives)} Actives | {len(out_inactives)} Inactives")

if __name__ == "__main__":
    main()