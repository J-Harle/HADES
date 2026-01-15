import argparse
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem, GraphDescriptors
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
default_csv_path = os.path.join(script_dir, "..", "hades_out.csv")

parser = argparse.ArgumentParser(
    description="Calculate generic molecular properties and overwrite/append them in the CSV"
)
parser.add_argument(
    "--input", "-i", type=str, default=default_csv_path,
    help="CSV file to read from and write back to"
)
args = parser.parse_args()


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    # print(df.info())
    return df


def calc_genprop(df):
    features = {
        "phi": [],
        "rotatable_bond_count": [],
        "no2_count": [],
        "n_count": [],
        "o_count": [],
        "c_count": [],
        "h_count": [],
        "arom_c_count": [],
        "arom_n_count": [],
        "arom_o_count": [],
        "atom_count": [],
        "mol_wt": [],
        "n_n_bond_count": [],
        "o_h_bond_count": [],
        "n_o_bond_count": [],
        "o_o_bond_count": [],
        "h_bond_acceptors": [],
        "h_bond_donors": [],
        "h_bond_ratio": [],
        "c_ratio": [],
        "h_ratio": [],
        "n_ratio": [],
        "o_ratio": [],
        "bond_dicts": []
    }

    all_bond_keys = set()

    nitro_smarts = Chem.MolFromSmarts("[N+](=O)[O-]")
    hbd_smarts = Chem.MolFromSmarts("[N,H,O;!$(*=O)]")
    hba_smarts = Chem.MolFromSmarts("[N,O;!$(*=O)]")

    for smi in tqdm(
        df["SMILES"],
        total=len(df),
        desc="Calculating molecular descriptors",
        unit="molecule"
    ):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            for key in features:
                features[key].append(None)
            continue

        mol = Chem.AddHs(mol)

        # --- atom counts ---
        atoms = list(mol.GetAtoms())

        c = sum(a.GetSymbol() == "C" and not a.GetIsAromatic() for a in atoms)
        h = sum(a.GetSymbol() == "H" for a in atoms)
        n = sum(a.GetSymbol() == "N" and not a.GetIsAromatic() for a in atoms)
        o = sum(a.GetSymbol() == "O" and not a.GetIsAromatic() for a in atoms)

        arom_c = sum(a.GetSymbol() == "C" and a.GetIsAromatic() for a in atoms)
        arom_n = sum(a.GetSymbol() == "N" and a.GetIsAromatic() for a in atoms)
        arom_o = sum(a.GetSymbol() == "O" and a.GetIsAromatic() for a in atoms)

        total_atoms = c + h + n + o + arom_c + arom_n + arom_o

        features["c_count"].append(c)
        features["h_count"].append(h)
        features["n_count"].append(n)
        features["o_count"].append(o)
        features["arom_c_count"].append(arom_c)
        features["arom_n_count"].append(arom_n)
        features["arom_o_count"].append(arom_o)
        features["atom_count"].append(total_atoms)

        # --- ratios ---
        if total_atoms > 0:
            features["c_ratio"].append((c + arom_c) / total_atoms)
            features["h_ratio"].append(h / total_atoms)
            features["n_ratio"].append((n + arom_n) / total_atoms)
            features["o_ratio"].append((o + arom_o) / total_atoms)
        else:
            features["c_ratio"].append(None)
            features["h_ratio"].append(None)
            features["n_ratio"].append(None)
            features["o_ratio"].append(None)

        # --- nitro groups ---
        features["no2_count"].append(len(mol.GetSubstructMatches(nitro_smarts)))

        # --- molecular weight ---
        features["mol_wt"].append(AllChem.CalcExactMolWt(mol))

        # --- bonds ---
        bond_info = {}
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetSymbol()
            a2 = bond.GetEndAtom().GetSymbol()
            atoms_key = "-".join(sorted([a1, a2]))
            bond_type = str(bond.GetBondType())
            key = f"{atoms_key}_{bond_type}"
            bond_info[key] = bond_info.get(key, 0) + 1
            all_bond_keys.add(key)

        features["bond_dicts"].append(bond_info)

        # --- H-bonding ---
        donors = len(mol.GetSubstructMatches(hbd_smarts))
        acceptors = len(mol.GetSubstructMatches(hba_smarts))

        features["h_bond_donors"].append(donors)
        features["h_bond_acceptors"].append(acceptors)

        if acceptors > 0:
            features["h_bond_ratio"].append(donors / acceptors)
        elif donors > 0:
            features["h_bond_ratio"].append("Div0Error")
        else:
            features["h_bond_ratio"].append(0)

        # --- rotatable bonds ---
        features["rotatable_bond_count"].append(
            AllChem.CalcNumRotatableBonds(mol)
        )

        # --- Kier flexibility ---
        k1 = GraphDescriptors.Kappa1(mol)
        k2 = GraphDescriptors.Kappa2(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()

        features["phi"].append(
            (k1 * k2) / heavy_atoms if heavy_atoms > 0 else None
        )

        # --- specific bond counts ---
        features["n_n_bond_count"].append(bond_info.get("N-N_SINGLE", 0))
        features["o_h_bond_count"].append(bond_info.get("H-O_SINGLE", 0))
        features["n_o_bond_count"].append(bond_info.get("N-O_SINGLE", 0))
        features["o_o_bond_count"].append(bond_info.get("O-O_SINGLE", 0))

    feat_df = pd.DataFrame(features)

    bond_df = pd.DataFrame(
        [{k: d.get(k, 0) for k in all_bond_keys} for d in feat_df["bond_dicts"]]
    ).fillna(0).astype(int)

    feat_df = pd.concat(
        [feat_df.drop(columns=["bond_dicts"]), bond_df],
        axis=1
    )

    return feat_df


if __name__ == "__main__":
    csv_path = args.input

    df = read_csv(csv_path)
    feat_df = calc_genprop(df)

    for col in feat_df.columns:
        df[col] = feat_df[col]

    df.to_csv(csv_path, index=False)
    print(f"\nAppended to CSV: {csv_path}")
