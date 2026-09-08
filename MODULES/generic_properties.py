#!/usr/bin/env python3
"""Calculate generic molecular descriptors from a SMILES-containing CSV file.

This script reads a CSV file containing a `SMILES` column, calculates a set of
generic molecular descriptors using RDKit, and appends or overwrites those
descriptor columns in the same CSV file.

The calculated descriptors include atom counts, atom ratios, nitro-group counts,
molecular weight, rotatable bonds, Kier flexibility, hydrogen-bonding features,
and bond-type counts.
"""

import argparse
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, GraphDescriptors, Lipinski
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "..", "hades_out.csv")


def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the input CSV path.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Calculate generic molecular properties and overwrite/append them "
            "in the CSV"
        )
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="CSV file to read from and write back to"
    )

    return parser.parse_args()


def read_csv(csv_path):
    """Read a CSV file into a pandas DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the CSV contents.
    """
    df = pd.read_csv(csv_path)

    return df


def calc_genprop(df):
    """Calculate generic molecular properties for each molecule in a DataFrame.

    The input DataFrame must contain a `SMILES` column. Each SMILES string is
    parsed with RDKit, explicit hydrogens are added, and molecular descriptors
    are calculated.

    Invalid SMILES strings are assigned missing values for all calculated
    features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a `SMILES` column.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing calculated molecular descriptors. The returned
        DataFrame has the same number of rows as the input DataFrame.
    """
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
        "bond_dicts": [],
    }

    all_bond_keys = set()

    nitro_smarts = Chem.MolFromSmarts("[N+](=O)[O-]")
    for smi in tqdm(
        df["SMILES"],
        total=len(df),
        desc="Calculating molecular descriptors",
        unit="molecule",
    ):
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            for key in features:
                features[key].append({} if key == "bond_dicts" else None)

            continue

        mol = Chem.AddHs(mol)

        # --------------------------------------------------
        # Atom counts
        # --------------------------------------------------
        atoms = list(mol.GetAtoms())

        c = sum(
            a.GetSymbol() == "C" and not a.GetIsAromatic()
            for a in atoms
        )
        h = sum(a.GetSymbol() == "H" for a in atoms)
        n = sum(
            a.GetSymbol() == "N" and not a.GetIsAromatic()
            for a in atoms
        )
        o = sum(
            a.GetSymbol() == "O" and not a.GetIsAromatic()
            for a in atoms
        )

        arom_c = sum(
            a.GetSymbol() == "C" and a.GetIsAromatic()
            for a in atoms
        )
        arom_n = sum(
            a.GetSymbol() == "N" and a.GetIsAromatic()
            for a in atoms
        )
        arom_o = sum(
            a.GetSymbol() == "O" and a.GetIsAromatic()
            for a in atoms
        )

        total_atoms = c + h + n + o + arom_c + arom_n + arom_o

        features["c_count"].append(c)
        features["h_count"].append(h)
        features["n_count"].append(n)
        features["o_count"].append(o)
        features["arom_c_count"].append(arom_c)
        features["arom_n_count"].append(arom_n)
        features["arom_o_count"].append(arom_o)
        features["atom_count"].append(total_atoms)

        # --------------------------------------------------
        # Atom ratios
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Nitro groups
        # --------------------------------------------------
        features["no2_count"].append(
            len(mol.GetSubstructMatches(nitro_smarts))
        )

        # --------------------------------------------------
        # Molecular weight
        # --------------------------------------------------
        features["mol_wt"].append(AllChem.CalcExactMolWt(mol))

        # --------------------------------------------------
        # Bond counts
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Hydrogen-bonding features
        # --------------------------------------------------
        donors = Lipinski.NumHDonors(mol)
        acceptors = Lipinski.NumHAcceptors(mol)

        features["h_bond_donors"].append(donors)
        features["h_bond_acceptors"].append(acceptors)

        if acceptors > 0:
            features["h_bond_ratio"].append(donors / acceptors)
        elif donors > 0:
            features["h_bond_ratio"].append(None)
        else:
            features["h_bond_ratio"].append(0)

        # --------------------------------------------------
        # Rotatable bonds
        # --------------------------------------------------
        features["rotatable_bond_count"].append(
            AllChem.CalcNumRotatableBonds(mol)
        )

        # --------------------------------------------------
        # Kier flexibility
        # --------------------------------------------------
        k1 = GraphDescriptors.Kappa1(mol)
        k2 = GraphDescriptors.Kappa2(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()

        features["phi"].append(
            (k1 * k2) / heavy_atoms
            if heavy_atoms > 0
            else None
        )

        # --------------------------------------------------
        # Specific bond counts
        # --------------------------------------------------
        features["n_n_bond_count"].append(
            bond_info.get("N-N_SINGLE", 0)
        )
        features["o_h_bond_count"].append(
            bond_info.get("H-O_SINGLE", 0)
        )
        features["n_o_bond_count"].append(
            bond_info.get("N-O_SINGLE", 0)
        )
        features["o_o_bond_count"].append(
            bond_info.get("O-O_SINGLE", 0)
        )

    feat_df = pd.DataFrame(features)

    bond_df = pd.DataFrame(
        [
            {k: d.get(k, 0) for k in all_bond_keys}
            for d in feat_df["bond_dicts"]
        ]
    ).fillna(0).astype(int)

    feat_df = pd.concat(
        [
            feat_df.drop(columns=["bond_dicts"]),
            bond_df,
        ],
        axis=1,
    )

    return feat_df


def main():
    """Run the generic molecular descriptor workflow.

    The workflow reads the input CSV, checks for a `SMILES` column, calculates
    molecular descriptors, appends or overwrites those descriptor columns in the
    DataFrame, and writes the updated CSV back to disk.
    """
    args = parse_args()

    csv_path = os.path.abspath(args.input)

    print(f"\nInput CSV: {csv_path}")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = read_csv(csv_path)

    if "SMILES" not in df.columns:
        raise KeyError(
            f"'SMILES' column not found in {csv_path}. "
            f"Available columns are: {list(df.columns)}"
        )

    feat_df = calc_genprop(df)

    for col in feat_df.columns:
        df[col] = feat_df[col]

    df.to_csv(csv_path, index=False)

    print(f"\nAppended generic properties to CSV: {csv_path}")

if __name__ == "__main__":
    main()
