"""
Calculate oxygen balance values from a SMILES-containing CSV file.

This script reads a CSV file containing a `SMILES` column, calculates the oxygen
balance for each molecule using its elemental composition, and writes the values
back to the same CSV file or to a separate output file.
"""

import csv
import os
import argparse
from collections import Counter

from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")


# Constants
ATOMIC_WEIGHTS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
}


# CLI
def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the input CSV path and optional
        output CSV path.
    """
    parser = argparse.ArgumentParser(
        description="Calculate oxygen balance from a SMILES CSV"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default="hades_out.csv",
        help="Input CSV containing a SMILES column. Default: hades_out.csv"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV. Default: overwrite input CSV"
    )

    return parser.parse_args()


# Path handling
def resolve_path(path):
    """Convert a user-provided path into an absolute path.

    Parameters
    ----------
    path : str
        Input file or directory path.

    Returns
    -------
    str
        Absolute version of the input path.
    """
    return os.path.abspath(path)


# Chemistry helpers
def oxygen_balance(C, H, O, mol_weight):
    """Calculate the oxygen balance of a molecule.

    The oxygen balance is calculated using:

    OB% = (-1600 / molecular weight) * (2C + H/2 - O)

    where C, H, and O are the number of carbon, hydrogen, and oxygen atoms in
    the molecule.

    Parameters
    ----------
    C : int
        Number of carbon atoms.
    H : int
        Number of hydrogen atoms.
    O : int
        Number of oxygen atoms.
    mol_weight : float
        Molecular weight of the molecule in g mol-1.

    Returns
    -------
    float or None
        Oxygen balance percentage. Returns None if the molecular weight is zero.
    """
    if mol_weight == 0:
        return None

    return (-1600.0 / mol_weight) * (2 * C + (H / 2) - O)


def atom_counts_from_smiles(smiles):
    """Count atoms in a molecule from a SMILES string.

    Explicit hydrogens are added before counting so that hydrogen atoms are
    included in the final elemental composition.

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    collections.Counter or None
        Counter containing atom symbols and their counts. Returns None if the
        SMILES string cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


# Main logic
if __name__ == "__main__":

    args = parse_args()
    input_csv = resolve_path(args.input)

    if args.output is None:
        output_csv = input_csv
    else:
        output_csv = resolve_path(args.output)

    print(f"\nInput CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")

    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Input CSV is empty: {input_csv}")

    fieldnames = list(rows[0].keys())

    required = {"SMILES"}
    missing = required - set(fieldnames)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}. "
            f"Available columns are: {fieldnames}"
        )

    ob_col = "Oxygen Balance /%"

    if ob_col not in fieldnames:
        fieldnames.append(ob_col)

    pbar = tqdm(
        rows,
        desc="Calculating oxygen balance",
        unit="molecule"
    )

    for row in pbar:
        smiles = str(row.get("SMILES", "")).strip()

        OB = ""

        if smiles:
            counts = atom_counts_from_smiles(smiles)

            if counts is not None:
                C = counts.get("C", 0)
                H = counts.get("H", 0)
                O = counts.get("O", 0)

                mol_weight = sum(
                    ATOMIC_WEIGHTS.get(el, 0.0) * n
                    for el, n in counts.items()
                )

                val = oxygen_balance(C, H, O, mol_weight)

                if val is not None:
                    OB = round(val, 2)

        row[ob_col] = OB

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUpdated CSV written to: {output_csv}")



