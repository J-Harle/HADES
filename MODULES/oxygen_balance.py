import csv
import os
import argparse
from collections import Counter
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


# -------------------------
# CLI
# -------------------------
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

args = parser.parse_args()


# -------------------------
# Constants
# -------------------------
ATOMIC_WEIGHTS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
}


# -------------------------
# Path handling
# -------------------------
def resolve_path(path):
    """
    Converts a user-provided path into an absolute path.

    Examples:
        -i hades_out.csv              -> /current/working/dir/hades_out.csv
        -i /full/path/hades_out.csv   -> /full/path/hades_out.csv
    """
    return os.path.abspath(path)


# -------------------------
# Chemistry helpers
# -------------------------
def oxygen_balance(C, H, O, mol_weight):
    if mol_weight == 0:
        return None

    return (-1600.0 / mol_weight) * (2 * C + (H / 2) - O)


def atom_counts_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


# -------------------------
# Main logic
# -------------------------
def main():
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

    # Only SMILES is actually required for oxygen balance
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


if __name__ == "__main__":
    main()