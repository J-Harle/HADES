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
    description="Calculate oxygen balance from SMILES CSV"
)
parser.add_argument(
    "--input", "-i", type=str, default="large_data.csv",#"hades_out.csv",
    help="Input CSV with CID and SMILES columns (default: hades_out.csv)"
)
parser.add_argument(
    "--output", "-o", type=str, default=None,
    help="Output CSV (default: overwrite input)"
)
args = parser.parse_args()

# Directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# One directory above the script
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))

# Input CSV path (one directory up)
input_csv = os.path.join(parent_dir, args.input)

# Output CSV (default: overwrite input)
output_csv = (
    os.path.join(parent_dir, args.output)
    if args.output is not None
    else input_csv
)


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
with open(input_csv, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise RuntimeError("Input CSV is empty")

# Validate headers
required = {"CID", "SMILES"}
missing = required - set(rows[0].keys())
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# Add column if missing
fieldnames = list(rows[0].keys())
if "Oxygen Balance /%" not in fieldnames:
    fieldnames.append("Oxygen Balance /%")

pbar = tqdm(rows, desc="Calculating oxygen balance", unit="mol")

for row in pbar:
    smiles = row["SMILES"].strip()

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

    row["Oxygen Balance /%"] = OB

pbar.close()

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated CSV written to: {output_csv}")
