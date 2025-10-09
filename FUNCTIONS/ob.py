# Calculate oxygen balance and write to CSV

import os
import csv
from collections import Counter

ATOMIC_WEIGHTS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
}

def oxygen_balance(C, H, O, mol_weight):
    """Calculate oxygen balance (%) given atom counts and molecular weight."""
    if mol_weight == 0:
        return None
    return (-1600.0 / mol_weight) * (2 * C + (H / 2) - O)

def process_xyz_files(parent_dir=None, output_csv="out.csv"):
    """
    Walk through subdirectories of parent_dir, read xyz files,
    compute oxygen balance, and write results to CSV with columns:
    Filename, SMILES, Oxygen balance /%
    If the CSV exists, append new results; otherwise, create it.
    """
    if parent_dir is None:
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "OPTIMISED_STRUCTURES"))
    parent_dir = os.path.abspath(parent_dir)
    print(f"Parent directory: {parent_dir}")

    if not os.path.isdir(parent_dir):
        print(f"ERROR: parent_dir does not exist: {parent_dir}")
        return False

    # Absolute path for output CSV (one directory above parent_dir)
    abs_output = os.path.abspath(os.path.join(parent_dir, "..", output_csv))
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    fieldnames = ["Filename", "SMILES", "Oxygen balance /%"]

    # Determine if CSV exists already
    file_exists = os.path.isfile(abs_output)

    try:
        with open(abs_output, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only if file did not exist
            if not file_exists:
                writer.writeheader()

            found_any = False
            for root, dirs, files in os.walk(parent_dir):
                for file in files:
                    if file.startswith("."):
                        continue
                    if not file.lower().endswith(".xyz"):
                        continue

                    found_any = True
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r") as f:
                            lines = f.readlines()
                    except Exception as e:
                        print(f"WARNING: failed to read '{path}': {e}")
                        continue

                    # Extract SMILES from 2nd line using colon delimiter
                    if len(lines) > 1:
                        line2 = lines[1].strip()
                        if ":" in line2:
                            parts = line2.split(":", 1)  # split only once
                            smiles_part = parts[1].strip()
                            # remove "SPE ..." if present
                            if "SPE" in smiles_part:
                                smiles_part = smiles_part.split("SPE")[0].strip()
                            smiles = smiles_part
                        else:
                            smiles = line2
                    else:
                        smiles = ""

                    # Extract atom symbols (skip first two lines)
                    atom_lines = lines[2:]
                    atom_types = [ln.split()[0] for ln in atom_lines if ln.strip()]
                    atom_counts = Counter(atom_types)

                    # Molecular weight
                    mol_weight = sum(
                        ATOMIC_WEIGHTS.get(atom, 0.0) * cnt
                        for atom, cnt in atom_counts.items()
                    )

                    C = atom_counts.get("C", 0)
                    H = atom_counts.get("H", 0)
                    O = atom_counts.get("O", 0)
                    OB = oxygen_balance(C, H, O, mol_weight)

                    row = {
                        "Filename": file.split(".")[0],
                        "SMILES": smiles,
                        "Oxygen balance /%": round(OB, 2) if OB is not None else ""
                    }
                    writer.writerow(row)
                    print(f"Processed: {file}  OB = {row['Oxygen balance /%']}%")

            if not found_any:
                print("NOTICE: No .xyz files were found under the parent directory.")
        print(f"CSV written to: {abs_output}")
        return True
    except Exception as e:
        print(f"ERROR: failed to write CSV '{abs_output}': {e}")
        return False

if __name__ == "__main__":
    # Run with default parent_dir (../OPTIMISED_STRUCTURES) and default output CSV name
    process_xyz_files()
