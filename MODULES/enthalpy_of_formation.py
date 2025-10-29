import numpy
import csv
import os
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem

# Atomic and Bond contributions (Eh)
contrib_dict = {
"C":  0.020077,
"C_arom":  0.007487,
"H": -0.019859,
"N":  0.035046,
"O": -0.023906,
"C - C AROMATIC":  0.007487,
"C - C DOUBLE":  0.025537,
"C - C SINGLE": -0.007783,
"C - H SINGLE":  0.010525,
"C - N SINGLE": -0.007867,
"C - O SINGLE": -0.001446,
"H - N SINGLE":  0.001780,
"H - O SINGLE": -0.032165,
"N - N DOUBLE":  0.012195,
"N - N SINGLE":  0.005614,
"N - O DOUBLE":  0.010250,
"N - O SINGLE": -0.001990,
}

script_dir = os.path.dirname(os.path.abspath(__file__))
xyz_dir = os.path.join(script_dir, "../OPTIMISED_STRUCTURES")

def process_xyz_files(xyz_dir):
    for root, dirs, files in os.walk(xyz_dir):
        xyz_files = [f for f in files if f.endswith(".xyz")]
        for xyz_file in xyz_files:
            xyz_path = os.path.join(root, xyz_file)
            with open(xyz_path, "r") as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                print(f"Warning: {xyz_file} is missing SMILES line.")
                continue
            
            # Extract SMILES string (assuming second line: "SMILES <smiles>")
            parts = lines[1].strip().split()
            smiles = parts[1] if len(parts) > 1 else None
            if not smiles:
                print(f"Warning: {xyz_file} missing SMILES entry.")
                continue

            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)

            # --- Atom Counting ---
            atom_counts = Counter()
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if atom.GetIsAromatic():
                    symbol += "_arom"
                atom_counts[symbol] += 1

            # --- Bond Counting ---
            bond_counter = Counter()
            for bond in mol.GetBonds():
                atoms = sorted([
                    bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol()
                ])
                bond_type = str(bond.GetBondType())
                bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
                bond_counter[bond_label] += 1

            # --- Output section ---
            print(f"\n=== Molecule: {smiles} ({xyz_file}) ===")

            print("Atoms:")
            for atom, count in sorted(atom_counts.items()):
                print(f"  {atom}: {count}")

            print("Bonds:")
            for bond_label, count in sorted(bond_counter.items()):
                print(f"  {bond_label}: {count}")

def write_or_ammend_csv():
    pass


if __name__ == "__main__":
    process_xyz_files(xyz_dir)
