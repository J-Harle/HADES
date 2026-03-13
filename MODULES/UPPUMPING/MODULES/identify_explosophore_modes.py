import csv
import sys
import pandas as pd
from rdkit import Chem

csv.field_size_limit(sys.maxsize) #enable reading large csvs

"""
Construct a d matrix to identify the modes which are being caused by the explosphore bonds
Will most likely need to construct a list of which atoms are actually the explosphores, then track 
How the eigenvectors effect those atoms
"""

def read_csv():
    data = {}

    with open("raw.csv", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            filename = row["molecule"]
            smiles = row["SMILES"]
            h50 = float(row["H50"]) 

            mol_data = {
                "filename": filename,
                "smiles": smiles,
                "h50": h50,
            }

            data[filename] = mol_data 

    return data


def id_explosophore_atoms(data):
    explosophores = {
        "C - O SINGLE",
        "C - N SINGLE",
        "N - O SINGLE",
        "O - O SINGLE",
        "N - N SINGLE",
    }

    nitro_smarts = "[N+](=O)[O-]"
    nitro_pattern = Chem.MolFromSmarts(nitro_smarts)

    for entry in data.values():
        smiles = entry.get("smiles")
        name = entry.get("filename", "Unknown")

        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Could not parse SMILES for {name}")
            continue

        mol = Chem.AddHs(mol)

        # --- Identify nitro N–O bond indices ---
        nitro_bond_indices = set()
        nitro_matches = mol.GetSubstructMatches(nitro_pattern)

        for match in nitro_matches:
            nitro_n, nitro_o1, nitro_o2 = match
            bond1 = mol.GetBondBetweenAtoms(nitro_n, nitro_o1)
            bond2 = mol.GetBondBetweenAtoms(nitro_n, nitro_o2)
            if bond1:
                nitro_bond_indices.add(bond1.GetIdx())
            if bond2:
                nitro_bond_indices.add(bond2.GetIdx())

        print(f"\n{name} ({smiles})")

        total_bonds = mol.GetNumBonds()  # all bonds including H
        explosophore_count = 0

        for bond in mol.GetBonds():

            # Skip nitro N–O bonds
            if bond.GetIdx() in nitro_bond_indices:
                continue

            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            # Skip aromatic–aromatic bonds for explosophore count
            if a1.GetIsAromatic() and a2.GetIsAromatic():
                continue

            atoms = sorted([a1.GetSymbol(), a2.GetSymbol()])
            bond_type = bond.GetBondType().name
            bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"

            if bond_label in explosophores:
                explosophore_count += 1
                print(
                    f"  Explosophore bond found: "
                    f"{bond_label} "
                    f"(atoms {a1.GetIdx()}-{a2.GetIdx()}, bond idx {bond.GetIdx()})"
                )

        # --- Ratio calculation ---
        ratio = explosophore_count / total_bonds if total_bonds else 0.0

        print(f"  Total bonds: {total_bonds}")
        print(f"  Explosophore bonds: {explosophore_count}")
        print(f"  Explosophore ratio: {ratio:.4f}")


if __name__ == "__main__":
    data = read_csv()
    # print(data)
    id_explosophore_atoms(data)
