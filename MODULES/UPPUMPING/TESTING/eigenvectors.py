import numpy as np
import os
import csv
import json
import sys
from rdkit import Chem

csv.field_size_limit(sys.maxsize)

script_dir = os.path.dirname(os.path.abspath(__file__))


def read_data_from_csv(csv_name="raw.csv"):

    csv_path = os.path.join(script_dir, csv_name)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV not found: {csv_path}")

    data = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                data.append({
                    "molecule": row["molecule"],
                    "SMILES": row["SMILES"],
                    "cluster_id": row["cluster_id"],
                    "H50": float(row["H50"]),
                    "exp_ratio": float(row["exp_ratio"]),
                    "atom_count": int(row["atom_count"]),
                    "frequencies": json.loads(row["frequencies"]),
                    "coordinates": json.loads(row["coordinates"]),
                    "eigenvectors": json.loads(row["eigenvectors"]),
                    "all_freqs": json.loads(row["all_freqs"]),
                })
            except Exception as e:
                print(f"[WARN] Skipping row due to parsing error: {e}")

    return data


def unmassweight_evectors(mol):

    mass_dict = {"C": 12, "H": 1, "N": 14, "O": 16}

    print("\nProcessing molecule:", mol.get("molecule"))

    evecs = mol.get("eigenvectors")

    mol_modes = []

    for mode in evecs:

        new_mode = []

        for atom in mode:

            atom_index = atom[0]
            atom_id = atom[1]
            coords = atom[2]

            un_mw_factor = np.sqrt(mass_dict[atom_id])

            new_coords = [c * un_mw_factor for c in coords]

            new_atom = [atom_index, atom_id, new_coords]

            new_mode.append(new_atom)

        mol_modes.append(new_mode)

    return mol_modes


def construct_bonding(mol):

    smiles = mol.get("SMILES")

    rd_mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    bonds = []

    for bond in rd_mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bonds.append((i, j))

    return rd_mol, bonds


def print_atom_index_map(rd_mol):

    print("\nAtom index mapping:")

    for atom in rd_mol.GetAtoms():

        idx = atom.GetIdx() + 1
        symbol = atom.GetSymbol()

        print(f"{idx} = {symbol}")


def print_bonds(rd_mol):

    print("\nBond connectivity:")

    for bond in rd_mol.GetBonds():

        i = bond.GetBeginAtomIdx() + 1
        j = bond.GetEndAtomIdx() + 1

        atom_i = rd_mol.GetAtomWithIdx(i-1).GetSymbol()
        atom_j = rd_mol.GetAtomWithIdx(j-1).GetSymbol()

        print(f"{i}-{j} : {atom_i}-{atom_j}")


def bonded_atom_motion(modes, bonds, rd_mol, frequencies):

    explosophore_types = [
        ("C", "O"),
        ("C", "N"),
        ("N", "O"),
        ("O", "O"),
        ("N", "N")
    ]

    bond_orders = {}

    nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")
    matches = rd_mol.GetSubstructMatches(nitro_pattern)

    for match in matches:

        nitro_N = match[0]
        nitro_O1 = match[1]
        nitro_O2 = match[2]

        bond_orders[(nitro_N, nitro_O1)] = 1.5
        bond_orders[(nitro_N, nitro_O2)] = 1.5

    for mode_index, mode in enumerate(modes):

        freq = frequencies[mode_index]

        motions = []

        for i, j in bonds:

            atom_i = rd_mol.GetAtomWithIdx(i).GetSymbol()
            atom_j = rd_mol.GetAtomWithIdx(j).GetSymbol()

            # ONLY keep explosophore bonds
            if not (
                (atom_i, atom_j) in explosophore_types or
                (atom_j, atom_i) in explosophore_types
            ):
                continue

            dx_i, dy_i, dz_i = mode[i][2]
            dx_j, dy_j, dz_j = mode[j][2]

            rel = np.array([dx_i - dx_j, dy_i - dy_j, dz_i - dz_j])
            mag = np.linalg.norm(rel)

            bo = bond_orders.get((i, j), bond_orders.get((j, i), None))

            if bo is not None:
                bond_type_str = f"Custom({bo})"
            else:
                bond = rd_mol.GetBondBetweenAtoms(i, j)
                bond_type_str = str(bond.GetBondType()) if bond else "UNKNOWN"

            motions.append({
                "bond": (i+1, j+1),
                "atoms": (atom_i, atom_j),
                "motion": mag,
                "type": bond_type_str
            })

        if not motions:
            continue

        motions.sort(key=lambda x: x["motion"], reverse=True)

        print(f"\nMode {mode_index+1} ({freq:.2f} cm^-1)")

        for m in motions:

            i, j = m["bond"]
            atom_i, atom_j = m["atoms"]

            print(f"{i}({atom_i}) - {j}({atom_j}) [{m['type']}]   relative motion: {m['motion']:.4f}")

def modes_w_exp_char(mol, modes, bonds, rd_mol, frequencies, min_contribution=0.4):
    """
    Check which vibrational modes have significant motion along explosophore bonds.
    """

    explosophore_types = [
        ("C", "O"),
        ("C", "N"),
        ("N", "O"),
        ("O", "O"),
        ("N", "N")
    ]

    explosophore_freqs = []

    for mode_index, mode in enumerate(modes):

        freq = frequencies[mode_index]
        significant = False

        for i, j in bonds:

            atom_i = rd_mol.GetAtomWithIdx(i).GetSymbol()
            atom_j = rd_mol.GetAtomWithIdx(j).GetSymbol()

            # check if bond is explosophore type
            if (atom_i, atom_j) in explosophore_types or (atom_j, atom_i) in explosophore_types:

                dx_i, dy_i, dz_i = mode[i][2]
                dx_j, dy_j, dz_j = mode[j][2]

                rel = np.array([dx_i - dx_j, dy_i - dy_j, dz_i - dz_j])
                mag = np.linalg.norm(rel)

                if mag > min_contribution:
                    significant = True
                    break

        if significant:
            print(f"Mode {mode_index+1} ({freq:.2f} cm^-1) has significant explosophore motion")
            explosophore_freqs.append(freq)

    return explosophore_freqs
def write_explosophore_freqs(csv_name, freq_list):

    csv_path = os.path.join(script_dir, csv_name)

    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames

        if "explosophore freqs" not in fieldnames:
            fieldnames.append("explosophore freqs")

        for row, freqs in zip(reader, freq_list):

            row["explosophore freqs"] = json.dumps(freqs)

            rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(rows)


if __name__ == "__main__":

    runs = ["30_bench_raw.csv"]

    for csv_name in runs:

        print(f"\nProcessing: {csv_name}")

        data = read_data_from_csv(csv_name)

        all_exp_freqs = []

        for mol in data:

            mol_modes = unmassweight_evectors(mol)

            rd_mol, bonds = construct_bonding(mol)

            frequencies = mol.get("all_freqs")

            bonded_atom_motion(mol_modes, bonds, rd_mol, frequencies)

            exp_freqs = modes_w_exp_char(mol, mol_modes, bonds, rd_mol, frequencies, min_contribution=0.3)

            all_exp_freqs.append(exp_freqs)

        write_explosophore_freqs(csv_name, all_exp_freqs)
