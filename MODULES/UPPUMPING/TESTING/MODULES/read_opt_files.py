import os
import csv
import json
import h5py
import warnings
import numpy as np
from tqdm import tqdm
from rdkit import Chem

import sys
csv.field_size_limit(sys.maxsize)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))


def create_plot_dirs():
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    figure_dir = os.path.abspath(
        os.path.join(script_dir, "..", "..", "UPPUMPING", "FIGURES")
    )

    script_figure_dir = os.path.join(figure_dir, script_name)

    new_subdirs = [
        "DOS",
        "Box_DOS",
        "BE_Scaled_DOS",
        "Final",
        "First_Convolution",
        "Second_Convolution",
        "NO2_Angle",
    ]

    try:
        if not os.path.exists(script_figure_dir):
            os.makedirs(script_figure_dir)
            print(f"[INFO] Created directory: {script_figure_dir}")

        for sub in new_subdirs:
            path = os.path.join(script_figure_dir, sub)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"[INFO] Created subdirectory: {path}")

    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to create figure directories: {e}")

    return script_figure_dir

def read_extended_csv(
    subclustering=False,
    filter_cluster_id=None,
    filter_subcluster_id=None
):
    # csv_path = os.path.join(script_dir, "bak_in_uppumping.csv")
    csv_path = os.path.join(script_dir, "30_bench.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV not found: {csv_path}")

    if subclustering:
        molecules = {}  # {cluster_id: {sub_cluster_id: {filename: data}}}
    else:
        molecules = {}  # {filename: data}

    with open(csv_path, "r", encoding="latin-1") as f:
        reader = csv.DictReader(f)

        for row in reader:
            cluster_id = int(row["cluster_id"])
            sub_cluster_id = int(row["sub_cluster_id"])

            if filter_cluster_id is not None and cluster_id != filter_cluster_id:
                continue

            if filter_subcluster_id is not None and sub_cluster_id != filter_subcluster_id:
                continue

            filename = row["filename"]

            molecule_data = {
                "filename": filename,
                "SMILES": row["SMILES"],
                "cluster_id": cluster_id,
                "sub_cluster_id": sub_cluster_id,
                "H50": float(row["H50"]),
            }

            if subclustering:
                if cluster_id not in molecules:
                    molecules[cluster_id] = {}

                if sub_cluster_id not in molecules[cluster_id]:
                    molecules[cluster_id][sub_cluster_id] = {}

                molecules[cluster_id][sub_cluster_id][filename] = molecule_data
            else:
                molecules[filename] = molecule_data

    return molecules

def flatten_molecules(molecules, subclustering=False):
    if not subclustering:
        return molecules

    flat = {}

    for cluster_dict in molecules.values():
        for subcluster_dict in cluster_dict.values():
            for filename, mol in subcluster_dict.items():
                flat[filename] = mol

    return flat

def get_bonding(molecules):
    explosophores = {
        "C - O SINGLE",
        "C - N SINGLE",
        "N - O SINGLE",
        "O - O SINGLE",
        "N - N SINGLE",
    }

    nitro_smarts = "[N+](=O)[O-]"
    nitro_pattern = Chem.MolFromSmarts(nitro_smarts)

    exp_total = {}
    exp_ratio = {}

    for molecule in molecules.values():
        smiles = molecule["SMILES"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[ERROR] RDKit failed: {molecule['filename']}")
            continue

        mol = Chem.AddHs(mol)

        nitro_bond_indices = set()

        matches = mol.GetSubstructMatches(nitro_pattern)
        for match in matches:
            nitro_N = match[0]
            nitro_O1 = match[1]
            nitro_O2 = match[2]

            bond1 = mol.GetBondBetweenAtoms(nitro_N, nitro_O1)
            bond2 = mol.GetBondBetweenAtoms(nitro_N, nitro_O2)

            if bond1:
                nitro_bond_indices.add(bond1.GetIdx())
            if bond2:
                nitro_bond_indices.add(bond2.GetIdx())

        total_bonds = 0
        exp_bonds = 0

        for bond in mol.GetBonds():
            total_bonds += 1

            # Skip nitro N-O bonds
            if bond.GetIdx() in nitro_bond_indices:
                continue

            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            atoms = sorted([a1.GetSymbol(), a2.GetSymbol()])
            bond_type = bond.GetBondType().name.upper()

            bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"

            if bond_label in explosophores:
                exp_bonds += 1

        if total_bonds == 0:
            raise ValueError(f"[ERROR] No bonds in {molecule['filename']}")

        ratio = exp_bonds / total_bonds

        exp_total[smiles] = exp_bonds
        exp_ratio[smiles] = ratio
        molecule["exp_ratio"] = ratio

    return exp_total, exp_ratio


def read_txt_xyz(molecules):
    data = []

    # base_dir = os.path.join(script_dir, "OPTIMISED_STRUCTURES", "SMALL_MODEL")
    base_dir = os.path.join(script_dir, "OPTIMISED_STRUCTURES", "30_MOL")


    valid_dirs = {mol["filename"] for mol in molecules.values()}

    print()
    for dirname in tqdm(sorted(valid_dirs), desc="Reading Data Files", unit="molecule"):
        mol_dir = os.path.join(base_dir, dirname)

        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"[ERROR] SMALL_MODEL directory not found: {base_dir}")

        freqs, eigenvectors, coordinates = [], [], []
        atom_count = None

        for root, _, files in os.walk(mol_dir):
            txt_files = [f for f in files if f.endswith(".txt")]
            xyz_files = [f for f in files if f.endswith(".xyz") and "_jmol" not in f]

            for txt_file in txt_files:
                with open(os.path.join(root, txt_file), "r", encoding="latin-1") as f:                    
                    
                    current_mode = []

                    for line in f:
                        line = line.strip()

                        if "Number of atoms:" in line:
                            atom_count = int(line.split(":")[1])

                        elif line.startswith("Mode "):
                            if current_mode:
                                eigenvectors.append(current_mode)
                                current_mode = []

                            freq = float(line.split("=")[1].split("+")[0].strip())
                            if freq > 0:
                                freqs.append(freq)

                        elif line.startswith("Atom"):
                            atom, coords = line.split(":", 1)
                            idx = int(atom.split()[1])
                            symbol = atom.split()[2].strip("()")
                            vec = list(map(float, coords.split(",")))
                            current_mode.append((idx, symbol, vec))

                    if current_mode:
                        eigenvectors.append(current_mode)

            for xyz_file in xyz_files:
                with open(os.path.join(root, xyz_file), "r", encoding="latin-1") as f:
                    for line in f.readlines()[2:]:
                        parts = line.split()
                        if len(parts) == 4:
                            coordinates.append((parts[0], *map(float, parts[1:])))

        meta = molecules[dirname]

        data.append(
            {
                "molecule": dirname,
                "SMILES": meta["SMILES"],
                "cluster_id": meta["cluster_id"],
                "H50": meta["H50"],
                "exp_ratio": meta["exp_ratio"],
                "frequencies": freqs,
                "atom_count": atom_count,
                # "coordinates": coordinates,
                # "eigenvectors": eigenvectors,
            }
        )

    print()
    return data

def write_data_to_csv(data, filename="raw.csv"):
    output_path = os.path.join(script_dir, filename)

    if not data:
        raise ValueError("[ERROR] No data to write.")

    fieldnames = [
        "molecule",
        "SMILES",
        "cluster_id",
        "H50",
        "exp_ratio",
        "atom_count",
        "frequencies",
        # "coordinates",
        # "eigenvectors",
    ]

    with open(output_path, "w", newline="", encoding="latin-1") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in data:
            row = entry.copy()

            # Serialise list-like fields
            row["frequencies"] = json.dumps(entry["frequencies"])
            # row["coordinates"] = json.dumps(entry["coordinates"])
            # row["eigenvectors"] = json.dumps(entry["eigenvectors"])

            writer.writerow(row)

    # print(f"[INFO] Data written to: {output_path}")


if __name__ == "__main__":

    subclustering = True  
    filter_cluster_id = 1
    filter_subcluster_id = 1

    molecules = read_extended_csv(
        subclustering=subclustering,
        filter_cluster_id=filter_cluster_id,
        filter_subcluster_id=filter_subcluster_id
    )

    flat_molecules = flatten_molecules(
        molecules,
        subclustering=subclustering
    )

    get_bonding(flat_molecules)
    data = read_txt_xyz(flat_molecules)
    write_data_to_csv(data)

    # print(data[0].keys())



