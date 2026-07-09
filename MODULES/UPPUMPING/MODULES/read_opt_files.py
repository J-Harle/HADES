import os
import csv
import json
import warnings
from tqdm import tqdm
from rdkit import Chem
import argparse
import sys
csv.field_size_limit(sys.maxsize)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frequencies, coordinates, and metadata from optimised molecule directories"
    )

    parser.add_argument(
        "--input", "-i",
        default="hades_out.csv",
        help="Input CSV file to read. Default: hades_out.csv"
    )

    parser.add_argument(
        "--dir", "-dir",
        dest="base_dir",
        default="OPTIMISED_STRUCTURES/HADES",
        help="Directory containing optimised molecule subdirectories. Default: OPTIMISED_STRUCTURES/HADES"
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output raw CSV file. Default: <input_name>_raw.csv"
    )

    return parser.parse_args()


def read_extended_csv(
    csv_filename,
    use_clustering=False,
    filter_cluster_id=None
):
    csv_path = os.path.abspath(csv_filename)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV not found: {csv_path}")

    molecules = {}

    with open(csv_path, "r", encoding="latin-1", newline="") as f:

        # Auto-detect comma/tab delimiter
        sample = f.read(4096)
        f.seek(0)

        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(f, dialect=dialect)

        if reader.fieldnames is None:
            raise ValueError(f"[ERROR] No headers found in: {csv_filename}")

        # Strip whitespace from headers
        reader.fieldnames = [h.strip() if h is not None else h for h in reader.fieldnames]
        headers = reader.fieldnames

        # Prefer FILENAME, then CID, then molecule/MOLECULE
        id_candidates = ["FILENAME", "CID", "molecule", "MOLECULE"]

        id_column = None
        for candidate in id_candidates:
            if candidate in headers:
                id_column = candidate
                break

        if id_column is None:
            raise ValueError(
                f"[ERROR] No valid ID column found in {csv_filename}. "
                f"Expected one of {id_candidates}. "
                f"Available columns: {headers}"
            )
        
        # print(f"[INFO] Using ID column: {id_column}")

        has_mol_type = "MOL_TYPE" in headers

        for row in reader:

            # Strip whitespace from row keys
            row = {
                k.strip() if k is not None else k: v
                for k, v in row.items()
            }

            filename = str(row[id_column]).strip()

            if filename == "":
                print(f"[WARNING] Empty {id_column} found, skipping row")
                continue

            if has_mol_type:
                cluster_id = row["MOL_TYPE"].strip()
            else:
                cluster_id = "unknown"

            if (
                filter_cluster_id is not None
                and cluster_id not in filter_cluster_id
            ):
                continue

            if (
                "H50 /J" in headers
                and row["H50 /J"].strip() != ""
            ):
                h50_value = float(row["H50 /J"])
            else:
                h50_value = None

            molecule_data = {
                "filename": filename,
                "SMILES": row["SMILES"],
                "cluster_id": cluster_id,
                "H50": h50_value,
            }

            if use_clustering:
                molecules.setdefault(cluster_id, {})
                molecules[cluster_id][filename] = molecule_data
            else:
                molecules[filename] = molecule_data

    return molecules

def flatten_molecules(molecules, use_clustering=False):

    if not use_clustering:
        return molecules

    flat = {}

    for cluster_dict in molecules.values():

        for filename, mol in cluster_dict.items():
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

            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            atoms = sorted([a1.GetSymbol(), a2.GetSymbol()])

            bond_type = bond.GetBondType().name.upper()

            bond_label = (f"{atoms[0]} - {atoms[1]} {bond_type}")

            if bond_label in explosophores:
                exp_bonds += 1

        if total_bonds == 0:
            raise ValueError(f"[ERROR] No bonds in {molecule['filename']}")

        ratio = exp_bonds / total_bonds

        exp_total[smiles] = exp_bonds
        exp_ratio[smiles] = ratio

        molecule["exp_ratio"] = ratio

    return exp_total, exp_ratio


# =========================================================
# CSV WRITING
# =========================================================

def initialise_csv(filename):

    output_path = os.path.abspath(filename)

    fieldnames = [
        "molecule",
        "SMILES",
        "cluster_id",
        "H50",
        "exp_ratio",
        "atom_count",
        "frequencies",
        "all_freqs",
        "coordinates",
    ]

    with open(output_path, "w", newline="", encoding="latin-1") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_molecule_to_csv(entry, filename):
    output_path = os.path.abspath(filename)

    fieldnames = [
        "molecule",
        "SMILES",
        "cluster_id",
        "H50",
        "exp_ratio",
        "atom_count",
        "frequencies",
        "all_freqs",
        "coordinates",
    ]

    row = entry.copy()

    row["frequencies"] = json.dumps(entry["frequencies"])
    row["all_freqs"] = json.dumps(entry["all_freqs"])
    row["coordinates"] = json.dumps(entry["coordinates"])

    with open(output_path, "a", newline="", encoding="latin-1") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


# =========================================================
# MAIN DATA EXTRACTION
# =========================================================

def read_txt_xyz(molecules, base_dir, output_filename=None, save_interval=1000):
    valid_dirs = {mol["filename"] for mol in molecules.values()}

    print()

    for idx, dirname in enumerate(
        tqdm(sorted(valid_dirs), desc="Reading Data Files", unit=" molecules"), start=1):
        mol_dir = os.path.join(base_dir, dirname)
        if not os.path.isdir(mol_dir):
            # print(f"[WARNING] Molecule directory not found, skipping: {mol_dir}")

            continue

        freqs = []
        all_freqs = []
        eigenvectors = []
        coordinates = []

        atom_count = None

        for root, _, files in os.walk(mol_dir):
            txt_files = [f for f in files if f.endswith(".txt")]
            xyz_files = [f for f in files if (f.endswith(".xyz") and "_jmol" not in f)]

            # ==========================================
            # TXT FILES
            # ==========================================

            for txt_file in txt_files:

                txt_path = os.path.join(root, txt_file)

                with open(txt_path, "r", encoding="latin-1") as f:
                    current_mode = []
                    for line in f:
                        line = line.strip()
                        if "Number of atoms:" in line:
                            atom_count = int(line.split(":")[1])

                        elif line.startswith("Mode "):
                            if (atom_count is not None and len(all_freqs) >= 3 * atom_count):
                                break

                            if current_mode:
                                eigenvectors.append(current_mode)
                                current_mode = []

                            freq = float(line.split("=")[1].split("+")[0].strip())

                            if freq > 0:
                                freqs.append(freq)

                            all_freqs.append(freq)

                        elif line.startswith("Atom"):
                            atom, coords = line.split(":")
                            idx_atom = int(atom.split()[1])
                            symbol = (atom.split()[2].strip("()"))
                            vec = list(map(float, coords.split(",")))
                            current_mode.append((idx_atom, symbol, vec))

                    if current_mode:
                        eigenvectors.append(current_mode)

            # ==========================================
            # XYZ FILES
            # ==========================================

            for xyz_file in xyz_files:
                xyz_path = os.path.join(root, xyz_file)
                with open(xyz_path, "r", encoding="latin-1") as f:
                    for line in f.readlines()[2:]:
                        parts = line.split()
                        if len(parts) == 4:
                            coordinates.append((parts[0], *map(float, parts[1:])))

        meta = molecules[dirname]

        entry = {
            "molecule": dirname,
            "SMILES": meta["SMILES"],
            "cluster_id": meta["cluster_id"],
            "H50": meta["H50"],
            "exp_ratio": meta["exp_ratio"],
            "frequencies": freqs,
            "all_freqs": all_freqs,
            "atom_count": atom_count,
            "coordinates": coordinates,
            # "eigenvectors": eigenvectors,
        }

        # ==========================================
        # WRITE IMMEDIATELY
        # ==========================================

        append_molecule_to_csv(
            entry,
            output_filename
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    args = parse_args()

    csv_path = os.path.abspath(args.input)
    base_dir = os.path.abspath(args.base_dir)

    if args.output is None:
        input_stem = os.path.splitext(os.path.basename(csv_path))[0]
        output_name = os.path.abspath(f"{input_stem}_raw.csv")
    else:
        output_name = os.path.abspath(args.output)

    print(f"\nRunning dataset: {csv_path}")
    print(f"Reading molecule directories from: {base_dir}")
    print(f"Writing output to: {output_name}")

    molecules = read_extended_csv(
        csv_filename=csv_path,
        use_clustering=True,
        filter_cluster_id=None
    )

    flat_molecules = flatten_molecules(
        molecules,
        use_clustering=True
    )

    get_bonding(flat_molecules)

    initialise_csv(output_name)

    read_txt_xyz(
        flat_molecules,
        base_dir,
        output_filename=output_name,
        save_interval=1000
    )

    # print(f"[INFO] Finished writing: {output_name}\n")