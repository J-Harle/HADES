import os
import csv
import urllib.request
from collections import Counter
from rdkit import Chem
from ase.io import read
from mace.calculators import MACECalculator  # pip install mace-torch

# --- Atomic and Bond contributions (Eh in Hartree) ---
contrib_dict = {
    "C"                   : -23.165862,
    "C_arom"              : -17.201853,
    "H"                   : -3.756982,
    "N"                   : -41.401615,
    "O"                   : -52.247616,
    "C - C AROMATIC"      : -17.201853,
    "C - C DOUBLE"        : -14.983325,
    "C - C SINGLE"        : -7.492260,
    "C - H SINGLE"        : -0.580309,
    "C - N SINGLE"        : -17.179480,
    "C - O SINGLE"        : -17.187692,
    "H - N SINGLE"        :  3.178411,
    "H - O SINGLE"        : -6.355084,
    "N - N DOUBLE"        : -13.389528,
    "N - N SINGLE"        :  0.025311,
    "N - O DOUBLE"        : -22.931840,
    "N - O SINGLE"        : -22.960691,
}

# Conversion factors
HARTREE_TO_KCAL_MOL = 627.50947415
EV_TO_HARTREE = 1 / 27.211386245988
HARTREE_TO_KJMOL = 2625.5


script_dir = os.path.dirname(os.path.abspath(__file__))
xyz_dir = os.path.join(script_dir, "../OPTIMISED_STRUCTURES")

# ---------------------------------------------------------------------
# Setup the MACE calculator
# ---------------------------------------------------------------------
def get_mace_calculator(script_dir):
    calc_dir = os.path.join(script_dir, "..", "CALCULATORS")
    os.makedirs(calc_dir, exist_ok=True)
    model_file = os.path.join(calc_dir, "MACE-OFF23_large.model")

    if not os.path.exists(model_file):
        print(f"Missing model; downloading to {model_file}")
        url = "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_large.model?raw=true"
        urllib.request.urlretrieve(url, model_file)
        print("Download complete.")

    calc = MACECalculator(
        model_paths=[model_file],
        dispersion=False,
        default_dtype="float64",
        device="cpu"
    )
    return calc

# ---------------------------------------------------------------------
# Process xyz files and compute single-point energies
# ---------------------------------------------------------------------
def process_xyz_files(xyz_dir):
    calc = get_mace_calculator(script_dir)
    results = []

    for root, _, files in os.walk(xyz_dir):
        xyz_files = [f for f in files if f.endswith(".xyz")]

        for xyz_file in xyz_files:
            xyz_path = os.path.join(root, xyz_file)
            with open(xyz_path, "r") as f:
                lines = f.readlines()

            if len(lines) < 2:
                print(f"Warning: {xyz_file} is missing SMILES line.")
                continue

            parts = lines[1].strip().split()
            smiles = parts[1] if len(parts) > 1 else None
            if not smiles:
                print(f"Warning: {xyz_file} missing SMILES entry.")
                continue

            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)

            print("=" * 70)
            print(f"Processing: {xyz_file}")
            print(f"SMILES: {smiles}\n")

            # --- Atom counting ---
            atom_counts = Counter()
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if atom.GetIsAromatic():
                    symbol += "_arom"
                atom_counts[symbol] += 1

            # --- Bond counting ---
            bond_counter = Counter()
            for bond in mol.GetBonds():
                a1, a2 = bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()
                atoms = sorted([a1, a2])
                bond_type = str(bond.GetBondType())
                bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
                bond_counter[bond_label] += 1

            # --- Compute single-point energy with MACE ---
            try:
                atoms = read(xyz_path)
                atoms.calc = calc
                energy_eV = atoms.get_potential_energy()  # in eV
                energy_Ha = energy_eV * EV_TO_HARTREE

                # Compute total contributions
                atom_contrib_total = sum(
                    count * contrib_dict.get(atom, 0.0)
                    for atom, count in atom_counts.items()
                )

                bond_contrib_total = sum(
                    count * contrib_dict.get(bond, 0.0)
                    for bond, count in bond_counter.items()
                )

                total_contrib = atom_contrib_total + bond_contrib_total
                corrected_energy_Ha = energy_Ha - total_contrib

                # Convert to kJ/mol
                eof_kJmol = corrected_energy_Ha * HARTREE_TO_KJMOL

                results.append({
                    "Filename": xyz_file,
                    "SMILES": smiles,
                    "EoF / kJmol-1": eof_kJmol
                })

                print(f"MACE Potential Energy: {energy_eV:.6f} eV  =  {energy_Ha:.6f} Ha")
                print(f"Corrected Energy: {corrected_energy_Ha:.6f} Ha  =  {eof_kJmol:.2f} kJ/mol\n")

            except Exception as e:
                print(f"Error computing energy for {xyz_file}: {e}\n")

    write_to_csv(script_dir, results)


def write_to_csv(script_dir, results):
    csv_path = os.path.join(script_dir, "..", "hades_out.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Filename", "SMILES", "EoF / kJmol-1"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
                            


if __name__ == "__main__":
    process_xyz_files(xyz_dir)