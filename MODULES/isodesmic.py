from collections import Counter
from itertools import combinations_with_replacement
from rdkit import Chem
import json
import pandas as pd
import csv
import os
import re
import numpy as np
"""
TODO:
    - As json self populates, change from isodesmic -> hypohomodesmic ->
    homodesmotic -> hyperhomodesmotic
"""

script_dir = os.path.dirname(os.path.abspath(__file__))

EV_TO_HARTREE = 1.0 / 27.211386245988
HARTREE_TO_KJMOL = 2625.49962


# =========================================================
# ENERGY PARSING
# =========================================================
def read_mace_energy_from_thermo(script_dir, cid):

    folder_name = f"{cid}"
    file_name = f"{cid}_thermo.txt"


    # thermo_path = os.path.join(script_dir, "..", "..", "OPTIMISED_STRUCTURES", "SMALL_MODEL", folder_name, file_name)
    thermo_path = os.path.join(script_dir, "..", "OPTIMISED_STRUCTURES", "30_MOL", folder_name, file_name)

    # print(thermo_path)

    if not os.path.exists(thermo_path):
        return None

    with open(thermo_path, "r") as f:
        text = f.read()

    match = re.search(
        r"Electronic energy E0\s*:\s*(-?\d+\.\d+)\s*eV",
        text
    )

    if not match:
        return None

    energy_eV = float(match.group(1))

    return energy_eV * EV_TO_HARTREE


# =========================================================
# JSON REFERENCE DATA
# =========================================================
def read_json(json_path):
    with open(json_path) as f:
        raw_json = json.load(f)

    json_data = {}

    for name, data in raw_json.items():
        smi = data.get("SMILES")
        if not smi:
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.AddHs(mol)

        counts = Counter()

        for atom in mol.GetAtoms():
            counts[atom.GetSymbol()] += 1

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetSymbol()
            a2 = bond.GetEndAtom().GetSymbol()
            bond_type = str(bond.GetBondType())
            a, b = sorted([a1, a2])
            counts[f"{a}-{b} {bond_type}"] += 1

        json_data[name] = {
            "SMILES": canonical_smi,
            "exp_energy": data.get("exp_energy"),
            "MACE_energy_Ha": data.get("MACE_energy_Ha"),
            "atom_bond_dict": dict(counts)
        }

    print(f"Loaded {len(json_data)} reference molecules")
    return json_data


# =========================================================
# TARGET MOLECULES
# =========================================================
def read_target_mol(csv_path, nrows=None):
    df = pd.read_csv(csv_path, nrows=nrows)
    targets = []

    for _, row in df.iterrows():
        cid = str(row["CID"])
        smiles = row["SMILES"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        mol = Chem.AddHs(mol)

        counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetSymbol()
            a2 = bond.GetEndAtom().GetSymbol()
            bond_type = str(bond.GetBondType())
            a, b = sorted([a1, a2])
            counts[f"{a}-{b} {bond_type}"] += 1

        mace_energy = read_mace_energy_from_thermo(script_dir, cid)
        if mace_energy is None:
            print(f"[SKIPPED] CID {cid} — no thermo energy")
            continue

        targets.append({
            "CID": cid,
            "SMILES": smiles,
            "atom_bond_dict": dict(counts),
            "MACE_energy_Ha": mace_energy
        })

    print(f"Loaded {len(targets)} valid targets")
    return targets

# =========================================================
# DICT OPERATIONS
# =========================================================
def add_dicts(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def subtract_dicts(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) - v
    return out


def clean_dict(d):
    return {k: v for k, v in d.items() if v != 0}


# =========================================================
# BOND TABLE HELPERS
# =========================================================
def sum_atom_bond_dict(species_list, json_data, target_data):
    total = {}

    for s in species_list:
        if s in json_data:
            abd = json_data[s]["atom_bond_dict"]
        else:
            abd = target_data["atom_bond_dict"]

        total = add_dicts(total, abd)

    return clean_dict(total)


def print_bond_table(reactants, products, json_data, target_data):

    R_dict = sum_atom_bond_dict(reactants, json_data, target_data)
    P_dict = sum_atom_bond_dict(products, json_data, target_data)

    all_keys = sorted(set(R_dict.keys()) | set(P_dict.keys()))

    print("Bonding comparison:")
    print(f"{'Type':<20} {'Reactants':>10} {'Products':>10} {'Δ':>10}")
    print("-" * 55)

    for k in all_keys:
        r_val = R_dict.get(k, 0)
        p_val = P_dict.get(k, 0)
        diff = p_val - r_val

        print(f"{k:<20} {r_val:>10} {p_val:>10} {diff:>10}")

    print("-" * 55)


# =========================================================
# FIND ISODESMIC REACTIONS
# =========================================================
def find_isodesmics(json_data, target_data):

    target_dict = target_data["atom_bond_dict"]
    target_smiles = target_data["SMILES"]

    mol_entries = [(name, d["atom_bond_dict"]) for name, d in json_data.items()]

    reverse_lookup = {
        json.dumps(clean_dict(d), sort_keys=True): name
        for name, d in mol_entries
    }

    solutions = []

    for r in range(1, 4):
        for combo in combinations_with_replacement(mol_entries, r):

            total = {}
            for name, abd in combo:
                total = add_dicts(total, abd)

            leftover = subtract_dicts(total, target_dict)

            if any(v < 0 for v in leftover.values()):
                continue

            leftover_clean = clean_dict(leftover)

            reactants = [name for name, _ in combo]

            if not leftover_clean:
                products = [target_smiles]
            else:
                key = json.dumps(leftover_clean, sort_keys=True)
                if key not in reverse_lookup:
                    continue
                products = [target_smiles, reverse_lookup[key]]

            solutions.append({
                "reactants": reactants,
                "products": products,
            })

    return solutions


# =========================================================
# ENERGY CALCULATIONS
# =========================================================
def calculate_hr(solutions, target_energy, json_data, target_smiles):

    for sol in solutions:

        E_reactants = sum(json_data[r]["MACE_energy_Ha"] for r in sol["reactants"])

        E_products = 0.0
        for p in sol["products"]:
            if p in json_data:
                E_products += json_data[p]["MACE_energy_Ha"]
            else:
                E_products += target_energy

        sol["Delta_E_Ha"] = E_products - E_reactants


def calculate_hf(solutions, json_data, target_smiles):

    for sol in solutions:

        reactants = sol["reactants"]
        products = sol["products"]

        other_products = [p for p in products if p != target_smiles]

        sum_reactants = sum(json_data[r]["exp_energy"] for r in reactants)
        sum_products = sum(json_data[p]["exp_energy"] for p in other_products)

        delta_E = sol["Delta_E_Ha"]

        sol["Hf_target"] = sum_reactants - sum_products - delta_E


# =========================================================
# FILTER + WEIGHT
# =========================================================
def filter_and_average(solutions):

    filtered = [
        s for s in solutions
        if abs(s["Delta_E_Ha"]) < 0.005
    ]

    if not filtered:
        return None, None, []

    weights = np.array([1 / (abs(s["Delta_E_Ha"]) + 1e-6) for s in filtered])
    hf_values = np.array([s["Hf_target"] for s in filtered])

    weighted_mean = np.sum(weights * hf_values) / np.sum(weights)
    spread = np.max(hf_values) - np.min(hf_values)

    return weighted_mean, spread, filtered


# =========================================================
# CSV UPDATE
# =========================================================
def update_csv_with_results(csv_path, results):

    df = pd.read_csv(csv_path)
    df["CID"] = df["CID"].astype(str)

    hf_col = "Hf /kJmol-1"
    spread_col = "Hf_spread /kJmol-1"

    # If old columns exist, rename them
    if "Hf_kJmol" in df.columns:
        df.rename(columns={"Hf_kJmol": hf_col}, inplace=True)

    if "Hf_spread_kJmol" in df.columns:
        df.rename(columns={"Hf_spread_kJmol": spread_col}, inplace=True)

    # Create columns if missing
    if hf_col not in df.columns:
        df[hf_col] = None

    if spread_col not in df.columns:
        df[spread_col] = None

    # Overwrite values
    for i, row in df.iterrows():
        cid = str(row["CID"])

        if cid in results:
            df.at[i, hf_col] = results[cid]["Hf_kJmol"]
            df.at[i, spread_col] = results[cid]["Hf_spread_kJmol"]

    df.to_csv(csv_path, index=False)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # csv_path = os.path.join(script_dir, "..", "hades_out.csv")
    csv_path = os.path.join(script_dir, "..", "30_bench.csv")
    json_path = os.path.join(script_dir, "EOF", "isodesmic_bak.json")


    json_data = read_json(json_path)
    targets = read_target_mol(csv_path, nrows=50)   
    # targets = targets[:50]

    results = {}

    for target in targets:

        cid = target["CID"]
        smiles = target["SMILES"]

        print(f"\n=== CID {cid} ===")

        solutions = find_isodesmics(json_data, target)

        if not solutions:
            print("No reactions found")
            results[cid] = {"Hf_kJmol": None, "Hf_spread_kJmol": None}
            continue

        calculate_hr(solutions, target["MACE_energy_Ha"], json_data, smiles)
        calculate_hf(solutions, json_data, smiles)

        mean_hf, spread, filtered = filter_and_average(solutions)

        if not filtered:
            print("No good reactions after filtering")
            results[cid] = {"Hf_kJmol": None, "Hf_spread_kJmol": None}
            continue

        print(f"Valid reactions: {len(filtered)}")
        print(f"Mean Hf: {mean_hf * HARTREE_TO_KJMOL:.2f} kJ/mol")
        print(f"Spread : {spread * HARTREE_TO_KJMOL:.2f} kJ/mol")

        results[cid] = {
            "Hf_kJmol": mean_hf * HARTREE_TO_KJMOL,
            "Hf_spread_kJmol": spread * HARTREE_TO_KJMOL
        }

        print("\nTop reactions:\n")

        for sol in filtered[:5]:
            R = " + ".join(sol["reactants"])
            P = " + ".join(sol["products"])

            delta_e_kj = sol['Delta_E_Ha'] * HARTREE_TO_KJMOL
            hf_kj = sol['Hf_target'] * HARTREE_TO_KJMOL

            print(f"{R} -> {P}")
            print(f"ΔE: {delta_e_kj:.2f} kJ/mol  Hf: {hf_kj:.2f} kJ/mol\n")

            print_bond_table(sol["reactants"], sol["products"], json_data, target)
            print("\n")

    update_csv_with_results(csv_path, results)
