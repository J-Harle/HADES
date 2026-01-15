import json
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares
from rdkit import Chem
from collections import Counter
import numpy as np
from sklearn.metrics import r2_score
import os
import sys

HA_TO_KJMOL = 2625.499639

# === Initial constraints used for sensible p0 guesses ===
# Cant constrain all bonds as results in 10^9 magnitude
initial_atom_constraints = {
    'C': 0.0010,
    'H': -0.0005,
    'O(-1)': 0.0050,
    'N(+1)': 0.0020,
    'C_arom': 0.0008,
}

initial_bond_constraints = {
    'C - C SINGLE': 0.0001,
    'C - H SINGLE': -0.0001,
    'N - O DOUBLE': 0.0005,
}
# ===============================================================


def load_json_split(path):

    with open(path, "r") as f:
        raw = json.load(f)

    fitting = {}
    test = {}

    def normalise_entry(name, data):
        # Try multiple possible keys for MACE energy and experimental energy
        mace_keys = ["MACE_energy_Ha", "MACE energy", "MACE_energy", "MACE_energy_ha"]
        exp_keys = ["Exp_energy", "Experiment_Energy", "Experiment_Energy_Ha", "EXP_ENERGY"]
        mace_val = None
        exp_val = None

        for k in mace_keys:
            if k in data:
                mace_val = data[k]
                break

        for k in exp_keys:
            if k in data:
                exp_val = data[k]
                break

        # Defensive conversion: if energies are strings, try to cast to float
        try:
            if mace_val is not None:
                mace_val = float(mace_val)
        except Exception:
            print(f"Warning: could not convert MACE energy for {name} to float: {mace_val}")
            mace_val = None

        try:
            if exp_val is not None:
                exp_val = float(exp_val)
        except Exception:
            print(f"Warning: could not convert Experimental energy for {name} to float: {exp_val}")
            exp_val = None

        smiles = data.get("SMILES") or data.get("smiles") or data.get("Smiles")
        if smiles is None:
            raise KeyError(f"SMILES not found for molecule '{name}' in JSON.")

        entry = {
            "SMILES": smiles,
            "MACE energy": mace_val,
            "Experiment_Energy": exp_val
        }
        return entry

    # If top-level contains 'fitting' and/or 'test', use those buckets
    if isinstance(raw, dict) and ("fitting" in raw or "test" in raw):
        # If both present or one present, iterate
        for bucket_name in ("fitting", "test"):
            if bucket_name not in raw:
                continue
            bucket = raw[bucket_name]
            if not isinstance(bucket, dict):
                raise ValueError(f"Top-level key '{bucket_name}' must map to an object of molecule entries.")
            for name, data in bucket.items():
                try:
                    entry = normalise_entry(name, data)
                except KeyError as e:
                    print(f"Skipping {name}: {e}")
                    continue
                if bucket_name == "fitting":
                    fitting[name] = entry
                else:
                    test[name] = entry

    else:
        # Flat mapping: each entry must contain a 'flag' telling fitting/test
        for name, data in raw.items():
            if not isinstance(data, dict):
                print(f"Warning: skipping {name} because its value is not an object.")
                continue
            if "flag" not in data:
                print(f"Warning: no 'flag' for {name}; skipping.")
                continue
            flag = str(data["flag"]).lower()
            try:
                entry = normalise_entry(name, data)
            except KeyError as e:
                print(f"Skipping {name}: {e}")
                continue

            if flag.startswith("f"):
                fitting[name] = entry
            elif flag.startswith("t"):
                test[name] = entry
            else:
                print(f"Warning: unknown flag '{data['flag']}' for {name}; skipping.")

    # Basic warnings for missing energies
    missing_mace = [n for n, d in list(fitting.items()) + list(test.items()) if d["MACE energy"] is None]
    missing_exp = [n for n, d in list(fitting.items()) + list(test.items()) if d["Experiment_Energy"] is None]
    if missing_mace:
        print(f"Warning: {len(missing_mace)} molecules missing MACE energy (will cause errors if used): {missing_mace[:5]}")
    if missing_exp:
        print(f"Warning: {len(missing_exp)} molecules missing experimental energy: {missing_exp[:5]}")

    return fitting, test


# === Dynamic atom and bond label generation ===
def get_dynamic_labels(*smiles_dicts):
    """
    Determine all unique atom and bond labels present in one or more SMILES dictionaries.
    """
    atom_labels = set()
    bond_labels = set()

    for smiles_dict in smiles_dicts:
        for name, data in smiles_dict.items():
            smiles = data["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: could not parse SMILES for {name}")
                continue
            mol = Chem.AddHs(mol)

            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                charge = atom.GetFormalCharge()
                label = f"{symbol}({charge:+d})" if charge != 0 else symbol

                if atom.GetIsAromatic() and symbol in ["C", "N", "O"]:
                    label += "_arom"

                atom_labels.add(label)

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                a1_label = f"{a1.GetSymbol()}({a1.GetFormalCharge():+d})" if a1.GetFormalCharge() != 0 else a1.GetSymbol()
                a2_label = f"{a2.GetSymbol()}({a2.GetFormalCharge():+d})" if a2.GetFormalCharge() != 0 else a2.GetSymbol()
                atoms = sorted([a1_label, a2_label])
                bond_type = str(bond.GetBondType())
                bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
                bond_labels.add(bond_label)

    return sorted(atom_labels), sorted(bond_labels)


def add_charge_resonance(mol):
    """
    Adjust formal charges to represent nitro-like resonance structures.
    """
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N":
            oxygens = [nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "O"]
            if len(oxygens) == 2:
                bond_types = [mol.GetBondBetweenAtoms(atom.GetIdx(), o.GetIdx()).GetBondType() for o in oxygens]
                if set(bond_types) == {Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE}:
                    atom.SetFormalCharge(+1)
                    for o in oxygens:
                        if mol.GetBondBetweenAtoms(atom.GetIdx(), o.GetIdx()).GetBondType() == Chem.rdchem.BondType.SINGLE:
                            o.SetFormalCharge(-1)
    return mol


# === Feature vector builder ===
def build_feature_vectors(SMILES_dict, atom_labels, bond_labels):
    feature_vectors = {}
    atom_counts_dict = {}

    for name, data in SMILES_dict.items():
        smiles = data["SMILES"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: could not parse SMILES for {name}")
            continue
        mol = Chem.AddHs(mol)

        atom_counter = Counter()
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            charge = atom.GetFormalCharge()
            label = f"{symbol}({charge:+d})" if charge != 0 else symbol
            if atom.GetIsAromatic() and symbol in ["C", "N", "O"]:
                label += "_arom"
            atom_counter[label] += 1

        bond_counter = Counter()
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            a1_label = f"{a1.GetSymbol()}({a1.GetFormalCharge():+d})" if a1.GetFormalCharge() != 0 else a1.GetSymbol()
            a2_label = f"{a2.GetSymbol()}({a2.GetFormalCharge():+d})" if a2.GetFormalCharge() != 0 else a2.GetSymbol()
            atoms = sorted([a1_label, a2_label])
            bond_type = str(bond.GetBondType())
            bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
            bond_counter[bond_label] += 1

        simple_atom_counter = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
        atom_counts_dict[name] = dict(simple_atom_counter)

        atom_vector = [atom_counter.get(a, 0) for a in atom_labels]
        bond_vector = [bond_counter.get(b, 0) for b in bond_labels]
        feature_vectors[name] = atom_vector + bond_vector

    return feature_vectors, atom_counts_dict


def fit_atomic_bond_contributions(fitting_dict, atom_labels, bond_labels, show_plot=True):
    names, X, y_corr, y_exp, y_mace = [], [], [], [], []

    for name, data in fitting_dict.items():
        try:
            mol = Chem.MolFromSmiles(data["SMILES"])
            if mol is None:
                print(f"Could not parse SMILES for {name}. Skipping.")
                continue

            mol = Chem.AddHs(mol)
            mol = add_charge_resonance(mol)

            atom_counts = np.zeros(len(atom_labels))
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                charge = atom.GetFormalCharge()
                label = f"{symbol}({charge:+d})" if charge != 0 else symbol
                if atom.GetIsAromatic() and symbol in ["C", "N", "O"]:
                    label += "_arom"
                if label in atom_labels:
                    atom_counts[atom_labels.index(label)] += 1

            bond_counts = np.zeros(len(bond_labels))
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                a1_label = f"{a1.GetSymbol()}({a1.GetFormalCharge():+d})" if a1.GetFormalCharge() != 0 else a1.GetSymbol()
                a2_label = f"{a2.GetSymbol()}({a2.GetFormalCharge():+d})" if a2.GetFormalCharge() != 0 else a2.GetSymbol()
                atoms = sorted([a1_label, a2_label])
                bond_type = str(bond.GetBondType())
                bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
                if bond_label in bond_labels:
                    bond_counts[bond_labels.index(bond_label)] += 1

            features = np.concatenate([atom_counts, bond_counts])
            X.append(features)

            mace_e = data["MACE energy"]
            exp_e = data["Experiment_Energy"]

            if mace_e is None or exp_e is None:
                print(f"Skipping {name} due to missing energies (MACE: {mace_e}, Exp: {exp_e})")
                continue

            y_corr.append(mace_e - exp_e)
            y_exp.append(exp_e)
            y_mace.append(mace_e)
            names.append(name)

        except Exception as e:
            print(f"Error processing {name}: {e}")

    X = np.array(X)
    y_corr = np.array(y_corr)
    y_exp = np.array(y_exp)
    y_mace = np.array(y_mace)

    if X.shape[0] == 0:
        raise ValueError("No valid data found in fitting_dict.")

    # --- Least Squares Fitting with Initial Constraints ---
    # Append a column of ones to X for the intercept term (c_intercept)
    X_augmented = np.hstack([X, np.ones((X.shape[0], 1))])

    # The parameters vector p will be [c1, c2, ..., cn, c_intercept]
    n_params = X_augmented.shape[1]

    # 1. Combine Atom and Bond constraints for lookup
    initial_constraints = {**initial_atom_constraints, **initial_bond_constraints}

    # 2. Build the initial guess vector (p0) using constraints
    p0_coeffs = []
    all_labels = atom_labels + bond_labels
    for label in all_labels:
        # Use the constraint value if available, otherwise use 0.0
        p0_coeffs.append(initial_constraints.get(label, 0.0))

    # Initial guess for the intercept (defaulting to 0.0)
    p0_intercept = 0.0

    # The complete initial guess vector for least_squares
    p0 = np.array(p0_coeffs + [p0_intercept])
    print(f"Initial guess vector (p0) created using {len(p0_coeffs)} constraints and 1 intercept.")
    # =========================================================================================

    # Define the residual function: Model Prediction - Target
    def residual_func(p, X_aug, y_target):
        return (X_aug @ p) - y_target

    # Perform the least-squares minimization
    res = least_squares(residual_func, p0, args=(X_augmented, y_corr), method='lm')

    # Extract the coefficients and intercept
    coeffs_all = res.x
    coeffs = coeffs_all[:-1]
    intercept = coeffs_all[-1]

    # --- Calculate Predicted Values and R-squared ---
    y_pred_corr = X_augmented @ coeffs_all
    y_pred_mace_corr = y_mace - y_pred_corr
    y_pred_exp = y_pred_mace_corr
    r2_fit = r2_score(y_exp, y_pred_exp)

    # ===== Fitting Table (converted to kJ/mol) =====
    print("\n===== Fitting Set Performance (SciPy least_squares) =====")
    print(f"R² (fit) = {r2_fit:.4f}")
    print(f"{'Molecule':25s}  {'Pred (kJ/mol)':>14s}  {'Exp (kJ/mol)':>14s}  "
          f"{'Δ (kJ/mol)':>14s}  {'% Error':>10s}")
    print("-" * 90)

    results = []
    for n, y_p, y_e in zip(names, y_pred_exp, y_exp):
        delta = y_p - y_e
        percent_error = (delta / abs(y_e)) * 100 if y_e != 0 else float('nan')
        results.append((n, y_p, y_e, delta, percent_error))

    results.sort(key=lambda x: abs(x[4]), reverse=True)

    for n, y_p, y_e, delta, percent_error in results:
        print(f"{n:25s}  "
              f"{y_p*HA_TO_KJMOL: 14.2f}  "
              f"{y_e*HA_TO_KJMOL: 14.2f}  "
              f"{delta*HA_TO_KJMOL: 14.2f}  "
              f"{percent_error: 10.2f}")

    # ===== Coeff Summary (Hartrees) =====
    print("\n===== Linear Fit Summary (SciPy least_squares) =====")
    print(f"R² (fit) = {r2_fit:.4f}")
    print(f"Intercept = {intercept:.6f} Ha")
    print("\nCoefficients (Ha per atom/bond):")
    all_labels = atom_labels + bond_labels
    for i, label in enumerate(all_labels):
        print(f"{label:>12s}: {coeffs[i]: .6f}")

    # ===== Plotting in kJ/mol (Same as original) =====
    if show_plot:
        y_exp_kj = y_exp * HA_TO_KJMOL
        y_pred_exp_kj = y_pred_exp * HA_TO_KJMOL

        # Percentage error
        percent_errors = abs((y_pred_exp - y_exp) / np.abs(y_exp) * 100)

        fig, ax1 = plt.subplots(figsize=(10, 10))
        ax1.scatter(y_exp_kj, y_pred_exp_kj, c='black', s=50, alpha=0.7)

        lims = [
            min(np.min(y_exp_kj), np.min(y_pred_exp_kj)),
            max(np.max(y_exp_kj), np.max(y_pred_exp_kj))
        ]
        ax1.plot(lims, lims, 'k--', linewidth=1.5)
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.grid(alpha=0.3)

        ax1.set_xlabel("Experimental Enthalpy of Formation / kJ mol⁻¹")
        ax1.set_ylabel("Predicted Enthalpy of Formation / kJ mol⁻¹")
        ax1.set_title(f"Fitting Set (SMALL Model)\nR² = {r2_fit:.4f}")

        # Second axis: percentage error
        ax2 = ax1.twinx()
        ax2.scatter(y_exp_kj, percent_errors, c='red', s=40, alpha=0.8)
        ax2.set_ylabel("Prediction Error / %", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # for i, n in enumerate(names):
        #     ax1.text(y_exp_kj[i], y_pred_exp_kj[i], n, fontsize=8,
        #             ha='right', va='bottom', alpha=0.7)

        plt.tight_layout()
        plt.show()
        contributions = {n: c for n, c in zip(names, y_pred_corr)}
        return coeffs, contributions, y_pred_exp, r2_fit, intercept


def predict_test_set(test_dict, atom_labels, bond_labels, coeffs, intercept=0.0, show_plot=True):
    names, X_test, y_mace, y_exp = [], [], [], []
    for name, data in test_dict.items():
        try:
            mol = Chem.MolFromSmiles(data["SMILES"])
            if mol is None:
                print(f"Could not parse SMILES for {name}. Skipping.")
                continue

            mol = Chem.AddHs(mol)
            mol = add_charge_resonance(mol)

            atom_counts = np.zeros(len(atom_labels))
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                charge = atom.GetFormalCharge()
                label = f"{symbol}({charge:+d})" if charge != 0 else symbol
                if atom.GetIsAromatic() and symbol in ["C", "N", "O"]:
                    label += "_arom"
                if label in atom_labels:
                    atom_counts[atom_labels.index(label)] += 1

            bond_counts = np.zeros(len(bond_labels))
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                a1_label = f"{a1.GetSymbol()}({a1.GetFormalCharge():+d})" if a1.GetFormalCharge() != 0 else a1.GetSymbol()
                a2_label = f"{a2.GetSymbol()}({a2.GetFormalCharge():+d})" if a2.GetFormalCharge() != 0 else a2.GetSymbol()
                atoms = sorted([a1_label, a2_label])
                bond_type = str(bond.GetBondType())
                bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
                if bond_label in bond_labels:
                    bond_counts[bond_labels.index(bond_label)] += 1

            features = np.concatenate([atom_counts, bond_counts])
            X_test.append(features)
            y_mace.append(data["MACE energy"])
            y_exp.append(data["Experiment_Energy"])
            names.append(name)

        except Exception as e:
            print(f"Error processing {name}: {e}")

    X_test = np.array(X_test)
    y_mace = np.array(y_mace)
    y_exp = np.array(y_exp)

    expected_len = len(atom_labels) + len(bond_labels)
    if X_test.shape[1] != expected_len:
        raise ValueError(
            f"Feature length mismatch: expected {expected_len}, got {X_test.shape[1]}."
        )

    y_corr_pred = X_test @ coeffs + intercept
    y_pred_exp = y_mace - y_corr_pred
    r2_test = r2_score(y_exp, y_pred_exp)

    # ===== Test Table (converted to kJ/mol) =====
    print("\n===== Test Set Performance =====")
    print(f"R² (test) = {r2_test:.4f}")
    print(f"{'Molecule':25s}  {'Pred (kJ/mol)':>14s}  {'Exp (kJ/mol)':>14s}  {'Δ (kJ/mol)':>14s}  {'% Error':>10s}")
    print("-" * 90)

    results = []
    for n, y_p, y_e in zip(names, y_pred_exp, y_exp):
        delta = y_p - y_e
        percent_error = (delta / abs(y_e)) * 100 if y_e != 0 else float('nan')
        results.append((n, y_p, y_e, delta, percent_error))

    results.sort(key=lambda x: abs(x[4]), reverse=True)

    for n, y_p, y_e, delta, percent_error in results:
        print(f"{n:25s}  "
              f"{y_p*HA_TO_KJMOL: 14.2f}  "
              f"{y_e*HA_TO_KJMOL: 14.2f}  "
              f"{delta*HA_TO_KJMOL: 14.2f}  "
              f"{percent_error: 10.2f}")

        # ===== Plotting in kJ/mol =====
    if show_plot:
        y_exp_kj = y_exp * HA_TO_KJMOL
        y_pred_exp_kj = y_pred_exp * HA_TO_KJMOL

        # Percentage error (%)
        percent_errors = abs((y_pred_exp - y_exp) / np.abs(y_exp) * 100)

        fig, ax1 = plt.subplots(figsize=(10, 10))
        ax1.scatter(y_exp_kj, y_pred_exp_kj, c='black', s=50, alpha=0.7)

        lims = [
            min(np.min(y_exp_kj), np.min(y_pred_exp_kj)),
            max(np.max(y_exp_kj), np.max(y_pred_exp_kj))
        ]
        ax1.plot(lims, lims, 'k--')
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.grid(alpha=0.3)

        ax1.set_xlabel("Experimental Enthalpy of Formation / kJ mol⁻¹")
        ax1.set_ylabel("Predicted Enthalpy of Formation / kJ mol⁻¹")
        ax1.set_title(f"Test Set (SMALL Model)\nR² = {r2_test:.4f}")

        # Second axis: percentage error
        ax2 = ax1.twinx()
        ax2.scatter(y_exp_kj, percent_errors, c='red', s=40, alpha=0.8)
        ax2.set_ylabel("Prediction Error / %", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # for i, n in enumerate(names):
        #     ax1.text(y_exp_kj[i], y_pred_exp_kj[i], n, fontsize=8,
        #             ha='right', va='bottom', alpha=0.7)

        plt.tight_layout()
        plt.show()

    return y_pred_exp, r2_test


# === Run everything ===
if __name__ == "__main__":
    json_path = "eof.json"
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found in {os.getcwd()}. Please create the JSON (see earlier examples).")
        sys.exit(1)

    fitting_dict, test_dict = load_json_split(json_path)

    print(f"Loaded {len(fitting_dict)} fitting molecules and {len(test_dict)} test molecules from {json_path}.")

    atom_labels, bond_labels = get_dynamic_labels(fitting_dict, test_dict)
    print(f"Discovered {len(atom_labels)} atom labels and {len(bond_labels)} bond labels.")

    coeffs, contributions, y_pred, r2_fit, intercept = fit_atomic_bond_contributions(
        fitting_dict, atom_labels, bond_labels
    )
    y_pred_test, r2_test = predict_test_set(
        test_dict, atom_labels, bond_labels, coeffs, intercept=intercept
    )

    print(f"\nTotal molecules in fitting set used: {len(fitting_dict)}")
    print(f"Total molecules in test set used: {len(test_dict)}")
    print(f"R² (fit) = {r2_fit:.4f}, R² (test) = {r2_test:.4f}")
