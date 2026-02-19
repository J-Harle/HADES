import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from rdkit import Chem

import sys
csv.field_size_limit(sys.maxsize)  # handle large JSON arrays

script_dir = os.path.dirname(os.path.abspath(__file__))


def read_data_from_csv(csv_name="raw.csv"):
    csv_path = os.path.join(script_dir, csv_name)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV not found: {csv_path}")

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mol_data = {}
            for key, value in row.items():
                # Attempt to parse JSON for array-like columns
                try:
                    parsed_value = json.loads(value)
                    mol_data[key] = parsed_value
                except (json.JSONDecodeError, TypeError):
                    # Keep as string or float
                    try:
                        mol_data[key] = float(value)
                    except (ValueError, TypeError):
                        mol_data[key] = value
            data.append(mol_data)

    # print(f"[INFO] Loaded {len(data)} molecules from CSV\n")
    return data


def plot_dos(frequency_axis, dos, dpi):
    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.plot(frequency_axis, dos)
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    plt.ylabel("g(ω)", fontsize=24)
    plt.xlim(left=0)
    plt.tick_params(axis="both", labelsize=16)
    plt.tight_layout()
    plt.show()


def plot_be_dos(frequency_axis, be_dos, dpi):
    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.plot(frequency_axis, be_dos)
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    plt.ylabel("g(ω)", fontsize=24)
    plt.xlim(left=0)
    plt.tick_params(axis="both", labelsize=16)
    plt.tight_layout()
    plt.show()


def plot_first_convolution(
    frequency_axis,
    be_dos_200,
    first_conv_freq_axis,
    first_conv_projection,
    dpi
):
    fig, ax1 = plt.subplots(figsize=(12, 9), dpi=dpi)

    ax1.plot(frequency_axis, first_conv_projection, label="Projected First Convolution")
    ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    ax1.set_ylabel("First Convolution / g(ω)", fontsize=24)
    ax1.set_xlim(left=0)
    ax1.tick_params(axis="both", labelsize=16)

    be_interp = interp1d(frequency_axis, be_dos_200, bounds_error=False, fill_value=0.0)
    be_extended = be_interp(first_conv_freq_axis)

    ax2 = ax1.twinx()
    ax2.plot(first_conv_freq_axis, be_extended, color="#FFA500", label="BE Scaled DOS")
    ax2.set_ylabel("BE Scaled DOS / g(ω)", fontsize=24, color="#FFA500")
    ax2.tick_params(axis="y", labelcolor="#FFA500", labelsize=16)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.grid(False)
    plt.tight_layout()
    plt.show()


def plot_second_convolution(
    frequency_axis,
    be_dos,
    first_conv_freq_axis,
    second_conv_projection,
    molecule,
    dpi
):
    fig, ax1 = plt.subplots(figsize=(12, 9), dpi=dpi)
    ax1.plot(frequency_axis, second_conv_projection, color="#000000", label="Second Convolution (projected)")
    ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    ax1.set_ylabel("Second Convolution / g(ω)", fontsize=24)
    ax1.set_xlim(left=0)
    ax1.tick_params(axis="both", labelsize=16)

    be_interp = interp1d(frequency_axis, be_dos, bounds_error=False, fill_value=0.0)
    be_extended = be_interp(first_conv_freq_axis)

    ax2 = ax1.twinx()
    ax2.plot(first_conv_freq_axis, be_extended, color="#FFA500", label="BE Scaled DOS")
    ax2.set_ylabel("BE Scaled DOS / g(ω)", fontsize=24, color="#FFA500")
    ax2.tick_params(axis="y", labelcolor="#FFA500", labelsize=16)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(molecule, fontsize=24)
    ax1.grid(False)
    plt.tight_layout()
    plt.show()


def plot_final(all_data, dpi):
    h50s = []
    areas = []

    for mol in all_data:
        exp_ratio = mol.get("exp_ratio")
        raw_integral = mol.get("raw_integral")
        area = raw_integral * exp_ratio 
        # area = mol.get("exp_ratio_integral") # This is the main version
        h50 = mol.get("H50")
        if area is None or h50 is None:
            continue
        h50s.append(h50)
        areas.append(area)

    h50s = np.array(h50s)
    areas = np.array(areas)

    inv_h50 = 1.0 / h50s
    coeffs = np.polyfit(inv_h50, areas, deg=1)
    a, b = coeffs
    predicted = a * inv_h50 + b
    r_squared = r2_score(areas, predicted)

    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.scatter(h50s, areas)
    # Sort for smooth line
    sorted_idx = np.argsort(h50s)
    plt.plot(h50s[sorted_idx], predicted[sorted_idx], '--', color="gray", linewidth=2)
    plt.xlabel("Experimental impact sensitivity / J", fontsize=24)
    plt.ylabel("Up-pumped metric / arb.", fontsize=24)
    plt.tick_params(axis="both", labelsize=18)
    plt.grid(True)
    # plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()

def summary_table(all_data):
    headers = [
        "Molecule",
        "Omega_max",
        "Exp_ratio",
        "Raw_integral",
        "H50",
        "Integral * exp_ratio"
    ]

    # --- Sort by ascending H50 (None values last) ---
    sorted_data = sorted(
        all_data,
        key=lambda m: (m.get("H50") is None, m.get("H50"))
    )

    rows = []

    for mol in sorted_data:
        omega = mol.get("omega_max")
        exp_ratio = mol.get("exp_ratio")
        integral = mol.get("raw_integral")
        h50 = mol.get("H50")

        product = (
            exp_ratio * integral
            if exp_ratio is not None and integral is not None
            else None
        )

        rows.append([
            str(mol.get("molecule", "")),
            omega,
            exp_ratio,
            integral,
            h50,
            product
        ])

    formatted_rows = []
    for row in rows:
        formatted_rows.append([
            row[0],
            f"{row[1]:.2f}" if row[1] is not None else "",
            f"{row[2]:.3f}" if row[2] is not None else "",
            f"{row[3]:.3f}" if row[3] is not None else "",
            f"{row[4]:.2f}" if row[4] is not None else "",
            f"{row[5]:.3f}" if row[5] is not None else "",
        ])

    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in formatted_rows))
        for i in range(len(headers))
    ]

    header_line = "  ".join(
        headers[i].ljust(col_widths[i]) for i in range(len(headers))
    )
    print(header_line)
    print("-" * len(header_line))

    for row in formatted_rows:
        aligned = [
            row[i].ljust(col_widths[i]) if i == 0
            else row[i].rjust(col_widths[i])
            for i in range(len(row))
        ]
        print("  ".join(aligned))

def plot_atom_count_vs_h50(all_data):
    heavy_atoms = []
    h50_values = []
    

    for mol in all_data:
        h50 = mol.get("H50")
        exp_ratio = mol.get("exp_ratio")
        raw_integral = mol.get("raw_integral")
        area = raw_integral * exp_ratio 

        smiles = mol.get("SMILES")

        if h50 is None or smiles is None:
            continue

        rd_mol = Chem.MolFromSmiles(smiles)
        if rd_mol is None:
            continue

        num_heavy_atoms = rd_mol.GetNumHeavyAtoms()

        heavy_atoms.append(num_heavy_atoms)
        h50_values.append(area)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(heavy_atoms, h50_values)
    plt.xlabel("Number of Heavy Atoms")
    plt.ylabel("Uppumped metric /arb.")
    plt.title("Heavy Atom Count vs H50")
    plt.tight_layout()
    plt.show()
    
def plot_total_atom_count_vs_h50(all_data):
    total_atoms = []
    h50_values = []

    for mol in all_data:
        h50 = mol.get("H50")
        exp_ratio = mol.get("exp_ratio")
        raw_integral = mol.get("raw_integral")
        area = raw_integral * exp_ratio 

        smiles = mol.get("SMILES")

        if h50 is None or smiles is None:
            continue

        rd_mol = Chem.MolFromSmiles(smiles)
        if rd_mol is None:
            continue

        # Add explicit hydrogens
        rd_mol_with_h = Chem.AddHs(rd_mol)

        num_atoms = rd_mol_with_h.GetNumAtoms()

        total_atoms.append(num_atoms)
        h50_values.append(h50)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(total_atoms, area)
    plt.xlabel("Total Atom Count (including H)")
    plt.ylabel("Uppumped metric /Arb")
    plt.title("Total Atom Count vs H50")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    dpi = 100

    all_data = read_data_from_csv("raw.csv")

    # plot_dos(mol["frequency_axis"], mol.get("dos"), dpi)
    # plot_be_dos(mol["frequency_axis"], mol.get("be_dos"), dpi)
    # plot_first_convolution(
    #     mol["frequency_axis"],
    # mol["be_dos_200"],
    # mol["first_conv_freq_axis"],
    # mol["first_convolved_projection"],
    # dpi
    # )

    summary_table(all_data)
    plot_final(all_data, dpi)
    plot_atom_count_vs_h50(all_data)
    # plot_total_atom_count_vs_h50(all_data)
