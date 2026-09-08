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

def create_plot_directories(dataset_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir))

    figures_root = os.path.join(root_dir, "..", "FIGURES", "v_1_0_3")

    # Decide base directory depending on dataset
    if dataset_name == "30_bench":
        base_dir = os.path.join(figures_root, "30_MOL_REFERENCE")
    else:
        base_dir = figures_root

    subdirs = [
        "dos",
        "be_dos",
        "first_convolution",
        "second_convolution",
        "final_h50_scatter",
        "atom_count_vs_IS_pred",
    ]

    try:
        os.makedirs(base_dir, exist_ok=True)

        print("\nSubdirectories prepared:")
        for subdir in subdirs:
            full_path = os.path.join(base_dir, subdir)
            os.makedirs(full_path, exist_ok=True)
            print(f"  - {full_path}")

    except Exception as e:
        print(f"Warning: Could not create plot directories: {e}")

    return base_dir
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


def plot_dos(frequency_axis, dos, dpi, save_path=None):
    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.plot(frequency_axis, dos, color="#000000")
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    plt.ylabel("g(ω)", fontsize=24)
    plt.xlim(left=0)
    plt.tick_params(axis="both", labelsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_be_dos(frequency_axis, be_dos, omega_max, dpi, save_path=None):
    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.plot(frequency_axis, be_dos, color="#000000")

    # Omega max guide line
    plt.axvline(
        omega_max,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=r"$\Omega_{\max}$"
    )

    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    plt.ylabel("g(ω)", fontsize=24)
    plt.xlim(left=0)
    plt.tick_params(axis="both", labelsize=16)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_first_convolution(
    frequency_axis,
    be_dos_200,
    first_conv_freq_axis,
    first_conv_projection,
    dpi,
    save_path=None
):

    frequency_axis = np.asarray(frequency_axis)
    be_dos_200 = np.asarray(be_dos_200)

    # --- force equal length ---
    n = min(len(frequency_axis), len(be_dos_200))
    frequency_axis = frequency_axis[:n]
    be_dos_200 = be_dos_200[:n]
    ########
    fig, ax1 = plt.subplots(figsize=(12, 9), dpi=dpi)

    ax1.plot(frequency_axis, first_conv_projection, label="Projected First Convolution", color="#000000")
    ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    ax1.set_ylabel("First Convolution / g(ω)", fontsize=24)
    ax1.set_xlim(left=0)
    ax1.tick_params(axis="both", labelsize=16)

    be_interp = interp1d(frequency_axis, be_dos_200, bounds_error=False, fill_value=0.0)
    be_extended = be_interp(first_conv_freq_axis)

    ax2 = ax1.twinx()
    ax2.plot(first_conv_freq_axis, be_extended, color="red", label="BE Scaled DOS", alpha=0.5)
    ax2.set_ylabel("BE Scaled DOS / g(ω)", fontsize=24, color="red")
    ax2.tick_params(axis="y", labelcolor="red", labelsize=16)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_second_convolution(
    frequency_axis,
    be_dos,
    first_conv_freq_axis,
    second_conv_projection,
    omega_max,
    molecule,
    dpi,
    save_path=None
):    

    frequency_axis = np.asarray(frequency_axis)
    be_dos = np.asarray(be_dos)

    n = min(len(frequency_axis), len(be_dos))
    frequency_axis = frequency_axis[:n]
    be_dos = be_dos[:n]
    ####
    fig, ax1 = plt.subplots(figsize=(12, 9), dpi=dpi)

    ax1.plot(
        frequency_axis,
        second_conv_projection,
        color="#000000",
        label="Second Convolution (projected)"
    )

    # Vertical guide lines
    ax1.axvline(
        omega_max,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=r"$\Omega_{\max}$"
    )

    ax1.axvline(
        3 * omega_max,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=r"$3\Omega_{\max}$"
    )

    ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    ax1.set_ylabel("Second Convolution / g(ω)", fontsize=24)
    ax1.set_xlim(left=0)
    ax1.tick_params(axis="both", labelsize=16)
    be_interp = interp1d(frequency_axis, be_dos, bounds_error=False, fill_value=0.0)
    be_extended = be_interp(first_conv_freq_axis)

    ax2 = ax1.twinx()
    ax2.plot(first_conv_freq_axis, be_extended, color="red", label="BE Scaled DOS", alpha=0.5)
    ax2.set_ylabel("BE Scaled DOS / g(ω)", fontsize=24, color="red")
    ax2.tick_params(axis="y", labelcolor="red", labelsize=16)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # ax1.set_title(molecule, fontsize=24)
    ax1.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_final(all_data, dpi, save_path=None):
    h50s = []
    areas = []
    labels = []
    mol_types = []

    for mol in all_data:
        name = mol.get("molecule", "UNKNOWN_NAME")

        exp_ratio = mol.get("exp_ratio")
        raw_integral = mol.get("raw_integral")
        h50 = mol.get("H50")

        if exp_ratio is None or raw_integral is None or h50 is None:
            print(f"[SKIP] {name} | exp_ratio={exp_ratio}, raw_integral={raw_integral}, H50={h50}")
            continue

        mol_type = mol.get("cluster_id")
        if mol_type is None:
            print(f"[MISSING TYPE] {name}")
            mol_type = "unknown"

        area = raw_integral * exp_ratio  ###### EXP_RATIO TOGGLE
        # area = area / atom_count

        h50s.append(h50)
        areas.append(area)
        mol_types.append(mol_type)
        labels.append(name)

    # Convert once
    h50s = np.array(h50s)
    areas = np.array(areas)
    mol_types = np.array(mol_types)

    # Fit
    inv_h50 = 1.0 / h50s
    coeffs = np.polyfit(inv_h50, areas, deg=1)
    a, b = coeffs

    predicted = a * inv_h50 + b

    r_squared = r2_score(areas, predicted)

    print("\nLine of best fit:")
    print(f"y = ({a:.6f}) * (1/H50) + ({b:.6f})")
    print(f"R² = {r_squared:.4f}")

    # Colour mapping
    # High-contrast categorical palette
    unique_types = np.unique(mol_types)

    palette = [
        "#E41A1C",  # red
        "#377EB8",  # blue
        "#4DAF4A",  # green
        "#984EA3",  # purple
        "#A65628",  # brown
        "#F781BF",  # pink
        "#17BECF",  # cyan
        "#999999",  # grey
        "#66C2A5",  # teal
        "#8DA0CB",  # lavender blue
        "#E78AC3",  # magenta
        "#A6D854",  # lime
        "#FFD92F",  # gold
    ]

    type_to_color = {
        t: palette[i % len(palette)]
        for i, t in enumerate(unique_types)
    }

    colors = [type_to_color[t] for t in mol_types]

    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.scatter(h50s, areas, c=colors)

    # Fit line
    sorted_idx = np.argsort(h50s)
    plt.plot(h50s[sorted_idx], predicted[sorted_idx],
             '--', color="gray", linewidth=2,)
            #  label=f"Fit (R² = {r_squared:.3f})")

    # ---- LABEL ALL POINTS ----
    # for i, (x, y, label) in enumerate(zip(h50s, areas, labels)):
    #     plt.annotate(
    #         label,
    #         (x, y),
    #         textcoords="offset points",
    #         xytext=(5, 5),
    #         fontsize=7,
    #         alpha=0.75
    #     )

    # Optional legend for molecule types
    for t in unique_types:
        plt.scatter([], [], color=type_to_color[t], label=t)
    plt.legend(title="Molecule type", fontsize=10, ncol=2)

    plt.xlabel("Experimental impact sensitivity / J", fontsize=24)
    plt.ylabel("Up-pumped metric / arb.", fontsize=24)
    plt.tick_params(axis="both", labelsize=18)
    # plt.grid(True)
    # plt.legend()
    plt.tight_layout()

    # plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    # plt.close()
    plt.show()


def plot_atom_count_vs_h50(all_data, dpi=100, save_path=None):
    heavy_atoms = []
    h50_values = []

    for mol in all_data:
        h50 = mol.get("H50")
        exp_ratio = mol.get("exp_ratio")
        # raw_integral = mol.get("raw_integral")
        raw_integral = mol.get("div_atom_count")
        area = raw_integral * exp_ratio if exp_ratio is not None and raw_integral is not None else None

        smiles = mol.get("SMILES")

        if h50 is None or smiles is None or area is None:
            continue

        rd_mol = Chem.MolFromSmiles(smiles)
        if rd_mol is None:
            continue

        num_heavy_atoms = rd_mol.GetNumHeavyAtoms()

        heavy_atoms.append(num_heavy_atoms)
        h50_values.append(area)

    plt.figure(figsize=(8, 6), dpi=dpi)
    plt.scatter(heavy_atoms, h50_values)
    plt.xlabel("Number of Heavy Atoms")
    plt.ylabel("Uppumped metric / arb.")
    plt.title("Heavy Atom Count vs H50")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
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

if __name__ == "__main__":
    dpi = 100

    runs = [
        ("Storm_dataset_raw.csv", "storm"),
        ("30_bench_raw.csv", "30_bench")
    ]

    for csv_name, dataset in runs:

        print(f"\nProcessing plots for: {csv_name}")

        base_plot_dir = create_plot_directories(dataset)
        all_data = read_data_from_csv(csv_name)

        summary_table(all_data)

        # ─────────────────────────
        # plotting controls
        # ─────────────────────────
        plot_enabled = {
            "dos": False,
            "be_dos": False,
            "first_convolution": False,
            "second_convolution": False,
            "final_h50_scatter": True,
            "heavy_atom_count": False,
        }

        def safe_filename(name):
            return "".join(c if c.isalnum() or c in ('-', '_', ' ') else '_' for c in str(name))

        # molecule plots
        for mol in all_data:
            
            mol_name = mol.get("molecule") or mol.get("SMILES", "mol_" + str(id(mol))[-6:])
            fname = safe_filename(mol_name)

            if plot_enabled["dos"] and "dos" in mol and "frequency_axis" in mol:
                path = os.path.join(base_plot_dir, "dos", f"dos_{fname}.png")
                plot_dos(mol["frequency_axis"], mol["dos"], dpi, save_path=path)

            if plot_enabled["be_dos"] and "be_dos" in mol and "frequency_axis" in mol:
                path = os.path.join(base_plot_dir, "be_dos", f"be_dos_{fname}.png")
                plot_be_dos(mol["frequency_axis"], mol["be_dos"], mol["omega_max"], dpi, save_path=path)
                
            if plot_enabled["first_convolution"] and "first_convolved_projection" in mol:
                path = os.path.join(base_plot_dir, "first_convolution", f"first_conv_{fname}.png")
                plot_first_convolution(
                    mol["frequency_axis"],
                    mol.get("be_dos_200"),
                    mol["first_conv_freq_axis"],
                    mol["first_convolved_projection"],
                    dpi,
                    save_path=path
                )

            if plot_enabled["second_convolution"] and "second_convolved_projection" in mol:
                path = os.path.join(base_plot_dir, "second_convolution", f"second_conv_{fname}.png")
                plot_second_convolution(
                    mol["frequency_axis"],
                    mol["be_dos"],
                    mol["first_conv_freq_axis"],
                    mol["second_convolved_projection"],
                    mol["omega_max"],
                    mol_name,
                    dpi,
                    save_path=path
                )

        # summary plots
        if plot_enabled["final_h50_scatter"]:
            path = os.path.join(base_plot_dir, "final_h50_scatter", "h50_vs_uppumped_metric.png")
            plot_final(all_data, dpi, save_path=path)

        if plot_enabled["heavy_atom_count"]:
            path = os.path.join(base_plot_dir, "atom_count_vs_IS_pred", "heavy_atoms_vs_uppumped.png")
            plot_atom_count_vs_h50(all_data, dpi=dpi, save_path=path)
