import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from sklearn.metrics import r2_score
import sys

csv.field_size_limit(sys.maxsize)

script_dir = os.path.dirname(os.path.abspath(__file__))

DROP_AFTER_PREDICTION_COLUMNS = {
    "exp_ratio",
    "atom_count",
    "frequencies",
    "all_freqs",
    "coordinates",
    "omega_max",
    "integral_freq_count",
    "frequency_axis",
    "dos",
    "be_dos",
    "first_convolved_raw",
    "first_conv_freq_axis",
    "first_convolved_projection",
    "be_dos_200",
    "second_convolved_raw",
    "second_conv_freq_axis",
    "second_convolved_projection",
    "raw_integral",
    "integral_over_freq_count",
    "exp_ratio_integral",
    "div_atom_count",
}

# =========================================================
# ARGPARSE
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict H50 values from calculated up-pumped metrics."
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input CSV filename or path."
    )

    parser.add_argument(
        "--dir", "-dir",
        dest="csv_dir",
        type=str,
        default=".",
        help="Directory containing the input CSV and where the output CSV should be written. Default: current directory."
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output prediction CSV filename or path. Default: <input>_predictions.csv"
    )

    parser.add_argument(
        "--a",
        type=float,
        default=0.065785,
        help="Gradient of manual H50 fit. Default: 0.065785"
    )

    parser.add_argument(
        "--b",
        type=float,
        default=0.011077,
        help="Intercept of manual H50 fit. Default: 0.011077"
    )

    return parser.parse_args()


def resolve_path(path, base_dir):
    """
    Resolve path robustly.

    Absolute paths are used directly.
    Relative paths are interpreted relative to base_dir.
    """
    if os.path.isabs(path):
        return path

    return os.path.abspath(os.path.join(base_dir, path))


def default_prediction_output(input_csv):
    """
    Create a safe default output name.

    Example:
        hades_out_raw.csv -> hades_out_raw_predictions.csv
    """
    base = os.path.basename(input_csv)
    stem, ext = os.path.splitext(base)

    if ext == "":
        ext = ".csv"

    return f"{stem}_predictions{ext}"


# =========================================================
# READ DATA
# =========================================================

def read_prediction_data(input_csv, csv_dir="."):

    csv_dir = os.path.abspath(csv_dir)
    csv_path = resolve_path(input_csv, csv_dir)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV not found: {csv_path}")

    data = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"[ERROR] No headers found in: {csv_path}")

        fieldnames = reader.fieldnames

        for row in tqdm(
            reader,
            desc=f"Reading {os.path.basename(csv_path)}",
            unit=" molecules"
        ):

            try:
                data.append({

                    # Keep the original CSV row so we can write it back later
                    "original_row": dict(row),

                    "molecule": row.get("molecule"),

                    "exp_ratio": (
                        float(row["exp_ratio"])
                        if row.get("exp_ratio")
                        else None
                    ),

                    "raw_integral": (
                        float(row["raw_integral"])
                        if row.get("raw_integral")
                        else None
                    ),

                    "H50": (
                        float(row["H50"])
                        if row.get("H50")
                        else None
                    ),

                    "cluster_id": row.get("cluster_id")
                })

            except Exception as e:
                print(f"[WARN] Failed row: {e}")

    return data, fieldnames

# =========================================================
# WRITE DATA
# =========================================================

def write_predictions_csv(data, output_csv, fieldnames, csv_dir="."):

    csv_dir = os.path.abspath(csv_dir)
    csv_path = resolve_path(output_csv, csv_dir)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Remove unwanted intermediate calculation columns
    output_fieldnames = [
        field
        for field in fieldnames
        if field not in DROP_AFTER_PREDICTION_COLUMNS
    ]

    # Add predicted_H50 as a new column if it is not already present
    if "predicted_H50" not in output_fieldnames:
        output_fieldnames.append("predicted_H50")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=output_fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        for mol in tqdm(
            data,
            desc=f"Writing {os.path.basename(csv_path)}",
            unit=" molecules"
        ):

            original_row = mol.get("original_row", {}).copy()

            # Keep only columns that are allowed in the output
            row = {
                key: value
                for key, value in original_row.items()
                if key in output_fieldnames
            }

            predicted_h50 = mol.get("predicted_H50")

            if predicted_h50 is None:
                row["predicted_H50"] = ""
            elif isinstance(predicted_h50, float) and np.isnan(predicted_h50):
                row["predicted_H50"] = "nan"
            else:
                row["predicted_H50"] = predicted_h50

            writer.writerow(row)

    return csv_path

# =========================================================
# OPTIONAL FITTING / PLOTTING FUNCTION
# =========================================================

def plot_final(all_data, dpi=100, save_path=None):

    h50s = []
    areas = []
    mol_types = []

    for mol in tqdm(
        all_data,
        desc="Fitting model",
        unit=" molecules"
    ):

        exp_ratio = mol.get("exp_ratio")
        raw_integral = mol.get("raw_integral")
        h50 = mol.get("H50")

        if (
            exp_ratio is None
            or raw_integral is None
            or h50 is None
        ):
            continue

        area = raw_integral * exp_ratio

        h50s.append(h50)
        areas.append(area)
        mol_types.append(mol.get("cluster_id", "unknown"))

    h50s = np.array(h50s)
    areas = np.array(areas)
    mol_types = np.array(mol_types)

    inv_h50 = 1.0 / h50s

    coeffs = np.polyfit(inv_h50, areas, deg=1)

    a, b = coeffs

    predicted = a * inv_h50 + b

    r_squared = r2_score(areas, predicted)

    print("\nLine of best fit:")
    print(f"y = ({a:.6f}) * (1/H50) + ({b:.6f})")
    print(f"R² = {r_squared:.4f}")

    unique_types = np.unique(mol_types)

    cmap = plt.get_cmap("tab20")

    type_to_color = {
        t: cmap(i % cmap.N)
        for i, t in enumerate(unique_types)
    }

    colors = [type_to_color[t] for t in mol_types]

    plt.figure(figsize=(12, 9), dpi=dpi)

    plt.scatter(h50s, areas, c=colors)

    sorted_idx = np.argsort(h50s)

    plt.plot(
        h50s[sorted_idx],
        predicted[sorted_idx],
        "--",
        color="gray",
        linewidth=2,
    )

    for t in unique_types:
        plt.scatter([], [], color=type_to_color[t], label=t)

    plt.legend(title="Molecule type", fontsize=10, ncol=2)

    plt.xlabel(
        "Experimental impact sensitivity / J",
        fontsize=24
    )

    plt.ylabel(
        "Up-pumped metric / arb.",
        fontsize=24
    )

    plt.tick_params(axis="both", labelsize=18)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    else:
        plt.show()

    return a, b, r_squared


# =========================================================
# PREDICTION
# =========================================================

def predict_h50(all_data, a, b):

    print("\nPredicted impact sensitivities")
    print("-" * 70)
    print(f"{'Molecule':<30} {'Predicted H50 / J':>20}")

    for mol in tqdm(
        all_data,
        desc="Predicting H50",
        unit=" molecules"
    ):

        exp_ratio = mol.get("exp_ratio")
        raw_integral = mol.get("raw_integral")

        if exp_ratio is None or raw_integral is None:
            mol["predicted_H50"] = None
            continue

        metric = raw_integral * exp_ratio

        denominator = metric - b

        if abs(denominator) < 1e-12:
            predicted_h50 = np.nan

        else:
            predicted_h50 = a / denominator

        mol["predicted_H50"] = predicted_h50

    return all_data


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    args = parse_args()

    csv_dir = os.path.abspath(args.csv_dir)

    if args.output is None:
        output_csv = default_prediction_output(args.input)
    else:
        output_csv = args.output

    print(f"\nProcessing: {args.input}")
    print(f"Input directory: {csv_dir}")
    print(f"Output CSV: {output_csv}")
    print(f"[INFO] Using manual fit: y = ({args.a}) * (1/H50) + ({args.b})")

    # ==========================================
    # LOAD DATA
    # ==========================================

    unknown_data, fieldnames = read_prediction_data(
        input_csv=args.input,
        csv_dir=csv_dir
    )

    # ==========================================
    # PREDICT H50
    # ==========================================

    unknown_data = predict_h50(
        unknown_data,
        args.a,
        args.b
    )

    # ==========================================
    # WRITE PREDICTIONS
    # ==========================================

    output_path = write_predictions_csv(
        unknown_data,
        output_csv,
        fieldnames,
        csv_dir=csv_dir
    )

    print(f"\n[INFO] CSV writing complete: {output_path}")

    del unknown_data