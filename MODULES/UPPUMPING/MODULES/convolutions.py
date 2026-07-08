import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import csv
import sys

csv.field_size_limit(sys.maxsize)

script_dir = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# ARGPARSE
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate first and second DOS convolutions for a CSV."
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
        help="Output CSV filename or path. If omitted, the input CSV is overwritten safely."
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Number of rows between output flushes. Default: 1000"
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


def load_json_list(value, default=None):
    """
    Safely load a JSON list from a CSV cell.
    """
    if value is None:
        return default

    value = str(value).strip()

    if value == "":
        return default

    return json.loads(value)


# =========================================================
# CONVOLUTION FUNCTIONS
# =========================================================

def compute_first_convolution(mol):

    omega_max = mol["omega_max"]
    be_dos = np.array(mol["be_dos"])
    freq_axis = np.array(mol["frequency_axis"])

    be_dos_200 = np.copy(be_dos)
    be_dos_200[freq_axis > omega_max] = 0

    first_conv_freq_axis = np.linspace(
        0,
        2 * np.max(freq_axis),
        len(freq_axis) * 2 - 1
    )

    convolved_raw = np.convolve(be_dos_200, be_dos_200, mode="full")

    first_interpolation = interp1d(
        first_conv_freq_axis,
        convolved_raw,
        bounds_error=False,
        fill_value=0
    )

    first_convolved_projection = first_interpolation(freq_axis)

    first_convolved_projection[be_dos <= 1e-6] = 0

    mol["first_convolved_raw"] = convolved_raw.tolist()
    mol["first_conv_freq_axis"] = first_conv_freq_axis.tolist()
    mol["first_convolved_projection"] = first_convolved_projection.tolist()
    mol["be_dos_200"] = be_dos_200.tolist()

    return mol


def compute_second_convolution(mol):

    omega_max = mol["omega_max"]

    freq_axis = np.array(mol["frequency_axis"])
    be_dos = np.array(mol["be_dos"])
    first_proj = np.array(mol["first_convolved_projection"])

    # Use exp_be_dos if present and same length.
    # Otherwise fall back to be_dos so the script does not crash.
    exp_be_dos_raw = mol.get("exp_be_dos")

    if exp_be_dos_raw is None:
        exp_be_dos = be_dos
    else:
        exp_be_dos = np.array(exp_be_dos_raw)

        if len(exp_be_dos) != len(be_dos):
            exp_be_dos = be_dos

    be_dos_200 = np.copy(be_dos)
    be_dos_200[freq_axis > omega_max] = 0

    conv_freq_axis = np.linspace(
        0,
        2 * np.max(freq_axis),
        len(freq_axis) * 2 - 1
    )

    first_interp = interp1d(
        freq_axis,
        first_proj,
        bounds_error=False,
        fill_value=0
    )

    be_interp = interp1d(
        freq_axis,
        be_dos_200,
        bounds_error=False,
        fill_value=0
    )

    first_extended = first_interp(conv_freq_axis)
    be_extended = be_interp(conv_freq_axis)

    convolved_raw = np.convolve(be_extended, first_extended, mode="full")

    second_conv_freq_axis = np.linspace(
        0,
        2 * conv_freq_axis.max(),
        len(convolved_raw)
    )

    second_interp = interp1d(
        second_conv_freq_axis,
        convolved_raw,
        bounds_error=False,
        fill_value=0
    )

    second_convolved_projection = second_interp(freq_axis)

    second_convolved_projection[exp_be_dos <= 1e-6] = 0

    mol["second_convolved_raw"] = convolved_raw.tolist()
    mol["second_conv_freq_axis"] = second_conv_freq_axis.tolist()
    mol["second_convolved_projection"] = second_convolved_projection.tolist()

    return mol


# =========================================================
# STREAMING CSV PROCESSING
# =========================================================

def process_csv_streaming(
    input_csv,
    output_csv=None,
    csv_dir=".",
    save_interval=1000
):

    csv_dir = os.path.abspath(csv_dir)

    input_path = resolve_path(input_csv, csv_dir)

    if output_csv is None:
        output_path = input_path
    else:
        output_path = resolve_path(output_csv, csv_dir)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"[ERROR] CSV not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temp_output = output_path + ".tmp"

    # =====================================================
    # OPEN INPUT + OUTPUT ONCE
    # =====================================================

    with open(input_path, "r", encoding="utf-8") as fin, \
            open(temp_output, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)

        if reader.fieldnames is None:
            raise ValueError(f"[ERROR] No headers found in: {input_path}")

        fieldnames = list(reader.fieldnames)

        new_cols = [
            "first_convolved_raw",
            "first_conv_freq_axis",
            "first_convolved_projection",
            "be_dos_200",
            "second_convolved_raw",
            "second_conv_freq_axis",
            "second_convolved_projection",
        ]

        for col in new_cols:
            if col not in fieldnames:
                fieldnames.append(col)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)

        writer.writeheader()

        # =================================================
        # STREAM ROWS
        # =================================================

        for idx, row in enumerate(
            tqdm(reader, desc="Calculating Convolutions", unit="molecules "),
            start=1
        ):

            try:
                mol = {
                    "molecule": row["molecule"],
                    "frequency_axis": load_json_list(row.get("frequency_axis"), default=[]),
                    "be_dos": load_json_list(row.get("be_dos"), default=[]),
                    "exp_be_dos": load_json_list(row.get("exp_be_dos"), default=None),
                    "omega_max": float(row["omega_max"])
                }

                # =========================================
                # FIRST CONVOLUTION
                # =========================================

                mol = compute_first_convolution(mol)

                # =========================================
                # SECOND CONVOLUTION
                # =========================================

                mol = compute_second_convolution(mol)

                # =========================================
                # WRITE RESULTS
                # =========================================

                for col in new_cols:
                    row[col] = json.dumps(mol[col])

            except Exception as e:

                print(f"\n[WARN] Failed row {row.get('molecule', 'UNKNOWN')}: {e}")

                for col in new_cols:
                    row[col] = None

            writer.writerow(row)

            # =============================================
            # PERIODIC FLUSH
            # =============================================

            if idx % save_interval == 0:
                fout.flush()
                print(f"\n[INFO] Processed {idx} molecules")

    # =====================================================
    # SAFE REPLACEMENT
    # =====================================================

    os.replace(temp_output, output_path)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    args = parse_args()

    print(f"\nProcessing: {args.input}")
    print(f"Input directory: {os.path.abspath(args.csv_dir)}")

    if args.output is None:
        print("[INFO] Output not provided. Input CSV will be overwritten safely.")
    else:
        print(f"Output CSV: {args.output}")

    process_csv_streaming(
        input_csv=args.input,
        output_csv=args.output,
        csv_dir=args.csv_dir,
        save_interval=args.save_interval
    )

    print(f"[INFO] Finished: {args.output if args.output else args.input}\n")