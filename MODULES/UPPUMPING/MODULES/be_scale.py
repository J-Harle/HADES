import os
import csv
import json
import math
import sys
import ast
import argparse
import numpy as np
from tqdm import tqdm
import scipy.constants

csv.field_size_limit(sys.maxsize)

script_dir = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# ARGPARSE
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DOS and Bose-Einstein-scaled DOS columns for a CSV."
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
        "--bwidth",
        type=float,
        default=0.5,
        help="DOS bin width. Default: 0.5"
    )

    parser.add_argument(
        "--gwidth",
        type=float,
        default=2.5,
        help="Gaussian broadening width. Default: 2.5"
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

# =========================================================
# DOS FUNCTIONS
# =========================================================

def histogram(freqs, bwidth, len_bins, base):
    
    hist = np.zeros(len_bins)

    for freq in freqs:
        bin_idx = int((freq - base) / bwidth)
        if 0 <= bin_idx < len_bins:
            hist[bin_idx] += 1.0

    return hist


def gaussian_broadening(histogram, bwidth, gwidth):

    len_bins = len(histogram)
    dos = np.zeros(len_bins)
    sigma = gwidth / 2.354

    if gwidth < bwidth:
        dos[:] = histogram / bwidth

    else:
        half_width = int(3.0 * gwidth / bwidth)

        for i in range(-half_width, half_width + 1):
            weight = math.exp(-((i * bwidth) ** 2) / (2.0 * sigma ** 2)) / (math.sqrt(2.0 * math.pi)* sigma)

            for h in range(max(i, 0), min(len_bins + i - 1, len_bins - 1) + 1):
                dos[h] += (histogram[h - i] * weight)

    return dos


def normalise_dos(dos, atom_count, bwidth, name):
    area = np.trapz(dos, dx=bwidth)

    normalisation_factor = 1

    if area > 0:
        dos *= (normalisation_factor / area)

    else:
        print(f"[ERROR] Zero DOS area for {name}")

    return dos


def generate_dos(freqs, name, atom_count, bwidth, gwidth):

    if not freqs or atom_count is None:
        print(f"[ERROR] Missing data for {name}")

        return None, None

    base = 0.0
    max_freq = np.max(freqs)
    len_bins = int((max_freq - base) / bwidth) + 1

    if len_bins <= 0:
        print(
            f"[ERROR] Invalid bin count for {name}")
        return None, None

    frequency_axis = np.linspace(base, max_freq, len_bins)
    hist = histogram(freqs, bwidth, len_bins, base)
    dos = gaussian_broadening(hist, bwidth, gwidth)
    dos = normalise_dos(dos, atom_count, bwidth, name)

    return frequency_axis, dos


# =========================================================
# BE SCALING
# =========================================================

def bose_einstein_scaling(mol):

    hbar = scipy.constants.hbar
    k = scipy.constants.k
    t = 300

    # =====================================================
    # STANDARD DOS
    # =====================================================

    if "dos" in mol:
        freq_axis = np.array(mol["frequency_axis"])
        dos = np.array(mol["dos"])
        be_dos = np.copy(dos)

        for i, w in enumerate(freq_axis):
            if w > 5:
                n = (1.0 / (np.exp(hbar * (w * (2 * math.pi) * (1E9 * 29.979245)) / (k * t)) - 1.0))

                be_dos[i] = (dos[i] * n)

        mol["be_dos"] = be_dos

    return mol


# =========================================================
# STREAMING CSV PROCESSING
# =========================================================

def process_csv_streaming(
    input_csv,
    output_csv=None,
    csv_dir=".",
    bwidth=0.5,
    gwidth=2.5,
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

    temp_output = output_path + ".tmp"

    # =====================================================
    # OPEN INPUT + OUTPUT ONCE
    # =====================================================

    with open(input_path, "r", encoding="utf-8") as fin, \
    open(temp_output, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames)

        new_cols = ["frequency_axis", "dos", "be_dos",]

        for col in new_cols:
            if col not in fieldnames:
                fieldnames.append(col)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)

        writer.writeheader()

        # =================================================
        # STREAM ROWS
        # =================================================

        for idx, row in enumerate(tqdm(reader, desc="Calculating DOS", unit="molecules "), start=1):

            try:
                frequencies = json.loads(row["frequencies"])

                h50_raw = row.get("H50", "").strip()

                if h50_raw == "":
                    h50_value = None

                else:
                    h50_value = float(h50_raw)

                mol = {
                    "molecule": row["molecule"],
                    "omega_max": row.get("omega_max"),
                    "SMILES": row["SMILES"],
                    "cluster_id": row["cluster_id"],
                    "H50": h50_value,
                    "exp_ratio": float(row["exp_ratio"]),
                    "atom_count": int(row["atom_count"]),
                    "frequencies": frequencies,
                    "coordinates": ast.literal_eval(row["coordinates"]),
                }

                # =========================================
                # GENERATE DOS
                # =========================================

                frequency_axis, dos = (
                    generate_dos(freqs=mol["frequencies"],
                        name=mol["molecule"],
                        atom_count=mol["atom_count"],
                        bwidth=bwidth,
                        gwidth=gwidth))

                if dos is not None:
                    mol["frequency_axis"] = (frequency_axis)
                    mol["dos"] = dos

                    # =====================================
                    # BE SCALING
                    # =====================================

                    mol = bose_einstein_scaling(mol)

                    # =====================================
                    # WRITE RESULTS
                    # =====================================

                    row["frequency_axis"] = (json.dumps(mol["frequency_axis"].tolist()))
                    row["dos"] = json.dumps(mol["dos"].tolist())
                    row["be_dos"] = json.dumps(mol["be_dos"].tolist())

                else:

                    row["frequency_axis"] = None
                    row["dos"] = None
                    row["be_dos"] = None

            except Exception as e:

                print(f"\n[WARN] Failed row {row.get('molecule', 'UNKNOWN')}")
                print(f"Error: {e}")

                row["frequency_axis"] = None
                row["dos"] = None
                row["be_dos"] = None

            writer.writerow(row)

            # =============================================
            # PERIODIC FLUSH
            # =============================================

            if idx % save_interval == 0:
                fout.flush()


    # =====================================================
    # SAFE FILE REPLACEMENT
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
        bwidth=args.bwidth,
        gwidth=args.gwidth,
        save_interval=args.save_interval
    )

    print(f"[INFO] Finished: {args.output if args.output else args.input}\n")