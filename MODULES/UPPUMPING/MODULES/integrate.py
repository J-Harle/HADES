import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import csv
import sys

csv.field_size_limit(sys.maxsize)

script_dir = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# ARGPARSE
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate projected convolution integrals for a CSV."
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


def set_integral_cols_none(row):
    row["raw_integral"] = None
    row["integral_over_freq_count"] = None
    row["exp_ratio_integral"] = None
    row["div_atom_count"] = None

    return row


# =========================================================
# INTEGRAL CALCULATION
# =========================================================

def compute_integral(row):

    name = row.get("molecule", "UNKNOWN")

    try:
        atom_count = int(row["atom_count"]) if row.get("atom_count") else None

        integral_freq_count = (
            float(row["integral_freq_count"])
            if row.get("integral_freq_count")
            else None
        )

        exp_ratio = (
            float(row["exp_ratio"])
            if row.get("exp_ratio")
            else None
        )

        omega_max = (
            float(row["omega_max"])
            if row.get("omega_max")
            else None
        )

        freq_axis = np.array(json.loads(row.get("frequency_axis", "[]")))
        second_proj = np.array(json.loads(row.get("second_convolved_projection", "[]")))

        # =============================================
        # VALIDATION
        # =============================================

        if omega_max is None:
            print(f"\n[WARN] omega_max missing for {name}")
            return set_integral_cols_none(row)

        if len(freq_axis) == 0:
            print(f"\n[WARN] frequency_axis missing for {name}")
            return set_integral_cols_none(row)

        if len(second_proj) == 0:
            print(f"\n[WARN] second_convolved_projection missing for {name}")
            return set_integral_cols_none(row)

        if len(freq_axis) != len(second_proj):
            print(
                f"\n[WARN] Length mismatch for {name}: "
                f"frequency_axis={len(freq_axis)}, "
                f"second_convolved_projection={len(second_proj)}"
            )
            return set_integral_cols_none(row)

        mask = (
            (freq_axis >= omega_max)
            & (freq_axis <= 3.0 * omega_max)
        )

        if not np.any(mask):
            print(f"[WARN] No data in 1-3 Omax window for {name}")
            return set_integral_cols_none(row)

        # =============================================
        # INTEGRALS
        # =============================================

        raw_integral = float(np.trapz(second_proj[mask], freq_axis[mask]))

        integral_over_freq_count = (
            raw_integral / integral_freq_count
            if integral_freq_count
            else None
        )

        exp_ratio_integral = (
            integral_over_freq_count * exp_ratio
            if integral_over_freq_count is not None and exp_ratio is not None
            else None
        )

        div_atom_count = (
            raw_integral / atom_count
            if atom_count
            else None
        )

        row["raw_integral"] = raw_integral
        row["integral_over_freq_count"] = integral_over_freq_count
        row["exp_ratio_integral"] = exp_ratio_integral
        row["div_atom_count"] = div_atom_count

    except Exception as e:

        print(
            f"\n[WARN] Failed row "
            f"{name}: {e}"
        )

        row = set_integral_cols_none(row)

    return row


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
            "raw_integral",
            "integral_over_freq_count",
            "exp_ratio_integral",
            "div_atom_count",
        ]

        for col in new_cols:
            if col not in fieldnames:
                fieldnames.append(col)

        writer = csv.DictWriter(
            fout,
            fieldnames=fieldnames
        )

        writer.writeheader()

        # =================================================
        # STREAM ROWS
        # =================================================

        for idx, row in enumerate(
            tqdm(reader, desc="Calculating Integrals", unit="molecules "),
            start=1
        ):

            row = compute_integral(row)
            writer.writerow(row)

            # =============================================
            # PERIODIC FLUSH
            # =============================================

            if idx % save_interval == 0:
                fout.flush()
                print(f"\n[INFO] Processed {idx} molecules")

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
        save_interval=args.save_interval
    )

    print(f"[INFO] Finished: {args.output if args.output else args.input}\n")