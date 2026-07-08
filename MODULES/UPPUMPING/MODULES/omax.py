import os
import csv
import json
import sys
import argparse
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

script_dir = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate omega_max and integral frequency count"
    )

    parser.add_argument(
        "--input", "-i",
        default="hades_out.csv",
        help="Input CSV name used to derive the raw CSV. Example: hades_out.csv -> hades_out_raw.csv"
    )

    return parser.parse_args()

# =========================================================
# OMEGA MAX CALCULATION
# =========================================================

def calculate_omega_max(freqs):

    if not freqs:
        return None, None

    freqs = [f for f in freqs if f > 100]
    low_freqs = [f for f in freqs if f <= 200]

    if low_freqs:
        omega_max = max(low_freqs)

    else:
        omega_max = min(freqs, key=lambda x: abs(x - 200))

    freq_count = sum(1 for f in freqs if omega_max <= f <= 3 * omega_max)

    return omega_max, freq_count


# =========================================================
# STREAM PROCESS CSV
# =========================================================

def process_csv_streaming(csv_path):

    input_path = os.path.abspath(csv_path)

    temp_path = os.path.join(
        os.path.dirname(input_path),
        f"temp_{os.path.basename(input_path)}"
    )

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"[ERROR] CSV not found: {input_path}")

    # ==========================================
    # COUNT ROWS FOR PROGRESS BAR
    # ==========================================

    with open(input_path, "r", encoding="utf-8") as f:
        total_rows = sum(1 for _ in f) - 1

    # ==========================================
    # STREAM READ + WRITE
    # ==========================================

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(temp_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)

        fieldnames = list(reader.fieldnames)

        for col in ["omega_max", "integral_freq_count"]:

            if col not in fieldnames:
                fieldnames.append(col)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)

        writer.writeheader()

        for row in tqdm(reader, total=total_rows, desc="Calculating Omega_max", unit="molecules "):

            try:
                freqs = json.loads(row["frequencies"])
                omega_max, freq_count = (calculate_omega_max(freqs))
                row["omega_max"] = omega_max
                row["integral_freq_count"] = freq_count

            except Exception as e:
                print(f"\n[WARN] Failed molecule {row.get('molecule', 'UNKNOWN')}: {e}")
                row["omega_max"] = None
                row["integral_freq_count"] = None

            writer.writerow(row)

    # ==========================================
    # REPLACE ORIGINAL FILE
    # ==========================================

    os.replace(temp_path, input_path)
    print(f"[INFO] Updated CSV written: {input_path}")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    args = parse_args()

    input_stem = os.path.splitext(os.path.basename(args.input))[0]

    if input_stem.endswith("_raw"):
        raw_csv_name = f"{input_stem}.csv"
    else:
        raw_csv_name = f"{input_stem}_raw.csv"

    raw_csv_path = os.path.join(script_dir, raw_csv_name)

    print(f"\nProcessing: {raw_csv_path}")

    process_csv_streaming(raw_csv_path)

    print()