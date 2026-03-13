import os
import csv
import json
import sys
from tqdm import tqdm

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
            try:
                data.append({
                    "molecule": row["molecule"],
                    "frequencies": json.loads(row["frequencies"])
                })
            except Exception as e:
                print(f"[WARN] Skipping row due to parsing error: {e}")

    return data


def write_updated_csv(data, original_csv="raw.csv", output_csv="raw.csv"):
    csv_path_in = os.path.join(script_dir, original_csv)
    csv_path_out = os.path.join(script_dir, output_csv)

    with open(csv_path_in, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["omega_max", "integral_freq_count"]

        rows = list(reader)

    omega_map = {mol["molecule"]: mol for mol in data}

    with open(csv_path_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            mol_name = row["molecule"]
            if mol_name in omega_map:
                row["omega_max"] = omega_map[mol_name]["omega_max"]
                row["integral_freq_count"] = omega_map[mol_name]["integral_freq_count"]
            else:
                row["omega_max"] = None
                row["integral_freq_count"] = None
            writer.writerow(row)


def non_dihedral_omax(all_data):
    for mol in tqdm(all_data, desc="Calculating Omega_max", unit="molecule"):
        freqs = mol.get("frequencies", [])

        low_freqs = [f for f in freqs if f <= 200]

        if not low_freqs:
            mol["omega_max"] = None
            mol["integral_freq_count"] = None
            continue

        omega_max = max(low_freqs)
        freq_count = sum(1 for f in freqs if omega_max <= f <= 3 * omega_max)

        mol["omega_max"] = omega_max
        mol["integral_freq_count"] = freq_count

    return all_data





if __name__ == "__main__":

    mode = "torsional"  # or "matrix"

    runs = [
        "Storm_dataset_raw.csv",
        "30_bench_raw.csv"
    ]

    for csv_name in runs:

        print(f"\nProcessing: {csv_name}")

        data = read_data_from_csv(csv_name)

        if mode == "torsional":
            data = non_dihedral_omax(data)

            write_updated_csv(
                data,
                original_csv=csv_name,
                output_csv=csv_name
            )

        else:
            pass  # Placeholder for the d matrix

        print()
