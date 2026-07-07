import os
import json
import numpy as np
from tqdm import tqdm
import csv
import sys

csv.field_size_limit(sys.maxsize)  # handle large JSON arrays

script_dir = os.path.dirname(os.path.abspath(__file__))


def integrate(all_data):
    print()
    for mol in tqdm(all_data, desc="Calculating Integral", unit="molecule"):
        name = mol.get("molecule", "UNKNOWN")
        atom_count = mol.get("atom_count")
        integral_freq_count = mol.get("integral_freq_count")
        exp_ratio = mol.get("exp_ratio")
        omega_max = mol.get("omega_max")
        freq_axis = np.array(mol.get("frequency_axis", []))
        second_proj = np.array(mol.get("second_convolved_projection", []))

        if omega_max is None:
            print(f"[WARN] omega_max missing for {name}")
            mol["raw_integral"] = None
            mol["integral_over_freq_count"] = None
            mol["exp_ratio_integral"] = None
            continue

        mask = (freq_axis >= omega_max) & (freq_axis <= 3.0 * omega_max)

        if not np.any(mask):
            print(f"[WARN] No data in 1-3 Omax window for {name}")
            mol["raw_integral"] = None
            mol["integral_over_freq_count"] = None
            mol["exp_ratio_integral"] = None
            continue

        raw_integral = float(np.trapz(second_proj[mask], freq_axis[mask]))
        integral_over_freq_count = raw_integral / integral_freq_count if integral_freq_count else None
        exp_ratio_integral = integral_over_freq_count * exp_ratio if integral_over_freq_count is not None else None

        mol["raw_integral"] = raw_integral
        mol["integral_over_freq_count"] = integral_over_freq_count
        mol["exp_ratio_integral"] = exp_ratio_integral
        mol["div_atom_count"] = raw_integral / int(atom_count)
        
    return all_data


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
                    "molecule": row.get("molecule"),
                    "frequency_axis": json.loads(row.get("frequency_axis", "[]")),
                    "second_convolved_projection": json.loads(row.get("second_convolved_projection", "[]")),
                    "integral_freq_count": float(row.get("integral_freq_count")) if row.get("integral_freq_count") else None,
                    "exp_ratio": float(row.get("exp_ratio")) if row.get("exp_ratio") else None,
                    "omega_max": float(row.get("omega_max")) if row.get("omega_max") else None,
                    # Preserve any other columns dynamically
                    **{k: v for k, v in row.items() if k not in ["molecule", "frequency_axis", "second_convolved_projection", "integral_freq_count", "exp_ratio", "omega_max"]}
                })
            except Exception as e:
                print(f"[WARN] Skipping row due to parsing error: {e}")

    # print(f"[INFO] Loaded {len(data)} molecules from CSV\n")
    return data


def write_updated_csv(data, output_csv="raw.csv"):
    csv_path = os.path.join(script_dir, output_csv)

    if not data:
        print("[ERROR] No data to write.")
        return

    # Collect all keys dynamically to preserve all columns
    all_keys = set()
    for mol in data:
        all_keys.update(mol.keys())
    fieldnames = list(all_keys)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for mol in data:
            row = {}
            for key in fieldnames:
                value = mol.get(key)
                if isinstance(value, (list, dict, np.ndarray)):
                    row[key] = json.dumps(value)
                else:
                    row[key] = value
            writer.writerow(row)

    # print(f"[INFO] CSV updated with integrals: {output_csv}")

if __name__ == "__main__":

    runs = [
        "Storm_dataset_raw.csv",
        "30_bench_raw.csv"
    ]

    for csv_name in runs:

        print(f"\nProcessing: {csv_name}")

        # Load molecules
        all_data = read_data_from_csv(csv_name)

        # Compute integrals
        all_data = integrate(all_data)

        # Write results back
        write_updated_csv(all_data, csv_name)

        # if all_data:
        #     print(all_data[0].keys())
