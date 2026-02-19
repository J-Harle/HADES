import os
import json
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import csv
import sys

csv.field_size_limit(sys.maxsize)

script_dir = os.path.dirname(os.path.abspath(__file__))


def first_convolution(all_data):
    for mol in tqdm(all_data, desc="Calculating First Convolution", unit="molecule"):
        be_dos = np.array(mol["be_dos"])
        freq_axis = np.array(mol["frequency_axis"])

        be_dos_200 = np.copy(be_dos)
        be_dos_200[freq_axis > 200] = 0

        first_conv_freq_axis = np.linspace(0, 2 * np.max(freq_axis), len(freq_axis) * 2 - 1)
        convolved_raw = np.convolve(be_dos_200, be_dos_200, mode="full")

        first_interpolation = interp1d(first_conv_freq_axis, convolved_raw, bounds_error=False, fill_value=0)
        first_convolved_projection = first_interpolation(freq_axis)
        first_convolved_projection[be_dos <= 1e-6] = 0

        mol["first_convolved_raw"] = convolved_raw.tolist()
        mol["first_conv_freq_axis"] = first_conv_freq_axis.tolist()
        mol["first_convolved_projection"] = first_convolved_projection.tolist()
        mol["be_dos_200"] = be_dos_200.tolist()

    return all_data


def second_convolution(all_data):
    for mol in tqdm(all_data, desc="Calculating Second Convolution", unit="molecule"):
        freq_axis = np.array(mol["frequency_axis"])
        be_dos = np.array(mol["be_dos"])
        first_proj = np.array(mol["first_convolved_projection"])

        be_dos_200 = np.copy(be_dos)
        be_dos_200[freq_axis > 200] = 0

        conv_freq_axis = np.linspace(0, 2 * np.max(freq_axis), len(freq_axis) * 2 - 1)
        first_interp = interp1d(freq_axis, first_proj, bounds_error=False, fill_value=0)
        be_interp = interp1d(freq_axis, be_dos_200, bounds_error=False, fill_value=0)

        first_extended = first_interp(conv_freq_axis)
        be_extended = be_interp(conv_freq_axis)

        convolved_raw = np.convolve(be_extended, first_extended, mode="full")
        second_conv_freq_axis = np.linspace(0, 2 * conv_freq_axis.max(), len(convolved_raw))
        second_interp = interp1d(second_conv_freq_axis, convolved_raw, bounds_error=False, fill_value=0)

        second_convolved_projection = second_interp(freq_axis)
        second_convolved_projection[be_dos <= 1e-6] = 0

        mol["second_convolved_raw"] = convolved_raw.tolist()
        mol["second_conv_freq_axis"] = second_conv_freq_axis.tolist()
        mol["second_convolved_projection"] = second_convolved_projection.tolist()

    return all_data


def read_csv_for_convolutions(csv_name="raw.csv"):
    csv_path = os.path.join(script_dir, csv_name)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV not found: {csv_path}")

    all_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)

    # Extract frequency_axis and be_dos as lists for convolution
    data = []
    for row in all_rows:
        try:
            data.append({
                "molecule": row["molecule"],
                "frequency_axis": json.loads(row["frequency_axis"]),
                "be_dos": json.loads(row["be_dos"]),
            })
        except Exception as e:
            print(f"[WARN] Skipping row {row.get('molecule', 'UNKNOWN')} due to parsing error: {e}")

    return all_rows, data


def write_updated_csv(all_rows, convolution_data, output_csv="raw.csv"):
    # Map results by molecule
    conv_map = {mol["molecule"]: mol for mol in convolution_data}

    # Add new column names if missing
    new_cols = [
        "first_convolved_raw",
        "first_conv_freq_axis",
        "first_convolved_projection",
        "be_dos_200",
        "second_convolved_raw",
        "second_conv_freq_axis",
        "second_convolved_projection",
    ]
    fieldnames = list(all_rows[0].keys()) + new_cols

    csv_path = os.path.join(script_dir, output_csv)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_rows:
            mol_name = row["molecule"]
            if mol_name in conv_map:
                for col in new_cols:
                    row[col] = json.dumps(conv_map[mol_name][col])
            else:
                for col in new_cols:
                    row[col] = None
            writer.writerow(row)

    # print(f"[INFO] CSV updated with convolutions: {output_csv}")


if __name__ == "__main__":
    # Load CSV
    all_rows, conv_data = read_csv_for_convolutions("raw.csv")

    # Compute convolutions
    conv_data = first_convolution(conv_data)
    print()
    conv_data = second_convolution(conv_data)
    # Merge back and write full CSV
    write_updated_csv(all_rows, conv_data, "raw.csv")
