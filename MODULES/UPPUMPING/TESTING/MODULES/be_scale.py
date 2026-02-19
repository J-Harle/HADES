import os
import csv
import json
import math
import sys
import numpy as np
from tqdm import tqdm
import scipy.constants

csv.field_size_limit(sys.maxsize)

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
                    "SMILES": row["SMILES"],
                    "cluster_id": int(row["cluster_id"]),
                    "H50": float(row["H50"]),
                    "exp_ratio": float(row["exp_ratio"]),
                    "atom_count": int(row["atom_count"]),
                    "frequencies": json.loads(row["frequencies"]),
                    # "coordinates": json.loads(row["coordinates"]),
                    # "eigenvectors": json.loads(row["eigenvectors"]),
                })
            except Exception as e:
                print(f"[WARN] Skipping row due to parsing error: {e}")

    # print(f"[INFO] Loaded {len(data)} molecules from CSV\n")
    return data


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
    sigma = gwidth / 2.354  # FWHM → σ

    if gwidth < bwidth:
        dos[:] = histogram / bwidth
    else:
        half_width = int(3.0 * gwidth / bwidth)

        for i in range(-half_width, half_width + 1):
            weight = math.exp(
                -((i * bwidth) ** 2) / (2.0 * sigma ** 2)
            ) / (math.sqrt(2.0 * math.pi) * sigma)

            for h in range(
                max(i, 0),
                min(len_bins + i - 1, len_bins - 1) + 1
            ):
                dos[h] += histogram[h - i] * weight

    return dos


def normalise_dos(dos, atom_count, bwidth, name):
    area = np.trapz(dos, dx=bwidth)
    normalisation_factor = 3 * atom_count

    if area > 0:
        dos *= normalisation_factor / area
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
        print(f"[ERROR] Invalid bin count for {name}")
        return None, None

    frequency_axis = np.linspace(base, max_freq, len_bins)

    hist = histogram(freqs, bwidth, len_bins, base)
    dos = gaussian_broadening(hist, bwidth, gwidth)
    dos = normalise_dos(dos, atom_count, bwidth, name)

    return frequency_axis, dos


def compute_all_dos(data, bwidth, gwidth):
    for mol in tqdm(data, desc="Calculating DOS", unit="molecule"):

        freqs = mol["frequencies"]
        atom_count = mol["atom_count"]
        name = mol["molecule"]

        if not freqs or atom_count is None:
            print(f"[WARN] Skipping {name}")
            continue

        frequency_axis, dos = generate_dos(
            freqs, name, atom_count, bwidth, gwidth
        )

        if dos is None:
            continue

        mol["frequency_axis"] = frequency_axis
        mol["dos"] = dos

    print()
    return data


def bose_einstein_scaling(all_results):
    hbar = scipy.constants.hbar
    k = scipy.constants.k
    t = 300

    for mol in all_results:

        freq_axis = mol["frequency_axis"]
        dos = mol["dos"]

        be_dos = np.copy(dos)

        for i, w in enumerate(freq_axis):
            if w > 5:
                n = (
                    1.0 /
                    (
                        np.exp(
                            hbar *
                            (w * (2 * math.pi) * (1E9 * 29.979245))
                            / (k * t)
                        ) - 1.0
                    )
                )
                be_dos[i] = dos[i] * n

        mol["be_dos"] = be_dos

    return all_results


def write_updated_csv(data, csv_name="raw.csv"):
    csv_path = os.path.join(script_dir, csv_name)

    if not data:
        raise ValueError("[ERROR] No data to write.")

    # Ensure new fields exist in header
    fieldnames = list(data[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for mol in data:
            row = mol.copy()

            # Convert numpy arrays to JSON strings
            if "frequency_axis" in row:
                row["frequency_axis"] = json.dumps(
                    row["frequency_axis"].tolist()
                )

            if "dos" in row:
                row["dos"] = json.dumps(
                    row["dos"].tolist()
                )

            if "be_dos" in row:
                row["be_dos"] = json.dumps(
                    row["be_dos"].tolist()
                )

            writer.writerow(row)

    # print("[INFO] CSV updated with frequency_axis, dos, be_dos\n")


if __name__ == "__main__":

    gwidth = 2.5
    bwidth = 0.5

    data = read_data_from_csv("raw.csv")

    all_results = compute_all_dos(data, bwidth, gwidth)
    all_results = bose_einstein_scaling(all_results)

    write_updated_csv(all_results, "raw.csv")

    # print(all_results[0].keys())
    # print(f"[DEBUG] DOS + BE DOS computed for {len(all_results)} molecules")
