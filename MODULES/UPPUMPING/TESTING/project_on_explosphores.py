import numpy as np
import json
import csv
import os

def read_data_from_csv(csv_name):
    csv_path = os.path.join(script_dir, csv_name)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV not found: {csv_path}")
    
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                    "molecule": row["molecule"],
                    "SMILES": row["SMILES"],
                    "cluster_id": (row["cluster_id"]),
                    "H50": float(row["H50"]),
                    "exp_ratio": float(row["exp_ratio"]),
                    "atom_count": int(row["atom_count"]),
                    "frequencies": json.loads(row["frequencies"]),
                    "coordinates": json.loads(row["coordinates"]),
                    "eigenvectors": json.loads(row["eigenvectors"]),
                })

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    runs = ["Storm_dataset_raw.csv", "30_bench.csv"]

    for csv_name in runs:
        print(f"\n Processing {csv_name}")

        data = read_data_from_csv(csv_name)