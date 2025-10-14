import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

i50_values = {
    "TET-1": (2, 1, 76.5324),
    "NP1": (2.5, 2, 66.1026),
    "HNB": (2.75, 3, 83.0775),
    "PETN": (3, 4, 56.2282),
    "CL20": (3, 5, 81.0393),
    "BP-1": (4.5, 6, 78.9906),
    "TET-2": (5, 7, 72.5534),
    "CH-PETN": (6, 8, 45.1597),
    "HMX": (8, 9, 108.3263),
    "TET-3": (8, 10, 86.7160),
    "RDX": (13, 11, 77.8322),
    "TNT-1": (14, 12, 86.9846),
    "PCA": (16, 13, 85.0698),
    "TET-4": (15, 14, 99.7584),
    "BP-2": (20, 15, 92.2816),
    "ADNP-5": (23, 16, 103.7134),
    "TNT": (24.5, 17, 49.0603),
    "DNP-35": (25, 18, 69.3667),
    "DNP-13": (25, 19, 52.6453),
    "TNT-2": (26.8, 20, 89.2217),
    "BP-3": (30, 21, 35.4174),
    "FOX7": (31, 22, 65.2550),
    "DNP-34": (40, 23, 56.9645),
    "TET-5": (40, 24, 96.6349),
    "ADNP-4": (41, 25, 105.6198),
    "nitrotriazolone": (73, 26, 43.2361),
    "nitroguanidine": (80, 27, 108.1432),
    "TET-6": (100, 28, 87.0307),
    "TATB": (120, 29, 84.7751),
}

# Create directory structure for saving plots
def create_plot_directories(script_name=None):
    """
    Creates a directory structure for storing plots under the 'FIGURES' directory.
    Args:
        script_name (str, optional): Name of the script; defaults to the current script name.
    Returns:
        str: Path to the base directory for plots.
    """
    if script_name is None:
        script_name = os.path.splitext(os.path.basename(__file__))[0]

    figures_dir = os.path.join(os.getcwd(), "FIGURES")
    base_dir = os.path.join(figures_dir, f"{script_name}")

    subdirs = [
        "DOS", "Box_DOS", "BE_Scaled_DOS", "First_Convolution",
        "Second_Convolution", "NO2_Angle", "Max_Min_Normalised"
    ]

    try:
        os.makedirs(figures_dir, exist_ok=True)
        print(f"Created figures directory: {figures_dir}")

        os.makedirs(base_dir, exist_ok=True)
        print(f"Created base directory: {base_dir}")

        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            print(f"Created subdirectory: {subdir_path}")

    except Exception as e:
        print(f"Error creating directories: {e}")

    return base_dir



import os

def read_txt_xyz_files(xyz_dir):
    """
    Read eigenvector .txt files and associated .xyz coordinate files.
    
    Args:
        xyz_dir (str): Directory containing subdirectories for each molecule 
                       with both eigenvector (.txt) and coordinate (.xyz) files.

    Returns:
        data (list of dict): Each dict contains:
            {
                "molecule": str,
                "frequencies": list of float,
                "atom_count": int,
                "coordinates": list of tuples (symbol, x, y, z)
            }
    """
    data = []

    for root, dirs, files in os.walk(xyz_dir):
        txt_files = [f for f in files if f.endswith(".txt")]
        xyz_files = [f for f in files if f.endswith(".xyz")]

        for txt_file in txt_files:
            txt_path = os.path.join(root, txt_file)
            freqs = []
            atom_count = None
            coordinates = []

            with open(txt_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 2 and ":" in lines[2]:
                    atom_count = int(lines[2].strip().split(":")[-1])
                for line in lines:
                    if "=" in line and "+" in line:
                        try:
                            freq_str = line.split("=")[1].split("+")[0].strip()
                            freqs.append(float(freq_str))
                        except ValueError:
                            continue

            # Try to find .xyz with the same stem as .txt
            txt_stem = os.path.splitext(txt_file)[0]
            xyz_candidate = os.path.join(root, f"{txt_stem.replace('_eigenvectors', '')}.xyz")

            if os.path.exists(xyz_candidate):
                xyz_path = xyz_candidate
            elif xyz_files:
                xyz_path = os.path.join(root, xyz_files[0])
            else:
                xyz_path = None

            if xyz_path and os.path.exists(xyz_path):
                try:
                    with open(xyz_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()[2:]
                except UnicodeDecodeError:
                    with open(xyz_path, "r", encoding="latin-1", errors="ignore") as f:
                        lines = f.readlines()[2:]

                for line in lines:
                    parts = line.split()
                    if len(parts) == 4:
                        symbol = parts[0]
                        x, y, z = map(float, parts[1:])
                        coordinates.append((symbol, x, y, z))
            else:
                print(f"⚠️ No .xyz file found for {txt_file}")

            data.append({
                "molecule": os.path.basename(root),
                "frequencies": freqs,
                "atom_count": atom_count,
                "coordinates": coordinates
            })


    return data


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xyz_dir = os.path.join(script_dir, "UPPUMPING/OPTIMISED_STRUCTURES/LARGE_MODEL") 

    all_data = read_txt_xyz_files(xyz_dir)
    for entry in all_data:
        print(f"\nMolecule:\n {entry['molecule']}")
        print(f"Atom count: {entry['atom_count']}")
        print(f"Frequencies:\n {entry['frequencies']}")
        print(f"Coordinates:\n {entry['coordinates']}")
