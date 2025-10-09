import numpy
import csv
import os

atomic_energies = {}

bond_energies = {}


def process_xyz_files(parent_dir=None, output_csv="out.csv"):
    """
    Walk through subdirectories of parent_dir, read xyz files,
    compute oxygen balance, and write results to CSV with columns:
    Filename, SMILES, Oxygen balance /%, Enthalpy of formation.
    If the CSV exists, append new results; otherwise, create it.
    """
    if parent_dir is None:
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "OPTIMISED_STRUCTURES"))
    parent_dir = os.path.abspath(parent_dir)
    print(f"Parent directory: {parent_dir}")

    if not os.path.isdir(parent_dir):
        print(f"ERROR: parent_dir does not exist: {parent_dir}")
        return False

    # Absolute path for output CSV (one directory above parent_dir)
    abs_output = os.path.abspath(os.path.join(parent_dir, "..", output_csv))
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    fieldnames = ["Filename", "SMILES", "Oxygen balance /%", "Enthalpy of formation"]

def calculate_enthalpy():
    

if __name__ == "__main__":
    process_xyz_files()