import os
import numpy as np
import pandas as pd
import re
import urllib.request
from io import StringIO
from ase import Atoms, units
from ase.io import read, write
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from rdkit import Chem
from rdkit.Chem import AllChem
from mace.calculators import MACECalculator


def read_csv(filename):
    """
    Reads the dataframe and keeps SMILES + CID columns.
    Args:
        filename (str): path to CSV
    Returns:
        pd.DataFrame: dataframe containing SMILES and CID
    """
    df = pd.read_csv(filename)

    # Normalise column names
    df.columns = [c.upper() for c in df.columns]

    # Ensure both CID and SMILES are present
    if not {"SMILES", "CID"}.issubset(df.columns):
        raise ValueError("CSV must contain 'SMILES' and 'CID' columns")

    df = df[["CID", "SMILES"]]
    # df.info()
    return df



def create_ase_objs(smiles_list):
    """
    Create 3D ASE Atoms objects from a list of SMILES.
    Args:
        smiles_list (list[str])
    Returns:
        list[ase.Atoms]: ASE atoms objects
    """
    atoms_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        # Export XYZ via RDKit
        xyz_block = Chem.MolToXYZBlock(mol)
        atoms = read(StringIO(xyz_block), format="xyz")
        atoms_list.append(atoms)

    return atoms_list


def safe_smiles_name(smiles):
    """
    Make SMILES safe for use as a directory name.
    Replace characters that are invalid on most filesystems.
    """
    return re.sub(r'[^A-Za-z0-9._-]', '_', smiles)


def get_mace_calculator():
    """
    Ensure MACE model is available in CALCULATORS directory and return calculator.
    """
    parent_dir = os.getcwd()
    calc_dir = os.path.join(parent_dir, "CALCULATORS")
    os.makedirs(calc_dir, exist_ok=True)

    model_file = os.path.join(calc_dir, "MACE-OFF23_large.model")

    # Download if missing
    if not os.path.exists(model_file):
        print(f"Downloading MACE model to {model_file} ...")
        url = "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_large.model?raw=true"
        urllib.request.urlretrieve(url, model_file)
        print("Download complete!")

    # Return calculator
    return MACECalculator(
        model_paths=[model_file],
        dispersion=False,
        default_dtype="float64",
        device="cpu"  # set "cuda" if GPU is available
    )


def optimise_and_write(atoms_list, smiles_list, cid_list, base_dir="OPTIMISED_STRUCTURES"):
    parent_dir = os.getcwd()
    optimised_dir = os.path.join(parent_dir, base_dir)
    os.makedirs(optimised_dir, exist_ok=True)

    calc = get_mace_calculator()

    for atoms, smiles, cid in zip(atoms_list, smiles_list, cid_list):
        mol_dir = os.path.join(optimised_dir, str(cid))
        os.makedirs(mol_dir, exist_ok=True)

        atoms.calc = calc

        log_file = os.path.join(mol_dir, f"{cid}.log")
        dyn = BFGS(atoms, logfile=log_file)
        dyn.run(fmax=0.001)

        xyz_path = os.path.join(mol_dir, f"{cid}.xyz")
        with open(xyz_path, "w") as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"Optimised structure for CID {cid} ({smiles})\n")
            for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                f.write(f"{symbol:2} {x:12.6f} {y:12.6f} {z:12.6f}\n")

        print(f"\nOptimised CID {cid} → {xyz_path}")

if __name__ == "__main__":
    df = read_csv("PubChem_Filtered_Mols.csv")
    smiles_list = df["SMILES"].tolist()
    cid_list = df["CID"].tolist()
    atoms_list = create_ase_objs(smiles_list)
    optimise_and_write(atoms_list, smiles_list, cid_list)

