import os
import numpy as np
import pandas as pd
from io import StringIO
from ase import Atoms, units
from ase.io import read, write
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from rdkit import Chem
from rdkit.Chem import AllChem


def read_csv(filename):
    """
    Reads the dataframe and drops all columns except SMILES.
    Args:
        filename (str): path to CSV
    Returns:
        pd.DataFrame: dataframe containing only the SMILES column
    """
    df = pd.read_csv(filename)
    if "SMILES" in df.columns:
        df = df[["SMILES"]]
    else:
        # try case-insensitive match
        df = df.filter(regex="smiles", axis=1)
        df.columns = ["SMILES"]
    df.info()
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

import re

def safe_smiles_name(smiles):
    """
    Make SMILES safe for use as a directory name.
    Replace characters that are invalid on most filesystems.
    """
    return re.sub(r'[^A-Za-z0-9._-]', '_', smiles)


def optimise_and_write(atoms_list, smiles_list, base_dir="OPTIMISED_STRUCTURES"):
    """
    Optimise molecules with ASE and write xyz files into subdirectories.
    Args:
        atoms_list (list[ase.Atoms]): molecules
        smiles_list (list[str]): original SMILES strings
        base_dir (str): name of parent directory
    """
    parent_dir = os.getcwd()
    optimised_dir = os.path.join(parent_dir, base_dir)

    # Make parent dir if it does not exist
    os.makedirs(optimised_dir, exist_ok=True)

    for atoms, smiles in zip(atoms_list, smiles_list):
        # Create subdir for each molecule (safe name)
        safe_name = safe_smiles_name(smiles)
        mol_dir = os.path.join(optimised_dir, safe_name)
        os.makedirs(mol_dir, exist_ok=True)

        # Run optimisation
        dyn = BFGS(atoms, logfile=os.path.join(mol_dir, f"{safe_name}.log"))
        dyn.run(fmax=0.001)  # eV/Å

        # Write optimised xyz inside subdir
        xyz_path = os.path.join(mol_dir, f"{safe_name}.xyz")
        write(xyz_path, atoms)

if __name__ == "__main__":
    df = read_csv("PubChem_Filtered_Mols.csv")
    smiles_list = df["SMILES"].tolist()
    atoms_list = create_ase_objs(smiles_list)
    optimise_and_write(atoms_list, smiles_list)

