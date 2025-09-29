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


def optimise_and_write(atoms_list, prefix="mol"):
    """
    Optimise molecules with ASE and write xyz files.
    Args:
        atoms_list (list[ase.Atoms]): molecules
        prefix (str): filename prefix
    """
    for i, atoms in enumerate(atoms_list):
        dyn = BFGS(atoms, logfile=f"{prefix}_{i}.log")
        dyn.run(fmax=0.001)  # eV/Å
        write(f"{prefix}_{i}.xyz", atoms) # This really needs a better naming convention


if __name__ == "__main__":
    df = read_csv("__file__")
    atoms_list = create_ase_objs(df["SMILES"].tolist())
    optimise_and_write(atoms_list)
