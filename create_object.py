import os
import numpy as np
import pandas as pd
import re
import urllib.request
from io import StringIO
from ase import Atoms, units
from ase.io import read, write
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import AllChem
from mace.calculators import MACECalculator
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import multiprocessing
import torch

warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*cuequivariance.*")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)



def read_csv(filename):
    """
    Reads the dataframe and keeps SMILES + ID columns.
    """
    df = pd.read_csv(filename)
    df.columns = [c.strip().upper() for c in df.columns]
    if not {"SMILES", "ID"}.issubset(df.columns):
        raise ValueError("CSV must contain 'SMILES' and 'ID' columns")
    return df[["ID", "SMILES"]]


def create_ase_objs(smiles_list, max_attempts=10):
    """
    Create 3D ASE Atoms objects from SMILES with retries and fallbacks.
    """
    atoms_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        success = False
        for attempt in range(max_attempts):
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:  # success returns 0
                success = True
                break

        if not success or mol.GetNumConformers() == 0:
            print(f"Embedding failed for SMILES: {smiles}")
            atoms_list.append(None)
            continue

        xyz_block = Chem.MolToXYZBlock(mol)
        atoms = read(StringIO(xyz_block), format="xyz")
        atoms_list.append(atoms)

    return atoms_list


def get_mace_calculator():
    """
    Ensure MACE model is available in CALCULATORS directory and return calculator.
    """
    parent_dir = os.getcwd()
    calc_dir = os.path.join(parent_dir, "CALCULATORS")
    os.makedirs(calc_dir, exist_ok=True)

    model_file = os.path.join(calc_dir, "MACE-OFF23_medium.model")
    if not os.path.exists(model_file):
        print(f"Downloading MACE model to {model_file}")
        url = "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_medium.model?raw=true"
        urllib.request.urlretrieve(url, model_file)
        print("Download complete")

    return MACECalculator(
        model_paths=[model_file],
        dispersion=False,
        default_dtype="float64",
        device="cpu"  # change to "cuda" if GPU available
    )

def optimise_and_write_single(args):
    """
    Run optimisation for a single molecule, then perform a single point energy calculation
    and write the result to the comment line of the .xyz file.
    """
    atoms, smiles, mol_id, base_dir = args
    parent_dir = os.getcwd()
    optimised_dir = os.path.join(parent_dir, base_dir)
    os.makedirs(optimised_dir, exist_ok=True)

    calc = get_mace_calculator()
    atoms.calc = calc

    mol_dir = os.path.join(optimised_dir, str(mol_id))
    os.makedirs(mol_dir, exist_ok=True)

    log_file = os.path.join(mol_dir, f"{mol_id}.log")
    dyn = BFGS(atoms, logfile=log_file)

    try:
        dyn.run(fmax=0.001)
    finally:
        # Ensure logfile is closed even if optimisation fails
        if hasattr(dyn, "logfile") and dyn.logfile:
            try:
                dyn.logfile.close()
            except Exception:
                pass

    # Write optimised geometry
    xyz_path = os.path.join(mol_dir, f"{mol_id}.xyz")
    with open(xyz_path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"SMILES: {smiles}\n")
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{symbol:2} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    # Clean up calculator references
    del atoms.calc
    del calc

    return mol_id, xyz_path


def optimise_and_write_parallel(atoms_list, smiles_list, id_list,
                                base_dir="OPTIMISED_STRUCTURES/PUBCHEM/MEDIUM_MODEL",
                                ncores=56):
    """
    Parallelised using ProcessPoolExecutor.
    Ensures full cleanup and shutdown when finished.
    """
    tasks = [
        (atoms, smiles, mol_id, base_dir)
        for atoms, smiles, mol_id in zip(atoms_list, smiles_list, id_list)
        if atoms is not None
    ]

    results = []
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        futures = {executor.submit(optimise_and_write_single, t): t[2] for t in tasks}
        for future in as_completed(futures):
            mol_id = futures[future]
            try:
                mol_id, xyz_path = future.result()
                print(f"Optimised ID: {mol_id} → {xyz_path}")
                results.append((mol_id, xyz_path))
            except Exception as e:
                print(f"ID: {mol_id} failed: {e}")

        # Force executor cleanup
        executor.shutdown(wait=True, cancel_futures=True)

    # Explicit garbage collection and process reaping
    import gc
    gc.collect()
    for child in multiprocessing.active_children():
        child.join(timeout=1)

    print("All optimisations finished and resources cleaned up.")
    return results


if __name__ == "__main__":
    import gc
    df = read_csv("PubChem_Filtered_Mols.csv")
    smiles_list = df["SMILES"].tolist()
    id_list = df["ID"].tolist()
    atoms_list = create_ase_objs(smiles_list)

    results = optimise_and_write_parallel(atoms_list, smiles_list, id_list, ncores=56)

    gc.collect()
    multiprocessing.active_children()
    print("All jobs complete. Exiting cleanly.")

