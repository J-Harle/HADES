import os
import numpy as np
import pandas as pd
import re
import urllib.request
import gc
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

# Suppress non-critical warnings from PyTorch and cuequivariance
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*cuequivariance.*")

# Restrict PyTorch thread usage to prevent oversubscription in parallel execution
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def read_csv(filename):
    """
    Read a CSV file containing SMILES strings and molecule IDs.

    The function ensures that the CSV contains at least the columns
    'SMILES' and 'ID' (case-insensitive), and returns a DataFrame
    restricted to those columns.

    Parameters
    ----------
    filename : str
        Path to the input CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns: 'ID' and 'SMILES'.

    Raises
    ------
    ValueError
        If required columns ('SMILES', 'ID') are missing.
    """
    df = pd.read_csv(filename)
    df.columns = [c.strip().upper() for c in df.columns]
    if not {"SMILES", "ID"}.issubset(df.columns):
        raise ValueError("CSV must contain 'SMILES' and 'ID' columns")
    return df[["ID", "SMILES"]]


def create_ase_objs(smiles_list, max_attempts=10):
    """
    Generate ASE Atoms objects from a list of SMILES strings.

    The function attempts to create 3D molecular geometries from SMILES
    using RDKit’s ETKDG embedding algorithm. It retries embedding
    multiple times if needed, and logs failed molecules.

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings to convert to ASE Atoms objects.
    max_attempts : int, optional
        Maximum number of embedding attempts per molecule (default: 10).

    Returns
    -------
    list of ase.Atoms or None
        List of ASE Atoms objects corresponding to input SMILES strings.
        Molecules that fail embedding are returned as None.
    """
    atoms_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        success = False
        for attempt in range(max_attempts):
            # ETKDG: distance geometry embedding with experimental torsion preferences
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
                success = True
                break

        if not success or mol.GetNumConformers() == 0:
            print(f"Embedding failed for SMILES: {smiles}")
            atoms_list.append(None)
            continue

        # Convert RDKit Mol object to XYZ and read as ASE Atoms
        xyz_block = Chem.MolToXYZBlock(mol)
        atoms = read(StringIO(xyz_block), format="xyz")
        atoms_list.append(atoms)

    return atoms_list


def get_mace_calculator():
    """
    Retrieve or download the MACE-OFF23 medium model and create an ASE calculator.

    The function checks for a pre-existing model file within a local
    `CALCULATORS` directory. If absent, it downloads the model from GitHub.

    Returns
    -------
    mace.calculators.MACECalculator
        A configured MACE calculator instance using double precision (float64)
        and CPU device by default.

    Notes
    -----
    - Dispersion corrections are disabled.
    - GPU computation can be enabled by setting device="cuda".
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
    Optimise the geometry of a single molecule and write results to disk.

    Performs energy minimisation using the MACE calculator and ASE’s BFGS
    optimiser. Upon completion, the optimised coordinates are written
    to an `.xyz` file, and the SMILES string is recorded in the comment line.

    Parameters
    ----------
    args : tuple
        (atoms, smiles, mol_id, base_dir)
        - atoms (ase.Atoms): initial molecular geometry.
        - smiles (str): SMILES string of the molecule.
        - mol_id (str or int): molecule identifier.
        - base_dir (str): parent directory for saving results.

    Returns
    -------
    tuple
        (mol_id, xyz_path) containing the molecule ID and path to the
        optimised `.xyz` file.

    Notes
    -----
    - The BFGS optimiser stops when the maximum force component falls below 0.001 eV/Å.
    - If optimisation fails, the function still ensures logfile closure.
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
        # Ensure logfile closure even if optimisation fails
        if hasattr(dyn, "logfile") and dyn.logfile:
            try:
                dyn.logfile.close()
            except Exception:
                pass

    # Write optimised geometry to XYZ
    xyz_path = os.path.join(mol_dir, f"{mol_id}.xyz")
    with open(xyz_path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"SMILES: {smiles}\n")
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{symbol:2} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    # Clean up references
    del atoms.calc
    del calc

    return mol_id, xyz_path


def optimise_and_write_parallel(atoms_list, smiles_list, id_list,
                                base_dir="OPTIMISED_STRUCTURES/PUBCHEM/MEDIUM_MODEL",
                                ncores=4):
    """
    Run geometry optimisations in parallel using multiple CPU cores.

    Each molecule is processed independently in its own subprocess
    to ensure isolation and memory safety. The function ensures clean
    shutdown of all subprocesses and garbage collection after completion.

    Parameters
    ----------
    atoms_list : list of ase.Atoms or None
        Molecular geometries generated from SMILES.
    smiles_list : list of str
        SMILES strings corresponding to molecules.
    id_list : list of str or int
        Unique molecule identifiers.
    base_dir : str, optional
        Base output directory for optimised structures.
    ncores : int, optional
        Number of CPU cores for parallel execution (default: 56).

    Returns
    -------
    list of tuple
        List of (mol_id, xyz_path) tuples for successfully optimised molecules.

    Notes
    -----
    - Failed optimisations are caught and logged.
    - All subprocesses are explicitly joined and cleaned up to prevent
      zombie processes in long runs.
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

        # Ensure executor is fully shut down
        executor.shutdown(wait=True, cancel_futures=True)

    # Force garbage collection and process cleanup
    import gc
    gc.collect()
    for child in multiprocessing.active_children():
        child.join(timeout=1)

    print("All optimisations finished and resources cleaned up.")
    return results


if __name__ == "__main__":

    df = read_csv("PubChem_Filtered_Mols.csv")
    smiles_list = df["SMILES"].tolist()
    id_list = df["ID"].tolist()
    atoms_list = create_ase_objs(smiles_list)

    results = optimise_and_write_parallel(atoms_list, smiles_list, id_list, ncores=4)

    gc.collect()
    multiprocessing.active_children()
    print("All jobs complete. Exiting cleanly.")
