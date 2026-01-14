import os
import sys
import gc
import warnings
import multiprocessing
import contextlib
import urllib.request
import pandas as pd

# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES TO MUTE E3NN / MACE WARNINGS
# -----------------------------------------------------------------------------
# Must be set before importing MACE / e3nn / torch
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "0"
os.environ["E3NN_NO_CUEQUIV"] = "1"

# -----------------------------------------------------------------------------
# GLOBAL SUPPRESSION (Python warnings)
# -----------------------------------------------------------------------------
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# -----------------------------------------------------------------------------
# SUPPRESS OUTPUT CONTEXT MANAGER
# -----------------------------------------------------------------------------
@contextlib.contextmanager
def suppress_output():
    """
    Suppress stdout and stderr (for noisy libraries in multiprocessing).
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# -----------------------------------------------------------------------------
# IMPORT MACE / torch / ASE / RDKit AFTER ENV VARS WITH OUTPUT SUPPRESSED
# -----------------------------------------------------------------------------
with suppress_output():
    import torch
    from mace.calculators import MACECalculator
    from ase.io import read
    from ase.optimize import BFGS
    from rdkit import Chem
    from rdkit.Chem import AllChem

from tqdm import tqdm
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed

# Limit Torch threads to avoid excessive multiprocessing overhead
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def read_csv(filename):
    df = pd.read_csv(filename)
    df.columns = [c.strip().upper() for c in df.columns]
    if not {"SMILES", "CID"}.issubset(df.columns):
        raise ValueError("CSV must contain 'SMILES' and 'CID' columns")
    return df[["CID", "SMILES"]]

# -----------------------------------------------------------------------------
# RDKit → ASE
# -----------------------------------------------------------------------------

def create_ase_objs(smiles_list, max_attempts=10):
    atoms_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        success = False
        for _ in range(max_attempts):
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
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

# -----------------------------------------------------------------------------
# MACE calculator
# -----------------------------------------------------------------------------

def get_mace_calculator():
    parent_dir = os.getcwd()
    calc_dir = os.path.join(parent_dir, "CALCULATORS")
    os.makedirs(calc_dir, exist_ok=True)

    model_file = os.path.join(calc_dir, "MACE-OFF23_small.model")
    if not os.path.exists(model_file):
        print(f"Downloading MACE model to {model_file}")
        url = (
            "https://github.com/ACEsuit/mace-off/blob/main/"
            "mace_off23/MACE-OFF23_small.model?raw=true"
        )
        urllib.request.urlretrieve(url, model_file)
        print("Download complete")

    with suppress_output():
        calc = MACECalculator(
            model_paths=[model_file],
            dispersion=False,
            default_dtype="float64",
            device="cpu",
        )
    return calc

# -----------------------------------------------------------------------------
# SINGLE WORKER TASK
# -----------------------------------------------------------------------------

def optimise_and_write_single(args):
    atoms, smiles, mol_id, base_dir = args
    warnings.simplefilter("ignore", UserWarning)

    parent_dir = os.getcwd()
    optimised_dir = os.path.join(parent_dir, base_dir)
    os.makedirs(optimised_dir, exist_ok=True)

    # Silence MACE / e3nn / torch initialisation noise
    with suppress_output():
        calc = get_mace_calculator()
        atoms.calc = calc

    mol_dir = os.path.join(optimised_dir, str(mol_id))
    os.makedirs(mol_dir, exist_ok=True)

    log_file = os.path.join(mol_dir, f"{mol_id}.log")
    dyn = BFGS(atoms, logfile=log_file)

    try:
        dyn.run(fmax=0.001)
    finally:
        if hasattr(dyn, "logfile") and dyn.logfile:
            try:
                dyn.logfile.close()
            except Exception:
                pass

    xyz_path = os.path.join(mol_dir, f"{mol_id}.xyz")
    with open(xyz_path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"SMILES: {smiles}\n")
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{symbol:2} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    del atoms.calc
    del calc
    return mol_id, xyz_path

# -----------------------------------------------------------------------------
# PARALLEL DRIVER
# -----------------------------------------------------------------------------

def optimise_and_write_parallel(atoms_list, smiles_list, id_list,
                                base_dir="../OPTIMISED_STRUCTURES/SMALL_MODEL",
                                ncores=1):
    tasks = [
        (atoms, smiles, mol_id, base_dir)
        for atoms, smiles, mol_id in zip(atoms_list, smiles_list, id_list)
        if atoms is not None
    ]

    n_tasks = len(tasks)
    print(f"Starting optimisation of {n_tasks} molecules using {ncores} cores")

    results = []

    with ProcessPoolExecutor(max_workers=ncores) as executor:
        futures = {executor.submit(optimise_and_write_single, task): task[2] for task in tasks}

        with tqdm(total=n_tasks, desc="Optimising molecules", unit="mol", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                mol_id = futures[future]
                try:
                    mol_id, xyz_path = future.result()
                    results.append((mol_id, xyz_path))
                except Exception as e:
                    print(f"CID: {mol_id} failed: {e}")
                finally:
                    pbar.update(1)

        executor.shutdown(wait=True, cancel_futures=True)

    gc.collect()
    for child in multiprocessing.active_children():
        child.join(timeout=1)

    print("All optimisations finished and resources cleaned up.")
    return results

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "hades_out.csv")

    df = read_csv(csv_path)
    smiles_list = df["SMILES"].tolist()
    id_list = df["CID"].tolist()

    atoms_list = create_ase_objs(smiles_list)

    results = optimise_and_write_parallel(
        atoms_list,
        smiles_list,
        id_list,
        ncores=1,
    )

    gc.collect()
    multiprocessing.active_children()
    print("All jobs complete. Exiting cleanly.")
