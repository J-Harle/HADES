import os
import sys
import gc
import warnings
import multiprocessing
import contextlib
import urllib.request
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "0"
os.environ["E3NN_NO_CUEQUIV"] = "1"

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# -----------------------------------------------------------------------------
# SUPPRESS OUTPUT
# -----------------------------------------------------------------------------
@contextlib.contextmanager
def suppress_output():
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
# IMPORTS (SUPPRESSED)
# -----------------------------------------------------------------------------
with suppress_output():
    import torch
    from mace.calculators import MACECalculator
    from ase.io import read
    from ase.optimize import BFGS
    from ase.vibrations import Vibrations
    from ase.thermochemistry import IdealGasThermo
    from rdkit import Chem
    from rdkit.Chem import AllChem

from tqdm import tqdm
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# -----------------------------------------------------------------------------
# CSV
# -----------------------------------------------------------------------------
def read_csv(filename):
    df = pd.read_csv(filename, dtype=str) 

    df.columns = [c.strip().upper() for c in df.columns]

    df = df[["CID", "SMILES"]]

    # Clean whitespace
    df["SMILES"] = df["SMILES"].str.strip()
    df["CID"] = df["CID"].str.strip()

    # Drop junk rows
    df = df.dropna(subset=["SMILES"])
    df = df[df["SMILES"] != ""]
    df = df[df["SMILES"].str.lower() != "nan"]

    return df

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
            atoms_list.append(None)
            continue

        xyz_block = Chem.MolToXYZBlock(mol)
        atoms = read(StringIO(xyz_block), format="xyz")
        atoms_list.append(atoms)

    return atoms_list

# -----------------------------------------------------------------------------
# GEOMETRY REGENERATION
# -----------------------------------------------------------------------------
def regenerate_atoms(smiles, method="rdkit", atoms=None, noise=0.05):
    if method == "rdkit":
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return None

        xyz_block = Chem.MolToXYZBlock(mol)
        return read(StringIO(xyz_block), format="xyz")

    elif method == "perturb" and atoms is not None:
        new_atoms = atoms.copy()
        disp = noise * (np.random.rand(len(new_atoms), 3) - 0.5)
        new_atoms.set_positions(new_atoms.get_positions() + disp)
        return new_atoms

    return None

# -----------------------------------------------------------------------------
# MACE
# -----------------------------------------------------------------------------
def get_mace_calculator():
    parent_dir = os.getcwd()
    calc_dir = os.path.join(parent_dir, "CALCULATORS")
    os.makedirs(calc_dir, exist_ok=True)

    model_file = os.path.join(calc_dir, "MACE-OFF23_small.model")

    if not os.path.exists(model_file):
        url = (
            "https://github.com/ACEsuit/mace-off/blob/main/"
            "mace_off23/MACE-OFF23_small.model?raw=true"
        )
        urllib.request.urlretrieve(url, model_file)

    with suppress_output():
        calc = MACECalculator(
            model_paths=[model_file],
            dispersion=False,
            default_dtype="float64",
            device="cpu",
        )
    return calc

# -----------------------------------------------------------------------------
# WORKER
# -----------------------------------------------------------------------------
def optimise_and_write_single(args):
    atoms, smiles, mol_id, base_dir = args

    parent_dir = os.getcwd()
    optimised_dir = os.path.join(parent_dir, base_dir)
    os.makedirs(optimised_dir, exist_ok=True)

    with suppress_output():
        calc = get_mace_calculator()
        atoms.calc = calc

    mol_dir = os.path.join(optimised_dir, str(mol_id))
    os.makedirs(mol_dir, exist_ok=True)

    # ---------------- OPTIMISATION ----------------
    dyn = BFGS(atoms, logfile=os.path.join(mol_dir, f"{mol_id}.log"))
    dyn.run(fmax=0.001)

    # ---------------- SPE ----------------
    spe = atoms.get_potential_energy()

    # ---------------- THERMO WITH RETRIES ----------------
    thermo_success = False
    max_attempts = 25

    for attempt in range(max_attempts):
        try:
            vib_dir = os.path.join(mol_dir, f"vib_{attempt}")
            vib = Vibrations(atoms, name=os.path.join(vib_dir, "vib"))
            vib.run()

            energies = vib.get_energies()
            vib_energies = [e for e in energies if e > 1e-6]

            if len(vib_energies) == 0:
                raise RuntimeError("No positive vibrational modes")

            thermo = IdealGasThermo(
                vib_energies=vib_energies,
                potentialenergy=spe,
                atoms=atoms,
                geometry='nonlinear',
                symmetrynumber=1,
                spin=0,
            )

            T = 300.00
            P = 101000.0

            zpe = thermo.get_ZPE_correction()
            H = thermo.get_enthalpy(T)
            G = thermo.get_gibbs_energy(T, P)

            thermo_success = True
            vib.clean()
            break

        except Exception as e:
            print(f"[THERMO FAILED - attempt {attempt}] {mol_id}: {e}")

            # --- regenerate geometry ---
            if attempt % 2 == 0:
                new_atoms = regenerate_atoms(smiles, method="perturb", atoms=atoms)
            else:
                new_atoms = regenerate_atoms(smiles, method="rdkit")

            if new_atoms is None:
                continue

            with suppress_output():
                calc = get_mace_calculator()
                new_atoms.calc = calc

            dyn = BFGS(new_atoms, logfile=os.path.join(mol_dir, f"{mol_id}_retry{attempt}.log"))
            dyn.run(fmax=0.001)

            atoms = new_atoms
            spe = atoms.get_potential_energy()

    # ---------------- WRITE XYZ ----------------
    xyz_path = os.path.join(mol_dir, f"{mol_id}.xyz")
    with open(xyz_path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"SMILES: {smiles}  SPE (eV): {spe}\n")
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{s:2} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    # ---------------- WRITE THERMO ----------------
    if thermo_success:
        thermo_path = os.path.join(mol_dir, f"{mol_id}_thermo.txt")
        with open(thermo_path, "w") as f:
            f.write("Thermochemistry from MACE + ASE IdealGasThermo\n")
            f.write(f"Temperature : {T} K\n")
            f.write(f"Pressure    : {P/1000:.1f} kPa (1 atm)\n")
            f.write("Geometry    : nonlinear\n\n")

            f.write(f"Electronic energy E0          : {spe:.10f} eV\n")
            f.write(f"Zero-point energy (ZPE)       : {zpe:.10f} eV\n")
            f.write(f"Enthalpy H({T} K)            : {H:.10f} eV\n")
            f.write(f"Gibbs free energy G({T} K)   : {G:.10f} eV\n")

    del atoms.calc
    return mol_id, xyz_path

# -----------------------------------------------------------------------------
# PARALLEL DRIVER
# -----------------------------------------------------------------------------
def optimise_and_write_parallel(atoms_list, smiles_list, id_list,
                                # base_dir="../OPTIMISED_STRUCTURES/SMALL_MODEL/AROMATIC",
                                base_dir="../OPTIMISED_STRUCTURES/DET_V_P_TEST",
                                # base_dir="../OPTIMISED_STRUCTURES/LARGE_DATASET",
                                ncores=None):

    tasks = [
        (atoms, smiles, mol_id, base_dir)
        for atoms, smiles, mol_id in zip(atoms_list, smiles_list, id_list)
        if atoms is not None
    ]

    results = []

    with ProcessPoolExecutor(max_workers=ncores) as executor:
        futures = {executor.submit(optimise_and_write_single, t): t[2] for t in tasks}

        with tqdm(total=len(tasks)) as pbar:
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Failed: {e}")
                pbar.update(1)

    return results

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # csv_path = os.path.join(script_dir, "..", "large_data.csv")
    csv_path = os.path.join(script_dir, "..", "bak_30_bench.csv")

    df = read_csv(csv_path)

    atoms_list = create_ase_objs(df["SMILES"].tolist())

    optimise_and_write_parallel(
        atoms_list,
        df["SMILES"].tolist(),
        df["CID"].tolist(),
        ncores=38,
    )

    print("All jobs complete.")
