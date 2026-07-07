import os
import sys
import gc
import warnings
import multiprocessing
import contextlib
import urllib.request
import pandas as pd

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
    df = pd.read_csv(filename)
    df.columns = [c.strip().upper() for c in df.columns]
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
            atoms_list.append(None)
            continue

        xyz_block = Chem.MolToXYZBlock(mol)
        atoms = read(StringIO(xyz_block), format="xyz")
        atoms_list.append(atoms)

    return atoms_list

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

    # ---------------- THERMO ----------------
    thermo_success = True
    try:
        vib_dir = os.path.join(mol_dir, "vib")
        vib = Vibrations(atoms, name=os.path.join(vib_dir, "vib"))
        vib.run()

        # ---------------- HESSIAN ----------------
        hessian = vib.get_vibrations().get_hessian()

        natoms = len(atoms)
        hessian = hessian.reshape(3 * natoms, 3 * natoms)
        hessian_path = os.path.join(mol_dir, f"{mol_id}_hessian.txt")
        
        with open(hessian_path, "w") as f:
            f.write("# Cartesian Hessian\n")
            f.write(f"# Dimensions: {hessian.shape[0]} x {hessian.shape[1]}\n")

            for row in hessian:
                f.write(" ".join(f"{x:20.10e}" for x in row) + "\n")

        print(hessian.shape)

        energies = vib.get_energies()
        vib_energies = [e for e in energies if e > 0]

        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            potentialenergy=spe,
            atoms=atoms,
            geometry='nonlinear',
            symmetrynumber=1,
            spin=0,
        )

        T = 298.15
        P = 100000.0

        zpe = thermo.get_ZPE_correction()
        H = thermo.get_enthalpy(T)
        G = thermo.get_gibbs_energy(T, P)

    except Exception as e:
        print(f"[THERMO FAILED] {mol_id}: {e}")
        thermo_success = False

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

        vib.clean()

    del atoms.calc
    return mol_id, xyz_path

# -----------------------------------------------------------------------------
# PARALLEL DRIVER
# -----------------------------------------------------------------------------
def optimise_and_write_parallel(atoms_list, smiles_list, id_list,
                                # base_dir="OPTIMISED_STRUCTURES/SMALL_MODEL/",
                                base_dir="OPTIMISED_STRUCTURES/30_MOL/",
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
    # csv_path = os.path.join(script_dir, "MODULES", "Storm_dataset.csv")
    csv_path = "30_bench.csv"

	
    df = read_csv(csv_path)

    atoms_list = create_ase_objs(df["SMILES"].tolist())

    optimise_and_write_parallel(
        atoms_list,
        df["SMILES"].tolist(),
        df["CID"].tolist(),
        ncores=40
    )

    print("All jobs complete.")

