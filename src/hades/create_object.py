"""
Optimise molecules using MACE and write thermochemistry outputs.

This script reads a CSV file containing `CID` and `SMILES` columns, generates
initial 3D structures with RDKit, optimises the structures using MACE through
ASE, calculates vibrational thermochemistry, and writes optimised XYZ,
thermochemistry, and Hessian files.

Each molecule is written to its own subdirectory inside the requested output
directory.
"""

import os
import sys
import warnings
import contextlib
import urllib.request
import pandas as pd
import argparse

# ARGPARSE
def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the input CSV path, requested
        CPU count, and output directory.
    """
    parser = argparse.ArgumentParser(
        description="Optimise molecules using MACE"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input CSV containing CID and SMILES columns"
    )

    parser.add_argument(
        "--cpus", "-c",
        type=int,
        default=-1,
        help="Number of CPUs to use. Use -1 for all available cores. Default: -1"
    )

    parser.add_argument(
        "--outdir", "-dir",
        type=str,
        default="OPTIMISED_STRUCTURES/HADES",
        help="Directory where optimised structures should be written. Default: OPTIMISED_STRUCTURES/HADES"
    )

    return parser.parse_args()


def resolve_ncores(cpus):
    """Resolve the number of worker processes to use.

    Parameters
    ----------
    cpus : int
        Requested CPU count. A value of -1 means use all available logical
        cores.

    Returns
    -------
    int
        Number of worker processes to use.

    Raises
    ------
    ValueError
        If `cpus` is less than 1 and is not equal to -1.
    """
    if cpus == -1:
        return os.cpu_count() or 1

    if cpus < 1:
        raise ValueError("CPU count must be -1 or a positive integer.")

    available = os.cpu_count() or 1

    if cpus > available:
        print(
            f"[WARNING] Requested {cpus} CPUs, but only {available} are available. "
            f"Using {available}."
        )
        return available

    return cpus


# ENVIRONMENT VARIABLES
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "0"
os.environ["E3NN_NO_CUEQUIV"] = "1"

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# SUPPRESS OUTPUT
@contextlib.contextmanager
def suppress_output():
    """Temporarily suppress stdout and stderr.

    This is mainly used to silence verbose output from Torch, MACE, ASE, and
    RDKit during imports, calculator creation, and optimisation setup.
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

# IMPORTS (SUPPRESSED)
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

# CSV
def read_csv(filename):
    """Read the input CSV and return CID and SMILES columns.

    Column names are stripped and converted to uppercase before selecting the
    required columns.

    Parameters
    ----------
    filename : str
        Path to the input CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only the `CID` and `SMILES` columns.

    Raises
    ------
    KeyError
        If the input CSV does not contain `CID` and `SMILES` columns.
    """
    df = pd.read_csv(filename)
    df.columns = [c.strip().upper() for c in df.columns]
    return df[["CID", "SMILES"]]

# RDKit → ASE
def create_ase_objs(
    smiles_list,
    n_conformers=10,
    mmff_max_iters=25):
    """
    Generate multiple conformers for each SMILES, perform a quick MMFF94
    optimisation, select the lowest-energy conformer, and convert only that
    conformer to an ASE Atoms object.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.

    n_conformers : int, optional
        Number of conformers to generate per molecule. Default is 10.

    mmff_max_iters : int, optional
        Maximum MMFF optimisation iterations for each conformer.
        Default is 25.

    Returns
    -------
    list[ase.Atoms or None]
        Lowest-energy conformer for each molecule as an ASE Atoms object.
        None is returned when conformer generation fails.
    """

    atoms_list = []

    for smiles in tqdm(smiles_list, desc="Conformer search"):

        # ---------------------------------------------------------
        # Build molecule
        # ---------------------------------------------------------
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"[SMILES FAILED] {smiles}")
            atoms_list.append(None)
            continue

        mol = Chem.AddHs(mol)

        # ---------------------------------------------------------
        # Generate conformers
        # ---------------------------------------------------------
        params = AllChem.ETKDGv3()

        # Reproducible conformer generation
        params.randomSeed = 42

        # We want exactly n_conformers where possible, so do not
        # prune geometrically similar conformers at this stage.
        params.pruneRmsThresh = -1.0

        # Avoid internal RDKit threading because the expensive
        # MACE stage is parallelised separately.
        params.numThreads = 1

        try:
            conf_ids = list(
                AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=n_conformers,
                    params=params,
                )
            )
        except Exception as e:
            print(f"[CONFORMER GENERATION FAILED] {smiles}: {e}")
            atoms_list.append(None)
            continue

        if len(conf_ids) == 0:
            print(f"[CONFORMER GENERATION FAILED] {smiles}")
            atoms_list.append(None)
            continue

        # ---------------------------------------------------------
        # Quick MMFF94 optimisation
        # ---------------------------------------------------------
        conformer_energies = []

        if AllChem.MMFFHasAllMoleculeParams(mol):

            mmff_props = AllChem.MMFFGetMoleculeProperties(
                mol,
                mmffVariant="MMFF94",
            )

            for conf_id in conf_ids:
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(
                        mol,
                        mmff_props,
                        confId=int(conf_id),
                    )

                    # Loose/quick optimisation
                    ff.Minimize(maxIts=mmff_max_iters)

                    # MMFF energy in kcal/mol
                    energy = ff.CalcEnergy()

                    conformer_energies.append(
                        (int(conf_id), energy)
                    )

                except Exception:
                    continue

        # ---------------------------------------------------------
        # UFF fallback
        # ---------------------------------------------------------
        elif AllChem.UFFHasAllMoleculeParams(mol):

            print(
                f"[WARNING] MMFF unavailable for {smiles}. "
                f"Using UFF."
            )

            for conf_id in conf_ids:
                try:
                    ff = AllChem.UFFGetMoleculeForceField(
                        mol,
                        confId=int(conf_id),
                    )

                    ff.Minimize(maxIts=mmff_max_iters)

                    energy = ff.CalcEnergy()

                    conformer_energies.append(
                        (int(conf_id), energy)
                    )

                except Exception:
                    continue

        # ---------------------------------------------------------
        # If no force field was available, use first conformer
        # ---------------------------------------------------------
        if not conformer_energies:
            print(
                f"[WARNING] Force-field conformer optimisation failed "
                f"for {smiles}. Using first embedded conformer."
            )

            best_conf_id = int(conf_ids[0])
            best_energy = None

        else:
            # -----------------------------------------------------
            # Select lowest-energy conformer
            # -----------------------------------------------------
            best_conf_id, best_energy = min(
                conformer_energies,
                key=lambda x: x[1],
            )

        # ---------------------------------------------------------
        # Convert ONLY the best conformer to ASE
        # ---------------------------------------------------------
        xyz_block = Chem.MolToXYZBlock(
            mol,
            confId=best_conf_id,
        )

        atoms = read(
            StringIO(xyz_block),
            format="xyz",
        )

        # Optional metadata
        atoms.info["selected_conformer"] = best_conf_id

        if best_energy is not None:
            atoms.info["mmff_energy_kcalmol"] = best_energy

        atoms_list.append(atoms)

    return atoms_list

# MACE
def get_mace_calculator():
    """Create a MACE calculator, downloading the model if needed.

    The MACE-OFF23 small model is stored in a local `CALCULATORS` directory
    relative to the current working directory. If the model file is not present,
    it is downloaded automatically.

    Returns
    -------
    MACECalculator
        Configured MACE calculator using CPU and float64 precision.
    """
    parent_dir = os.getcwd()
    calc_dir = os.path.join(parent_dir, "MODULES", "TOOLS", "CALCULATORS")
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

# WORKER
def optimise_and_write_single(args):
    """Optimise one molecule and write its output files.

    This function is designed to run inside a worker process. It assigns a MACE
    calculator, performs a geometry optimisation, calculates a single-point
    energy, attempts vibrational thermochemistry, writes the Hessian, writes the
    optimised XYZ structure, and writes a thermochemistry summary.

    Parameters
    ----------
    args : tuple
        Tuple containing `(atoms, smiles, mol_id, base_dir)`.

    Returns
    -------
    tuple[str, str]
        Molecule ID and path to the written XYZ file.
    """
    atoms, smiles, mol_id, base_dir = args

    optimised_dir = os.path.abspath(base_dir)
    os.makedirs(optimised_dir, exist_ok=True)

    with suppress_output():
        calc = get_mace_calculator()
        atoms.calc = calc

    mol_dir = os.path.join(optimised_dir, str(mol_id))
    os.makedirs(mol_dir, exist_ok=True)

    # OPTIMISATION 
    dyn = BFGS(atoms, logfile=os.path.join(mol_dir, f"{mol_id}.log"))
    dyn.run(fmax=0.001)

    # SPE
    spe = atoms.get_potential_energy()

    # THERMO
    thermo_success = True
    try:
        vib_dir = os.path.join(mol_dir, "vib")
        vib = Vibrations(atoms, name=os.path.join(vib_dir, "vib"))
        vib.run()

        # HESSIAN
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

    # WRITE XYZ 
    xyz_path = os.path.join(mol_dir, f"{mol_id}.xyz")
    with open(xyz_path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"SMILES: {smiles}  SPE (eV): {spe}\n")
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{s:2} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    # WRITE THERMO
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

# PARALLEL DRIVER
def optimise_and_write_parallel(
    atoms_list,
    smiles_list,
    id_list,
    base_dir,
    ncores=None
):
    """Optimise and write multiple molecules in parallel.

        Molecules with failed RDKit conformer generation, represented by None in
        `atoms_list`, are skipped.

        Parameters
        ----------
        atoms_list : list[ase.Atoms or None]
            ASE atoms objects to optimise.
        smiles_list : list[str]
            SMILES strings corresponding to the atoms objects.
        id_list : list[str]
            Molecule identifiers corresponding to the atoms objects.
        base_dir : str
            Directory where molecule subdirectories should be written.
        ncores : int or None, optional
            Number of worker processes to use.

        Returns
        -------
        list[tuple[str, str]]
            List of molecule IDs and written XYZ file paths.
        """
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

# MAIN
def main():
    args = parse_args()

    csv_path = args.input
    ncores = resolve_ncores(args.cpus)

    outdir = os.path.abspath(
        os.path.join(args.outdir)
    )

    print(f"Input CSV: {csv_path}")
    print(f"Optimised structure directory: {outdir}")
    print(f"CPUs requested: {args.cpus}")
    print(f"CPUs used: {ncores}")

    df = read_csv(csv_path)

    atoms_list = create_ase_objs(df["SMILES"].tolist(),
        n_conformers=10, mmff_max_iters=25)

    optimise_and_write_parallel(
        atoms_list,
        df["SMILES"].tolist(),
        df["CID"].tolist(),
        base_dir=outdir,
        ncores=ncores
    )

    print("All jobs complete.")

if __name__ == "__main__":
    main()