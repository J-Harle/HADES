"""
Calculate MACE vibrational frequencies and eigenvectors for XYZ files.

This script searches a directory tree for optimised XYZ structures, assigns a
MACE calculator to each structure, runs ASE vibrational analysis, and writes
eigenvector and Jmol-compatible output files for each molecule.

The script supports multiprocessing and is intended to be called either directly
or through the main HADES workflow.
"""

import os
import sys
import argparse
import warnings
import shutil
import hashlib
import contextlib
import numpy as np
import ase.io
from concurrent.futures import ProcessPoolExecutor, as_completed
from ase.vibrations import Vibrations


# ENVIRONMENT VARIABLES (MUST BE SET BEFORE TORCH / MACE IMPORT)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "0"
os.environ["E3NN_NO_CUEQUIV"] = "1"


# GLOBAL WARNING SUPPRESSION
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


# SUPPRESS STDOUT / STDERR CONTEXT MANAGER
@contextlib.contextmanager
def suppress_output():
    """Temporarily suppress stdout and stderr.

    This is mainly used to silence verbose output from Torch, MACE, and ASE
    calculator initialisation.
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


# IMPORT TORCH / MACE WITH OUTPUT SUPPRESSED
with suppress_output():
    import torch
    from mace.calculators import MACECalculator


# Limit torch threading. Important for multiprocessing.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# CLI
script_dir = os.path.dirname(os.path.abspath(__file__))

default_model_path = os.path.join(
    script_dir,
    "TOOLS","CALCULATORS",
    "MACE-OFF23_small.model"
)

default_xyz_dir = os.path.join(
    "OPTIMISED_STRUCTURES",
    "HADES"
)

parser = argparse.ArgumentParser(
    description="Calculate MACE vibrational frequencies and eigenvectors for XYZ structures"
)

parser.add_argument(
    "--input", "-i",
    type=str,
    default=None,
    help="Optional input CSV path. Included for compatibility with hades.py."
)

parser.add_argument(
    "--outdir", "-dir",
    type=str,
    default=default_xyz_dir,
    help="Directory containing optimised XYZ structures. Default: OPTIMISED_STRUCTURES/HADES"
)

parser.add_argument(
    "--cpus", "-c",
    type=int,
    default=-1,
    help="Number of worker processes. Use -1 for all available cores. Default: -1"
)

parser.add_argument(
    "--model",
    type=str,
    default=default_model_path,
    help="Path to MACE model file"
)

args = parser.parse_args()


# PATH HANDLING
def resolve_path(path):
    """Return the absolute path for a given file or directory path.

    Parameters
    ----------
    path : str
        Input path to resolve.

    Returns
    -------
    str
        Absolute version of the input path.
    """
    return os.path.abspath(path)


calc_dir = resolve_path(args.model)
xyz_dir = resolve_path(args.outdir)


def resolve_ncores(cpus):
    """Resolve the requested number of worker processes.

    Parameters
    ----------
    cpus : int
        Number of CPUs requested from the command line. A value of -1 means use
        all available logical cores.

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
        raise ValueError("Number of CPUs must be -1 or a positive integer")

    return cpus


# UTILITIES
def file_md5(path):
    """Calculate the MD5 hash of a file.

    Parameters
    ----------
    path : str
        Path to the file to hash.

    Returns
    -------
    str
        MD5 hash string for the file.
    """
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)

    return h.hexdigest()


def calculator(model_path):
    """Create and return a MACE calculator.

    The calculator must be initialised inside each worker process rather than
    shared between processes.

    Parameters
    ----------
    model_path : str
        Path to the MACE model file.

    Returns
    -------
    MACECalculator
        Configured MACE calculator using CPU and float64 precision.
    """
    with suppress_output():
        return MACECalculator(
            model_path=model_path,
            dispersion=False,
            default_dtype="float64",
            device="cpu",
        )


# CORE WORKER
def process_xyz(xyz_path):
    """Run vibrational analysis for a single XYZ structure.

    This function reads an XYZ file, attaches a MACE calculator, runs ASE
    vibrational analysis, writes eigenvectors, writes a Jmol-compatible XYZ file,
    and removes temporary vibration files.

    Parameters
    ----------
    xyz_path : str
        Path to the input XYZ file.

    Returns
    -------
    tuple[str, str]
        Tuple containing the processed XYZ filename or path and a status message.
    """
    try:
        xyz_file = os.path.basename(xyz_path)
        root = os.path.dirname(xyz_path)
        mol_name = os.path.splitext(xyz_file)[0]

        atoms = ase.io.read(xyz_path)

        with suppress_output():
            atoms.calc = calculator(calc_dir)

        print(
            f"[PID {os.getpid()}] Calculating vibrations for "
            f"{xyz_file} ({len(atoms)} atoms)"
        )

        vib_dir = os.path.join(root, f"vib_temp_{mol_name}_{os.getpid()}")

        if os.path.exists(vib_dir):
            shutil.rmtree(vib_dir)

        vib = Vibrations(atoms, name=vib_dir)
        vib.run()

        # Frequencies and eigenvectors
        frequencies = vib.get_frequencies()
        modes = [vib.get_mode(i) for i in range(len(frequencies))]

        eigenvectors_path = os.path.join(
            root,
            f"{mol_name}_eigenvectors.txt"
        )

        with open(eigenvectors_path, "w", encoding="utf-8") as f:
            f.write("# Vibrational modes and eigenvectors\n")
            f.write("# Mode index, Frequency (cm^-1), Eigenvector per atom\n")
            f.write(f"# Number of atoms: {len(atoms)}\n")

            for i, freq in enumerate(frequencies):
                f.write(f"\nMode {i + 1}: Frequency = {freq:.2f} cm^-1\n")

                mode = modes[i]

                for j, atom in enumerate(atoms):
                    vec = mode[j]
                    f.write(
                        f"Atom {j + 1} ({atom.symbol}): "
                        f"{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}\n"
                    )

        # JMOL FILE
        jmol_path = os.path.join(
            root,
            f"{mol_name}_jmol.xyz"
        )

        vib_data = vib.get_vibrations()
        vib_data.write_jmol(jmol_path)

        # CLEANUP
        vib.clean()

        if os.path.isdir(vib_dir):
            shutil.rmtree(vib_dir)

        print(f"[PID {os.getpid()}] Finished {xyz_file}")
        print(f"Eigenvectors: {eigenvectors_path}")
        print(f"Jmol file:    {jmol_path}")

        return xyz_file, "success"

    except Exception as e:
        return xyz_path, f"failed: {e}"


# PARALLEL DRIVER
def calc_vibrations_parallel(ncores):
    """Run vibrational analysis for all XYZ files in parallel.

    The function recursively searches the configured XYZ directory for `.xyz`
    files, checks that the requested model and input directory exist, and then
    distributes each vibration calculation across a process pool.

    Parameters
    ----------
    ncores : int
        Number of worker processes to use.

    Raises
    ------
    FileNotFoundError
        If the requested MACE model file does not exist.
    NotADirectoryError
        If the requested XYZ directory does not exist.
    """
    xyz_files = []

    for root, _, files in os.walk(xyz_dir):
        for f in files:
            if f.endswith(".xyz") and not f.startswith("._") and "jmol" not in f.lower():
                xyz_files.append(os.path.join(root, f))

    xyz_files = sorted(xyz_files)

    print("\nVibration calculation settings")
    print("-" * 50)

    if args.input is not None:
        print(f"Input CSV: {resolve_path(args.input)}")

    print(f"XYZ directory: {xyz_dir}")
    print(f"Model file:    {calc_dir}")
    print(f"Running on:    {ncores} workers")
    print(f"Found:         {len(xyz_files)} XYZ structures")

    if not os.path.isfile(calc_dir):
        raise FileNotFoundError(f"MACE model file not found: {calc_dir}")

    if not os.path.isdir(xyz_dir):
        raise NotADirectoryError(f"XYZ directory not found: {xyz_dir}")

    print(f"Model MD5:     {file_md5(calc_dir)}")
    print("-" * 50)

    if not xyz_files:
        print("No XYZ files found. Nothing to do.")
        return

    with ProcessPoolExecutor(max_workers=ncores) as executor:
        futures = {
            executor.submit(process_xyz, xyz): xyz
            for xyz in xyz_files
        }

        for future in as_completed(futures):
            xyz_file, status = future.result()
            print(f"{xyz_file}: {status}")

def main():

    ncores = resolve_ncores(args.cpus)
    calc_vibrations_parallel(ncores=ncores)



if __name__ == "__main__":
    main()