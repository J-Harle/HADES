#!/usr/bin/env python3
import os
import sys
import warnings
import shutil
import hashlib
import contextlib
import numpy as np
import ase.io
from concurrent.futures import ProcessPoolExecutor, as_completed
from ase.vibrations import Vibrations

# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES (MUST BE SET BEFORE TORCH / MACE IMPORT)
# -----------------------------------------------------------------------------
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "0"
os.environ["E3NN_NO_CUEQUIV"] = "1"

# -----------------------------------------------------------------------------
# GLOBAL WARNING SUPPRESSION
# -----------------------------------------------------------------------------
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# -----------------------------------------------------------------------------
# SUPPRESS STDOUT / STDERR CONTEXT MANAGER
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
# IMPORT TORCH / MACE WITH OUTPUT SUPPRESSED
# -----------------------------------------------------------------------------
with suppress_output():
    import torch
    from mace.calculators import MACECalculator

# Limit torch threading (important for multiprocessing)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

calc_dir = os.path.abspath(
    os.path.join(script_dir, "..", "CALCULATORS", "MACE-OFF23_small.model")
)

xyz_dir = os.path.join(script_dir, "OPTIMISED_STRUCTURES", "SMALL_MODEL")

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def file_md5(path):
    """Compute MD5 hash of a file for model verification."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def calculator(model_path):
    """Initialise MACE calculator (must be inside each process)."""
    with suppress_output():
        return MACECalculator(
            model_path=model_path,
            dispersion=False,
            default_dtype="float64",
            device="cpu",
        )

# -----------------------------------------------------------------------------
# CORE WORKER
# -----------------------------------------------------------------------------
def process_xyz(xyz_path):
    """Run vibrational analysis on a single XYZ file."""
    try:
        xyz_file = os.path.basename(xyz_path)
        root = os.path.dirname(xyz_path)

        atoms = ase.io.read(xyz_path)

        # Silence calculator creation + attachment
        with suppress_output():
            atoms.calc = calculator(calc_dir)

        print(
            f"[PID {os.getpid()}] Calculating vibrations for "
            f"{xyz_file} ({len(atoms)} atoms)"
        )

        vib_dir = os.path.join(root, f"vib_temp_{os.getpid()}")

        if os.path.exists(vib_dir):
            shutil.rmtree(vib_dir)

        vib = Vibrations(atoms, name=vib_dir)
        vib.run()

        # ---------------------------------------------------------------------
        # Frequencies and eigenvectors
        # ---------------------------------------------------------------------
        frequencies = vib.get_frequencies()
        modes = [vib.get_mode(i) for i in range(len(frequencies))]

        eigenvectors_path = os.path.join(
            root,
            f"{os.path.splitext(xyz_file)[0]}_eigenvectors.txt"
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

        # ---------------------------------------------------------------------
        # JMOL FILE
        # ---------------------------------------------------------------------
        jmol_path = os.path.join(
            root,
            f"{os.path.splitext(xyz_file)[0]}_jmol.xyz"
        )

        vib_data = vib.get_vibrations()
        vib_data.write_jmol(jmol_path)

        # ---------------------------------------------------------------------
        # CLEANUP
        # ---------------------------------------------------------------------
        vib.clean()
        if os.path.isdir(vib_dir):
            shutil.rmtree(vib_dir)

        print(f"[PID {os.getpid()}] Finished {xyz_file}")
        print(f"Eigenvectors: {eigenvectors_path}")
        print(f"Jmol file:    {jmol_path}")

        return xyz_file, "success"

    except Exception as e:
        return xyz_path, f"failed: {e}"

# -----------------------------------------------------------------------------
# PARALLEL DRIVER
# -----------------------------------------------------------------------------
def calc_vibrations_parallel(ncores=8):
    xyz_files = []

    for root, _, files in os.walk(xyz_dir):
        for f in files:
            if f.endswith(".xyz"):
                xyz_files.append(os.path.join(root, f))

    print(f"Found {len(xyz_files)} structures")
    print(f"Running on {ncores} workers")
    print(f"Model file: {calc_dir}")
    print(f"Model MD5:  {file_md5(calc_dir)}\n")

    with ProcessPoolExecutor(max_workers=ncores) as executor:
        futures = {
            executor.submit(process_xyz, xyz): xyz
            for xyz in xyz_files
        }

        for future in as_completed(futures):
            xyz_file, status = future.result()
            print(f"{xyz_file}: {status}")

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    calc_vibrations_parallel(ncores=8)
