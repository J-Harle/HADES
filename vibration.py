import os
import warnings
import numpy as np
import shutil
import hashlib
import ase.io
from concurrent.futures import ProcessPoolExecutor, as_completed
from ase.vibrations import Vibrations
from mace.calculators import MACECalculator


# Suppress MACE and torch warnings
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*cuequivariance.*")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
calc_dir = os.path.abspath(
    os.path.join(script_dir, "CALCULATORS", "MACE-OFF23_small.model")
)
xyz_dir = os.path.join(script_dir, "OPTIMISED_STRUCTURES", "SMALL_MODEL")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def file_md5(path):
    """Compute MD5 hash of a file for model verification."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def calculator(model_path):
    """Initialise a new calculator (must be done inside each process)."""
    print(f"[PID {os.getpid()}] Using model file: {model_path}")
    print(f"[PID {os.getpid()}] Model hash: {file_md5(model_path)}")

    return MACECalculator(
        model_path=model_path,
        dispersion=False,
        default_dtype="float64",
        device="cpu"
    )


# -----------------------------------------------------------------------------
# Core worker
# -----------------------------------------------------------------------------
def process_xyz(xyz_path):
    """Run vibrational analysis on a single XYZ file."""
    try:
        xyz_file = os.path.basename(xyz_path)
        root = os.path.dirname(xyz_path)

        atoms = ase.io.read(xyz_path)
        atoms.calc = calculator(calc_dir)

        print(
            f"[PID {os.getpid()}] Calculating vibrations for "
            f"{xyz_file} ({len(atoms)} atoms)"
        )

        vib_dir = os.path.join(root, f"vib_temp_{os.getpid()}")

        # Clean old vibration data if it exists
        if os.path.exists(vib_dir):
            shutil.rmtree(vib_dir)

        vib = Vibrations(atoms, name=vib_dir)
        vib.run()

        # ---------------------------------------------------------------------
        # Frequencies and modes (for your text output)
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

        jmol_path = os.path.join(
            root,
            f"{os.path.splitext(xyz_file)[0]}_jmol.xyz"
        )

        vib_data = vib.get_vibrations()
        vib_data.write_jmol(jmol_path)

        # ---------------------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------------------
        vib.clean()
        if os.path.isdir(vib_dir):
            shutil.rmtree(vib_dir)

        print(f"[PID {os.getpid()}] Finished {xyz_file}")
        print(f"  → Eigenvectors: {eigenvectors_path}")
        print(f"  → Jmol file:    {jmol_path}")

        return xyz_file, "success"

    except Exception as e:
        return xyz_path, f"failed: {e}"


# -----------------------------------------------------------------------------
# Parallel driver
# -----------------------------------------------------------------------------
def calc_vibrations_parallel(ncores=10):
    """Run vibrations in parallel across multiple XYZ files."""
    xyz_files = []

    for root, dirs, files in os.walk(xyz_dir):
        for f in files:
            if f.endswith(".xyz"):
                xyz_files.append(os.path.join(root, f))

    print(f"Found {len(xyz_files)} structures.")
    print(f"Running on {ncores} workers\n")
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
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    calc_vibrations_parallel(ncores=10)
