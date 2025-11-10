import os
import warnings
import numpy as np
import shutil
import ase.io
from concurrent.futures import ProcessPoolExecutor, as_completed
from ase.vibrations import Vibrations
from mace.calculators import MACECalculator


# Suppress non-critical warnings from MACE and torch initialisation
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*cuequivariance.*")

script_dir = os.path.dirname(os.path.abspath(__file__))
calc_dir = os.path.join(script_dir, "../CALCULATORS/MACE-OFF23_large.model")
xyz_dir = os.path.join(script_dir, "../OPTIMISED_STRUCTURES/LARGE_MODEL")


def calculator(model_path):
    """
    Create and return a MACE calculator for use in a subprocess.

    Parameters
    ----------
    model_path : str
        Path to the pre-trained MACE model file.

    Returns
    -------
    MACECalculator
        Configured MACE calculator (CPU-based, double precision).
    """
    return MACECalculator(
        model_path=model_path,
        dispersion=False,
        default_dtype="float64",
        device="cpu"
    )


def process_xyz(xyz_path):
    """
    Perform vibrational analysis for a single `.xyz` molecular structure.

    The function:
        1. Loads the structure using ASE.
        2. Runs finite-difference vibrational analysis via `ase.vibrations.Vibrations`.
        3. Writes all vibrational frequencies and eigenvectors to a text file.
        4. Cleans temporary vibration directories.

    Parameters
    ----------
    xyz_path : str
        Absolute path to the `.xyz` file to be processed.

    Returns
    -------
    tuple
        (xyz_file, status)
        - xyz_file : filename of the structure.
        - status : "success" or error message string if the job failed.

    Notes
    -----
    Each process creates a unique temporary directory for vibration data
    (e.g., `vib_temp_<PID>`) to avoid I/O conflicts between parallel workers.
    """
    try:
        xyz_file = os.path.basename(xyz_path)
        root = os.path.dirname(xyz_path)
        atoms = ase.io.read(xyz_path)
        calc = calculator(calc_dir)
        atoms.calc = calc

        print(f"[PID {os.getpid()}] Calculating vibrations for {xyz_file} ({len(atoms)} atoms)")

        vib_dir = os.path.join(root, f"vib_temp_{os.getpid()}")  # unique temp dir per process

        # Clean up any old vibration data
        if os.path.exists(vib_dir):
            shutil.rmtree(vib_dir)

        vib = Vibrations(atoms, name=vib_dir)
        vib.run()

        frequencies = vib.get_frequencies()
        n_modes = len(frequencies)

        eigenvectors_path = os.path.join(root, f"{os.path.splitext(xyz_file)[0]}_eigenvectors.txt")
        with open(eigenvectors_path, 'w', encoding='utf-8') as f:
            f.write('# Vibrational modes and eigenvectors\n')
            f.write('# Format: Mode index, Frequency (cm^-1), Eigenvector (x, y, z per atom)\n')
            for i in range(n_modes):
                freq = frequencies[i]
                mode = vib.get_mode(i)  # shape: (n_atoms, 3)
                f.write(f'\nMode {i+1}: Frequency = {freq:.2f} cm^-1\n')
                for j, atom in enumerate(atoms):
                    vec = mode[j]
                    f.write(f'Atom {j+1} ({atom.symbol}): '
                            f'{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}\n')

        # Clean up temporary vibration data
        vib.clean()
        if os.path.isdir(vib_dir):
            shutil.rmtree(vib_dir)

        print(f"[PID {os.getpid()}] Finished {xyz_file}. Saved to {eigenvectors_path}")
        return xyz_file, "success"

    except Exception as e:
        # Return failure message to main process
        return xyz_path, f"failed: {e}"


def calc_vibrations_parallel(ncores=56):
    """
    Perform vibrational frequency calculations for all `.xyz` files in parallel.

    Parameters
    ----------
    ncores : int, optional
        Number of CPU cores (processes) to use for parallel execution.
        Default is 56.

    Notes
    -----
    - Each subprocess creates its own calculator instance to prevent race conditions.
    - Results (success/failure) are streamed live to stdout.
    """
    xyz_files = []
    for root, dirs, files in os.walk(xyz_dir):
        for f in files:
            if f.endswith(".xyz"):
                xyz_files.append(os.path.join(root, f))

    with ProcessPoolExecutor(max_workers=ncores) as executor:
        futures = {executor.submit(process_xyz, xyz): xyz for xyz in xyz_files}

        for future in as_completed(futures):
            xyz_file, status = future.result()
            print(f"{xyz_file}: {status}")


if __name__ == "__main__":
    calc_vibrations_parallel(ncores=56)
