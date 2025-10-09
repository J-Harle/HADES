import os
import warnings
import numpy as np
import shutil
import ase.io
from ase.vibrations import Vibrations
from mace.calculators import MACECalculator

# Suppress MACE and torch warnings
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*cuequivariance.*")

# Define directories
script_dir = os.path.dirname(os.path.abspath(__file__))
calc_dir = os.path.join(script_dir, "../CALCULATORS/MACE-OFF23_small.model")
xyz_dir = os.path.join(script_dir, "../OPTIMISED_STRUCTURES")

# Define calculator
def calculator(model_path):
    return MACECalculator(
        model_path=model_path,
        dispersion=False,
        default_dtype="float64",
        device="cpu"
    )

# Perform vibration calculation for each XYZ file
def calc_vibrations():
    calc = calculator(calc_dir)
    for root, dirs, files in os.walk(xyz_dir):
        xyz_files = [f for f in files if f.endswith(".xyz")]
        for xyz_file in xyz_files:
            xyz_path = os.path.join(root, xyz_file)
            atoms = ase.io.read(xyz_path)
            atoms.calc = calc

            print(f"Calculating vibrations for {xyz_file} ({len(atoms)} atoms)...")

            # Define vibration directory
            vib_dir = os.path.join(root, "vib_temp")

            # Force remove any existing vibration data to avoid cache corruption
            if os.path.exists(vib_dir):
                print(f"Removing old vibration cache at {vib_dir} ...")
                shutil.rmtree(vib_dir)

            # Perform vibrational analysis
            vib = Vibrations(atoms, name=vib_dir)
            vib.run()

            # Now read the frequencies safely (freshly computed)
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

            vib.clean()  # Remove temporary displacement files

            # Double-check cleanup of temporary directory
            if os.path.isdir(vib_dir):
                shutil.rmtree(vib_dir)

            print(f"Finished {xyz_file}. Saved eigenvectors in {eigenvectors_path}\n")


if __name__ == "__main__":
    calc_vibrations()
