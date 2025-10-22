import os
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

# Suppress specific warnings to reduce console clutter
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Dictionary containing molecule data: (experimental impact sensitivity, molecule reference number, torsion frequency)
i50_values = {
    "TET-1": (2, 1,),# 76.5324),
    "NP1": (2.5, 2,),# 66.1026),
    "HNB": (2.75, 3,),#, 83.0775),
    "PETN": (3, 4,),# 56.2282),
    "CL20": (3, 5,), #81.0393),
    "BP-1": (4.5, 6,),# 78.9906),
    "TET-2": (5, 7,),# 72.5534),
    "CH-PETN": (6, 8,),# 45.1597),
    "HMX": (8, 9,),# 108.3263),
    "TET-3": (8, 10,),# 86.7160),
    "RDX": (13, 11,),# 77.8322),
    "TNT-1": (14, 12,),# 86.9846),
    "PCA": (16, 13,),# 85.0698),
    "TET-4": (15, 14,),# 99.7584),
    "BP-2": (20, 15,),# 92.2816),
    "ADNP-5": (23, 16,),# 103.7134),
    "TNT": (24.5, 17,),# 49.0603),
    "DNP-35": (25, 18,),# 69.3667),
    "DNP-13": (25, 19,),# 52.6453),
    "TNT-2": (26.8, 20,),# 89.2217),
    "BP-3": (30, 21,),# 35.4174),
    "FOX7": (31, 22,),# 65.2550),
    "DNP-34": (40, 23,),# 56.9645),
    "TET-5": (40, 24,),# 96.6349),
    "ADNP-4": (41, 25,),# 105.6198),
    "nitrotriazolone": (73, 26),# 43.2361),
    "nitroguanidine": (80, 27,),# 108.1432),
    "TET-6": (100, 28,),# 87.0307),
    "TATB": (120, 29,),# 84.7751),
}

# Sort i50_values by experimental impact sensitivity (I50)
sorted_i50 = dict(sorted(i50_values.items(), key=lambda item: item[1]))

# Create directory structure for saving plots
def create_plot_directories(script_name=None):
    """Create a directory structure for storing plots under the 'FIGURES' directory.

    Parameters
    ----------
    script_name : str, optional
        Name of the script; defaults to the current script name if None.

    Returns
    -------
    str
        Path to the base directory for plots.

    Notes
    -----
    Creates subdirectories for different types of plots (e.g., DOS, Box_DOS) under the base directory.
    """
    if script_name is None:
        script_name = os.path.splitext(os.path.basename(__file__))[0]
    
    figures_dir = os.path.join(os.getcwd(), "UPPUMPING/FIGURES")
    base_dir = os.path.join(figures_dir, f"{script_name}")
    
    subdirs = [
        "DOS", "Box_DOS", "BE_Scaled_DOS", "First_Convolution",
        "Second_Convolution", "NO2_Angle", "Max_Min_Normalised",
        "NO2_Angle_Chosen_Correct"
    ]
    
    try:
        os.makedirs(figures_dir, exist_ok=True)
        print(f"Created figures directory: {figures_dir}")
        
        os.makedirs(base_dir, exist_ok=True)
        print(f"Created base directory: {base_dir}")
        
        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            print(f"Created subdirectory: {subdir_path}")
            
    except Exception as e:
        print(f"Error creating directories: {e}")
        
    return base_dir

# Read vibrational data from .txt and .xyz files
def read_txt_xyz_files(xyz_dir):
    """Retrieve vibrational data and coordinates from .txt and .xyz files.

    Formats data for the main script in the expected tuple format.

    Parameters
    ----------
    subdir : str, optional
        Subdirectory containing .txt and .xyz files (default: "UPPUMPING/OPTIMISED_STRUCTURES/MEDIUM_MODEL").

    Returns
    -------
    tuple
        - list of tuples: Each tuple contains (filename, atom_count, frequencies, atoms, normal_modes)
        - str: Subdirectory path
    """
    data = []

    atomic_numbers = {"H": 1, "C": 6, "N": 7, "O": 8}

    for root, dirs, files in os.walk(xyz_dir):
        txt_files = [f for f in files if f.endswith(".txt")]
        xyz_files = [f for f in files if f.endswith(".xyz")]

        for txt_file in txt_files:
            txt_path = os.path.join(root, txt_file)
            freqs, coordinates, eigenvectors = [], [], []
            atom_count = None

            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                if "Number of atoms" in line:
                    atom_count = int(line.split(":")[-1].strip())

            mode_index = None
            current_mode_evecs = []

            for line in lines:
                if line.startswith("Mode "):
                    if mode_index is not None and current_mode_evecs:
                        eigenvectors.append(current_mode_evecs)
                        current_mode_evecs = []

                    try:
                        parts = line.split(":")
                        mode_index = int(parts[0].split()[1])
                        freq_part = parts[1].split("=")[1].split("+")[0].strip()
                        freq_real = float(freq_part.split("+")[0])
                        if "j" in freq_part:
                            imag_part = float(freq_part.split("+")[1].replace("j", ""))
                            freqs.append(complex(freq_real, imag_part))
                        else:
                            freqs.append(freq_real)
                    except Exception:
                        continue

                elif line.startswith("Atom "):
                    try:
                        atom_section, coords_section = line.split(":", 1)
                        atom_parts = atom_section.split()
                        atom_index = int(atom_parts[1])
                        symbol = atom_parts[2].strip("()")
                        x, y, z = map(float, coords_section.split(","))
                        current_mode_evecs.append((atom_index, symbol, [x, y, z]))
                    except Exception:
                        continue

            if current_mode_evecs:
                eigenvectors.append(current_mode_evecs)

            filtered_freqs, filtered_eigenvectors = [], []
            for f, evec in zip(freqs, eigenvectors):
                freq_value = f.real if isinstance(f, complex) else f
                if freq_value > 1:
                    filtered_freqs.append(f)
                    filtered_eigenvectors.append(evec)

            txt_stem = os.path.splitext(txt_file)[0]
            xyz_candidate = os.path.join(root, f"{txt_stem.replace('_eigenvectors', '')}.xyz")
            xyz_path = xyz_candidate if os.path.exists(xyz_candidate) else (os.path.join(root, xyz_files[0]) if xyz_files else None)

            if xyz_path and os.path.exists(xyz_path):
                try:
                    with open(xyz_path, "r", encoding="utf-8") as f:
                        xyz_lines = f.readlines()[2:]
                except UnicodeDecodeError:
                    with open(xyz_path, "r", encoding="latin-1", errors="ignore") as f:
                        xyz_lines = f.readlines()[2:]

                for line in xyz_lines:
                    parts = line.split()
                    if len(parts) == 4:
                        symbol = parts[0]
                        x, y, z = map(float, parts[1:])
                        coordinates.append((symbol, x, y, z))
            else:
                print(f"No .xyz file found for {txt_file}")

            # Convert to atom dicts for NO2 detection
            atoms = []
            for i, (symbol, x, y, z) in enumerate(coordinates):
                if symbol in atomic_numbers:
                    atoms.append({"index": i, "Z": atomic_numbers[symbol], "coords": (x, y, z)})

            no2_groups = identify_no2_groups(atoms, bond_threshold)

            data.append({
                "molecule": os.path.basename(root),
                "frequencies": filtered_freqs,
                "atom_count": atom_count,
                "coordinates": coordinates,
                "eigenvectors": filtered_eigenvectors,
                "no2_groups": no2_groups
            })

    return data

# New function to read .txt and .xyz files and format data for the main script
def get_file_data_from_txt_xyz(subdir="UPPUMPING/OPTIMISED_STRUCTURES/MEDIUM_MODEL"):
    """Retrieve vibrational data and coordinates from .txt and .xyz files.

    Formats data for the main script in the expected tuple format.

    Parameters
    ----------
    subdir : str, optional
        Subdirectory containing .txt and .xyz files (default: "UPPUMPING/OPTIMISED_STRUCTURES/MEDIUM_MODEL").

    Returns
    -------
    tuple
        - list of tuples: Each tuple contains (filename, atom_count, frequencies, atoms, normal_modes)
        - str: Subdirectory path
    """
    subdir_path = os.path.join(os.getcwd(), subdir)
    if not os.path.exists(subdir_path):
        print(f"Error: Directory {subdir_path} does not exist")
        return [], subdir_path

    # Call read_txt_xyz_files to get data
    try:
        data_list = read_txt_xyz_files(subdir_path)
    except Exception as e:
        print(f"Error reading files from {subdir_path}: {e}")
        return [], subdir_path

    file_data = []
    for data in data_list:
        molecule_name = data["molecule"]
        atom_count = data["atom_count"]
        frequencies = data["frequencies"]
        coordinates = data["coordinates"]
        eigenvectors = data["eigenvectors"]
        no2_groups = data["no2_groups"]

        # Convert coordinates to atoms format: list of dicts with index, Z, coords
        atomic_numbers = {"H": 1, "C": 6, "N": 7, "O": 8}
        atoms = [
            {
                "index": i + 1,  # 1-based indexing to match original script
                "Z": atomic_numbers.get(symbol, 6),  # Default to carbon if symbol not found
                "coords": (x, y, z)
            }
            for i, (symbol, x, y, z) in enumerate(coordinates)
        ]

        # Convert eigenvectors to normal_modes format: list of np.ndarray with shape (atom_count, 3)
        normal_modes = []
        for evec in eigenvectors:
            mode_array = np.zeros((atom_count, 3))
            for atom_index, _, (dx, dy, dz) in evec:
                mode_array[atom_index - 1] = [dx, dy, dz]  # Adjust for 0-based indexing
            normal_modes.append(mode_array)

        # Ensure frequencies and normal modes align
        if len(frequencies) != len(normal_modes):
            print(f"Error: Mismatch between frequencies ({len(frequencies)}) and normal modes ({len(normal_modes)}) for {molecule_name}")
            continue

        # Use molecule name as filename for compatibility
        filename = f"{molecule_name}.txt"
        file_data.append((filename, atom_count, frequencies, atoms, normal_modes))

    return file_data, subdir_path

# Apply Gaussian broadening to a histogram to create a density of states (DOS)
def gaussian_broadening(histogram, bwidth=0.5, gwidth=2.5):
    """Apply Gaussian broadening to a histogram to generate a smooth density of states (DOS).

    Parameters
    ----------
    histogram : numpy.ndarray
        Input histogram.
    bwidth : float
        Bin width for the histogram.
    gwidth : float
        Gaussian width for broadening.

    Returns
    -------
    numpy.ndarray
        Broadened density of states.
    """
    len_bins = len(histogram)
    dos = np.zeros(len_bins)
    sigma = gwidth / 2.354  # Convert FWHM to standard deviation
    if gwidth < bwidth:
        dos[:] = histogram / bwidth
    else:
        for i in range(-int(3.0 * gwidth / bwidth), int(3.0 * gwidth / bwidth) + 1):
            weight = math.exp(-((i * bwidth) ** 2) / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
            for h in range(max(i, 0), min(len_bins + i - 1, len_bins - 1) + 1):
                dos[h] += histogram[h - i] * weight
    return dos

# Create a histogram from vibrational frequencies
def histogram(freqs, bwidth, len_bins, base):
    """Create a histogram of vibrational frequencies.

    Parameters
    ----------
    freqs : list
        List of vibrational frequencies.
    bwidth : float
        Bin width for the histogram.
    len_bins : int
        Number of bins.
    base : float
        Minimum frequency for binning.

    Returns
    -------
    numpy.ndarray
        Frequency histogram.
    """
    histogram = np.zeros(len_bins)
    for freq in freqs:
        bins = int((freq - base) / bwidth)
        if 0 <= bins < len_bins:
            histogram[bins] += 1.0
    return histogram

# Normalise the DOS based on atom count
def normalise_dos(dos, atom_count, bwidth, molecule_name):
    """Normalize the density of states (DOS) based on the number of vibrational modes (3N).

    Parameters
    ----------
    dos : numpy.ndarray
        Density of states.
    atom_count : int
        Number of atoms in the molecule.
    bwidth : float
        Bin width.
    molecule_name : str
        Name of the molecule.

    Returns
    -------
    numpy.ndarray
        Normalized density of states.
    """
    area = np.trapz(dos, dx=bwidth)
    normalisation_factor = 3 * atom_count
    if area > 0:
        dos *= (normalisation_factor / area)
    return dos

# Generate DOS from vibrational frequencies
def generate_dos(freqs, log_file, atom_count, bwidth, gwidth, filtered_freqs):
    """Generate a normalized density of states (DOS) from vibrational frequencies.

    Parameters
    ----------
    freqs : list
        Vibrational frequencies.
    log_file : str
        Path to the log file.
    atom_count : int
        Number of atoms.
    bwidth : float
        Bin width for histogram.
    gwidth : float
        Gaussian width for broadening.
    filtered_freqs : list
        Filtered frequencies for determining maximum frequency.

    Returns
    -------
    tuple
        - numpy.ndarray: Frequency axis
        - numpy.ndarray: Density of states
        - float: Maximum frequency
    """
    if not filtered_freqs or atom_count is None:
        print(f"Skipping {log_file}, missing data.")
        return None, None, None
    base, max_freq = 0, np.max(filtered_freqs)
    max_freq = max_freq + 10
    len_bins = int((max_freq - base) / bwidth) + 1
    if len_bins <= 0:
        print(f"Error: Invalid number of bins ({len_bins}) for {log_file}")
        return None, None, None
    freq_axis = np.linspace(base, max_freq, len_bins)
    hist = histogram(freqs, bwidth, len_bins, base)
    dos = gaussian_broadening(hist, bwidth=bwidth, gwidth=gwidth)
    dos = normalise_dos(dos, atom_count, bwidth, log_file)
    return freq_axis, dos, max_freq

# Plot the DOS
def plot_dos(freq_axis, dos, filename, dpi, base_dir):
    """Plot the density of states (DOS) and save it to a file.

    Parameters
    ----------
    freq_axis : numpy.ndarray
        Frequency axis for the plot.
    dos : numpy.ndarray
        Density of states.
    filename : str
        Name of the log file.
    dpi : int
        Resolution for the plot.
    base_dir : str
        Base directory for saving plots.
    """
    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.plot(freq_axis, dos, label="", color="#000000")
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    plt.xlim(left=0)
    plt.ylabel("g(ω)", fontsize=24)
    plt.tick_params(axis="both", labelsize=16)
    molecule_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(base_dir, "DOS", f"{molecule_name}_dos.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # print(f"Saved DOS plot to: {save_path}")
    except Exception as e:
        print(f"Error saving DOS plot for {molecule_name}: {e}")
    plt.close()

# Generate box-shaped DOS
def generate_box_dos(freq_axis, dos, omega_max):
    """Create a box-shaped density of states (DOS) up to a maximum frequency.

    Parameters
    ----------
    freq_axis : numpy.ndarray
        Frequency axis.
    dos : numpy.ndarray
        Original density of states.
    omega_max : float
        Maximum frequency for the box.

    Returns
    -------
    tuple
        - numpy.ndarray: Frequency axis
        - numpy.ndarray: Box-shaped density of states
    """
    box_dos = np.copy(dos)
    # box_area = 6
    # intensity = box_area / omega_max
    # box_dos[freq_axis <= omega_max] = intensity
    return freq_axis, box_dos

# Plot box-shaped DOS
def plot_box_dos(filtered_freq_axis, filtered_dos, filename, dpi, base_dir):
    """Plot the box-shaped density of states (DOS) and save it to a file.

    Parameters
    ----------
    filtered_freq_axis : numpy.ndarray
        Frequency axis.
    filtered_dos : numpy.ndarray
        Box-shaped density of states.
    filename : str
        Name of the log file.
    dpi : int
        Resolution for the plot.
    base_dir : str
        Base directory for saving plots.
    """
    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.plot(filtered_freq_axis, filtered_dos, label="")
    plt.xlabel("Wavenumber (cm⁻¹)", fontsize=24)
    plt.xlim(left=0)
    plt.ylabel("g(ω)", fontsize=24)
    plt.tick_params(axis='both', labelsize=18)
    molecule_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(base_dir, "Box_DOS", f"{molecule_name}_box_dos.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # print(f"Saved Box DOS plot to: {save_path}")
    except Exception as e:
        print(f"Error saving Box DOS plot for {molecule_name}: {e}")
    plt.close()

# Apply Bose-Einstein scaling to DOS
def bose_einstein_dos(box_dos, freq_axis):
    """Apply Bose-Einstein scaling to the box-shaped density of states (DOS).

    Parameters
    ----------
    box_dos : numpy.ndarray
        Box-shaped density of states.
    freq_axis : numpy.ndarray
        Frequency axis.

    Returns
    -------
    numpy.ndarray
        Bose-Einstein scaled density of states.
    """
    be_dos = np.copy(box_dos)
    hbar, k, t = scipy.constants.hbar, scipy.constants.k, 300
    for i, w in enumerate(freq_axis):
        if w > 5:
            n = (1.0 / (np.exp(hbar * (w*(2*math.pi) * (1E9 * 29.979245)) / (k * t)) - 1.0))
            be_dos[i] = box_dos[i] * n
    return be_dos

# Plot Bose-Einstein scaled DOS
def plot_be_dos(freq_axis, be_dos, filename, atom_count, dpi, base_dir):
    """Plot the Bose-Einstein scaled density of states (DOS) and save it to a file.

    Parameters
    ----------
    freq_axis : numpy.ndarray
        Frequency axis.
    be_dos : numpy.ndarray
        Bose-Einstein scaled density of states.
    filename : str
        Name of the log file.
    atom_count : int
        Number of atoms.
    dpi : int
        Resolution for the plot.
    base_dir : str
        Base directory for saving plots.
    """
    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.plot(freq_axis, be_dos, label="", color="#000000")
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    plt.xlim(left=0)
    plt.ylabel("g(ω)", fontsize=24)
    plt.tick_params(axis="both", labelsize=16)
    molecule_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(base_dir, "BE_Scaled_DOS", f"{molecule_name}_be_dos.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # print(f"Saved BE Scaled DOS plot to: {save_path}")
    except Exception as e:
        print(f"Error saving BE Scaled DOS plot for {molecule_name}: {e}")
    plt.close()

# Perform first convolution of Bose-Einstein DOS
def first_convolution(freq_axis, be_dos, omega_max, bwidth):
    """Perform the first convolution of the Bose-Einstein density of states (DOS) with itself.

    Parameters
    ----------
    freq_axis : numpy.ndarray
        Frequency axis.
    be_dos : numpy.ndarray
        Bose-Einstein scaled density of states.
    omega_max : float
        Maximum frequency.
    bwidth : float
        Bin width.

    Returns
    -------
    tuple
        - numpy.ndarray: Convolved density of states
        - numpy.ndarray: Frequency axis
        - numpy.ndarray: Unprojected convolved density of states
        - numpy.ndarray: Original Bose-Einstein density of states
    """
    conv_freq_axis = np.linspace(0, 2 * np.max(freq_axis), len(freq_axis) * 2 - 1)
    convolved_raw = np.convolve(be_dos, be_dos, mode="full")
    f_interp = interp1d(conv_freq_axis, convolved_raw, bounds_error=False, fill_value=0)
    first_convolved_dos = f_interp(freq_axis)
    first_conv_no_projection = first_convolved_dos.copy()
    first_convolved_dos[be_dos <= 1E-6] = 0
    return first_convolved_dos, freq_axis, first_conv_no_projection, be_dos

# Plot first convolution
def plot_first_convolution(freq_axis, first_convolved_dos, first_conv_no_proj, be_dos, filename, atom_count, dpi, base_dir):
    """Plot the first convolved density of states (DOS) alongside the Bose-Einstein DOS.

    Parameters
    ----------
    freq_axis : numpy.ndarray
        Frequency axis.
    first_convolved_dos : numpy.ndarray
        First convolved density of states.
    first_conv_no_proj : numpy.ndarray
        Unprojected first convolved density of states.
    be_dos : numpy.ndarray
        Bose-Einstein scaled density of states.
    filename : str
        Name of the log file.
    atom_count : int
        Number of atoms.
    dpi : int
        Resolution for the plot.
    base_dir : str
        Base directory for saving plots.
    """
    fig, ax1 = plt.subplots(figsize=(12, 9), dpi=dpi)
    ax1.plot(freq_axis, first_convolved_dos, label="First Convolution (Projected)", color="#000000")
    ax1.set_ylabel("First Convolution projected onto DOS / g(ω)", fontsize=24)
    ax1.tick_params(axis='y', labelcolor='#000000')
    ax1.tick_params(axis="both", labelsize=16)
    ax1.set_xlim(left=0)
    ax2 = ax1.twinx()
    ax2.plot(freq_axis, be_dos, color='#FFA500', label="BE Scaled DOS", alpha=0.7)
    ax2.set_ylabel("BE Scaled DOS / g(ω)", color='#FFA500', fontsize=24)
    ax2.tick_params(axis='y', labelcolor='#FFA500', labelsize=16)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.grid(False)
    plt.tight_layout()
    molecule_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(base_dir, "First_Convolution", f"{molecule_name}_first_convolution.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # print(f"Saved First Convolution plot to: {save_path}")
    except Exception as e:
        print(f"Error saving First Convolution plot for {molecule_name}: {e}")
    plt.close()

# Perform second convolution
def second_convolution(first_convolved_dos, first_conv_freq_axis, be_dos, original_freq_axis, omega_max):
    """Perform the second convolution of the first convolved DOS with the Bose-Einstein DOS.

    Parameters
    ----------
    first_convolved_dos : numpy.ndarray
        First convolved density of states.
    first_conv_freq_axis : numpy.ndarray
        Frequency axis for first convolution.
    be_dos : numpy.ndarray
        Bose-Einstein scaled density of states.
    original_freq_axis : numpy.ndarray
        Original frequency axis.
    omega_max : float
        Maximum frequency.

    Returns
    -------
    tuple
        - numpy.ndarray: Second convolved density of states
        - numpy.ndarray: Unprojected second convolved density of states
    """
    conv_freq_axis = np.linspace(0, 2*np.max(original_freq_axis), len(original_freq_axis)*2 - 1)
    convolved_raw = np.convolve(first_convolved_dos, be_dos, mode='full')
    f_interp = interp1d(conv_freq_axis, convolved_raw, bounds_error=False, fill_value=0)
    second_convolved_dos = f_interp(original_freq_axis)
    second_conv_no_projection = second_convolved_dos.copy()
    second_convolved_dos[be_dos == 0] = 0
    return second_convolved_dos, second_conv_no_projection

# Plot second convolution
def plot_second_convolution(second_convolved_dos, freq_axis, be_dos, filename, lower_bound, upper_bound, second_conv_no_projection, omega_max, dpi, base_dir):
    """Plot the second convolved density of states (DOS) alongside the Bose-Einstein DOS.

    Parameters
    ----------
    second_convolved_dos : numpy.ndarray
        Second convolved density of states.
    freq_axis : numpy.ndarray
        Frequency axis.
    be_dos : numpy.ndarray
        Bose-Einstein scaled density of states.
    filename : str
        Name of the log file.
    lower_bound : float
        Lower frequency bound for integration.
    upper_bound : float
        Upper frequency bound for integration.
    second_conv_no_projection : numpy.ndarray
        Unprojected second convolved density of states.
    omega_max : float
        Maximum frequency.
    dpi : int
        Resolution for the plot.
    base_dir : str
        Base directory for saving plots.
    """
    fig, ax1 = plt.subplots(figsize=(12, 9), dpi=dpi)
    ax1.plot(freq_axis, second_convolved_dos, color='#000000', alpha=0.7, label="Second Convolution DOS")
    ax1.set_xlabel("Wavenumber /cm⁻¹", fontsize=24)
    ax1.set_ylabel("Second Convolution DOS / g(ω)", color='#000000', fontsize=24)
    ax1.tick_params(axis='y', labelcolor='#000000', labelsize=16)
    ax1.tick_params(axis='x', labelcolor='#000000', labelsize=16)
    ax1.set_xlim(left=0)
    ax2 = ax1.twinx()
    ax2.plot(freq_axis, be_dos, color='#FFA500', label="BE Scaled DOS", alpha=0.7)
    ax2.set_ylabel("BE Scaled DOS / g(ω)", color='#FFA500', fontsize=24)
    ax2.tick_params(axis='y', labelcolor='#FFA500', labelsize=16)
    ax1.grid(True, linestyle='--', alpha=0.5)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=16)
    plt.tight_layout()
    molecule_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(base_dir, "Second_Convolution", f"{molecule_name}_second_convolution.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # print(f"Saved Second Convolution plot to: {save_path}")
    except Exception as e:
        print(f"Error saving Second Convolution plot for {molecule_name}: {e}")
    plt.close()

# Plot normalised integrals vs I50 values
def plot_max_min_normalised(i50_values, integral_values, molecule_omega_max, freq_data, molecule_bounds, dpi, base_dir):
    """Plot normalized integral values vs I50 values with a linear fit on inverse I50.

    Parameters
    ----------
    i50_values : dict
        Dictionary of molecule data including I50 values.
    integral_values : dict
        Dictionary of integral values per molecule.
    molecule_omega_max : dict
        Dictionary of maximum frequencies per molecule.
    freq_data : list
        List of frequency data tuples.
    molecule_bounds : dict
        Dictionary of frequency bounds per molecule.
    dpi : int
        Resolution for the plot.
    base_dir : str
        Base directory for saving plots.

    Returns
    -------
    dict
        Empty dictionary (for compatibility with original code).
    """
    molecule_to_freqs = {}
    for filename, atom_count, frequencies, atoms, normal_modes in freq_data:
        if not frequencies:
            continue
        molecule_name = os.path.basename(filename).split(".")[0]
        if molecule_name not in molecule_to_freqs:
            molecule_to_freqs[molecule_name] = []
        molecule_to_freqs[molecule_name].extend(frequencies)

    freq_counts = {}
    for molecule_name, freqs in molecule_to_freqs.items():
        if molecule_name not in molecule_omega_max or molecule_name not in molecule_bounds:
            continue
        bounds = molecule_bounds[molecule_name]
        lower_bound = bounds['lower']
        upper_bound = bounds['upper']
        count = sum(1 for f in freqs if lower_bound <= f <= upper_bound)
        freq_counts[molecule_name] = count

    i50_to_integral = {name: name for name in i50_values.keys() if name in integral_values}
    sorted_molecule_names = sorted(i50_to_integral.keys(), key=lambda name: i50_values[name][0])

    avg_normalised_integrals = []
    i50_for_plot = []
    omega_max_for_plot = []
    valid_molecule_names = []

    for name in sorted_molecule_names:
        values = np.array(integral_values[name])
        if len(values) == 0 or name not in freq_counts or freq_counts[name] == 0:
            print(f"Skipping {name}: No valid frequency count or integral values")
            continue
        avg_area = np.mean(values)
        normalised_value = avg_area / freq_counts[name]
        avg_normalised_integrals.append(normalised_value)
        i50_for_plot.append(i50_values[name][0])
        omega_max_for_plot.append(molecule_omega_max.get(name, 0))
        valid_molecule_names.append(name)

    if not valid_molecule_names:
        print("No valid molecules for plot")
        return {}

    avg_normalised_integrals = np.array(avg_normalised_integrals)
    i50_for_plot = np.array(i50_for_plot)
    omega_max_for_plot = np.array(omega_max_for_plot)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=dpi)
    ax.plot(
        i50_for_plot,
        avg_normalised_integrals,
        'o',
        markersize=6,
        color='tab:green'
    )

    for x, y, label in zip(i50_for_plot, avg_normalised_integrals, valid_molecule_names):
        number = i50_values[label][1]
        ax.text(x, y, f"{number}", ha='right', va='bottom', rotation=0, fontsize=18)

    inv_i50 = 1 / i50_for_plot
    coeffs = np.polyfit(inv_i50, avg_normalised_integrals, deg=1)
    a, b = coeffs
    predicted = a * inv_i50 + b
    r_squared = r2_score(avg_normalised_integrals, predicted)

    x_fit = np.linspace(min(i50_for_plot), max(i50_for_plot), 300)
    y_fit = a / x_fit + b
    ax.plot(x_fit, y_fit, '--', color='gray', label=f"(R² = {r_squared:.3f})")
    ax.tick_params(axis='both', labelsize=18)

    ax.set_xlabel("I50 Value /Nm", fontsize=24)
    ax.set_ylabel("Up-pumped metric /arb.", fontsize=24)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
    # save_path = os.path.join(base_dir, "Max_Min_Normalised", "max_min_normalised.png")
    # try:
    #     plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    #     # print(f"Saved Max Min Normalised plot to: {save_path}")
    # except Exception as e:
    #     print(f"Error saving Max Min Normalised plot: {e}")
    plt.close()

# Print a summary table of results
def print_summary_table(original_displacement_frequencies, molecule_omega_max, area_by_molecule, i50_values, freq_data, molecule_bounds):
    """Print a summary table of molecule data, including omega_max, bounds, I50, and integrals.

    Parameters
    ----------
    original_displacement_frequencies : dict
        Dictionary of displacement frequencies.
    molecule_omega_max : dict
        Dictionary of maximum frequencies.
    area_by_molecule : dict
        Dictionary of integral areas.
    i50_values : dict
        Dictionary of molecule data including I50 values.
    freq_data : list
        List of frequency data tuples.
    molecule_bounds : dict
        Dictionary of frequency bounds.
    """
    print("\nSummary Table (with molecule-specific bounds):")
    print(f"{'Molecule':<20} {'ω_max (cm⁻¹)':>15} {'Upper Bound (cm⁻¹)':>20} {'i50 (Nm)':>10} {'# Peaks in DOS':>15} {'Area':>20} {'Area/# Peaks':>15}")
    print("-" * 140)

    molecule_to_freqs = {}
    for filename, atom_count, frequencies, atoms, normal_modes in freq_data:
        if not frequencies:
            continue
        molecule_name = os.path.basename(filename).split(".")[0]
        if molecule_name not in molecule_to_freqs:
            molecule_to_freqs[molecule_name] = []
        molecule_to_freqs[molecule_name].extend(frequencies)

    table_data = []
    for molecule in area_by_molecule:
        i50_tuple = i50_values.get(molecule)
        omega_max = molecule_omega_max.get(molecule)
        areas = area_by_molecule[molecule]
        freqs = molecule_to_freqs.get(molecule, [])
        bounds = molecule_bounds.get(molecule)

        if i50_tuple is not None and omega_max is not None and freqs and bounds:
            lower_bound = bounds['lower']
            upper_bound = bounds['upper']
            peak_count = sum(1 for f in freqs if lower_bound <= f <= upper_bound)
            avg_area = np.mean(areas)
            normalised_area = avg_area / peak_count if peak_count > 0 else avg_area
            table_data.append((i50_tuple[0], molecule, omega_max, lower_bound, upper_bound, avg_area, peak_count, normalised_area))

    table_data.sort(key=lambda x: i50_values[x[1]][1])

    for i50, molecule, omega_max, lower_bound, upper_bound, avg_area, peak_count, normalised_area in table_data:
        print(f"{molecule:<20} {omega_max:>15.2f} {upper_bound:>20.2f} {i50:>10} {peak_count:>15} {avg_area:>20.2e} {normalised_area:>15.2e}")

# Calculate Euclidean distance between two coordinates
def distance(coord1, coord2):
    """Compute the Euclidean distance between two 3D coordinates.

    Parameters
    ----------
    coord1 : tuple
        First coordinate (x, y, z).
    coord2 : tuple
        Second coordinate (x, y, z).

    Returns
    -------
    float
        Euclidean distance between the coordinates.
    """
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def unit(v):
    """Normalize a vector to unit length.

    Parameters
    ----------
    v : numpy.ndarray
        Input vector.

    Returns
    -------
    numpy.ndarray
        Normalized vector.

    Raises
    ------
    ValueError
        If the input vector has zero length.
    """
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n

# Define a plane basis from an axis
def plane_basis_from_axis(u):
    """Create an orthonormal basis for a plane perpendicular to the given axis.

    Parameters
    ----------
    u : numpy.ndarray
        Axis vector.

    Returns
    -------
    tuple
        Two orthonormal vectors (e1, e2) spanning the plane.
    """
    u = unit(u)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, u) * u
    e1 = unit(e1)
    e2 = np.cross(u, e1)
    return e1, e2

# Calculate angle in a fixed plane
def angle_in_fixed_plane(point, origin, u_axis, e1, e2):
    """Compute the angle of a point relative to an origin in a plane defined by basis vectors.

    Parameters
    ----------
    point : numpy.ndarray
        Point coordinates.
    origin : numpy.ndarray
        Origin coordinates.
    u_axis : numpy.ndarray
        Axis defining the plane normal.
    e1 : numpy.ndarray
        First basis vector of the plane.
    e2 : numpy.ndarray
        Second basis vector of the plane.

    Returns
    -------
    float
        Angle in radians, or np.nan if the point lies on the axis.
    """
    O = np.asarray(origin, float)
    P = np.asarray(point, float)
    u = unit(u_axis)
    r = P - O
    r_perp = r - np.dot(r, u) * u
    nrm = np.linalg.norm(r_perp)
    if nrm == 0:
        return np.nan
    x = np.dot(r_perp, e1)
    y = np.dot(r_perp, e2)
    return np.arctan2(y, x)

# Wrap angle to [-π, π]
def wrap_to_pi(a):
    """Wrap an angle to the range [-π, π].

    Parameters
    ----------
    a : float
        Input angle in radians.

    Returns
    -------
    float
        Wrapped angle in radians.
    """
    return (a + np.pi) % (2*np.pi) - np.pi

# Identify NO2 groups in the molecule
def identify_no2_groups(atoms, bond_threshold):
    """Identify NO2 groups based on atomic connectivity and bond distance threshold.

    Parameters
    ----------
    atoms : list
        List of atom dictionaries with index, atomic number, and coordinates.
    bond_threshold : float
        Maximum distance for a bond (in Angstroms).

    Returns
    -------
    list
        List of NO2 group dictionaries with anchor, N, O1, and O2 atoms.
    """
    no2_groups = []
    for atom in atoms:
        if atom['Z'] != 7:  # Only consider nitrogen atoms
            continue
        neighbors = [
            other for other in atoms
            if other['index'] != atom['index'] and distance(atom['coords'], other['coords']) < bond_threshold
        ]
        o_neighbors = [nb for nb in neighbors if nb['Z'] == 8]
        non_o_neighbors = [nb for nb in neighbors if nb['Z'] != 8]
        anchor = None
        terminal_os = []
        if len(o_neighbors) == 2 and len(non_o_neighbors) == 1:
            anchor = non_o_neighbors[0]
            terminal_os = o_neighbors
        elif len(o_neighbors) == 3 and len(non_o_neighbors) == 0:
            anchor_candidates = []
            for o in o_neighbors:
                o_neigh = [
                    other for other in atoms
                    if other['index'] != o['index'] and other['index'] != atom['index']
                    and distance(o['coords'], other['coords']) < bond_threshold
                ]
                if len(o_neigh) > 0:
                    anchor_candidates.append(o)  # Corrected: Append to anchor_candidates
            if len(anchor_candidates) == 1:
                anchor = anchor_candidates[0]
                terminal_os = [o for o in o_neighbors if o['index'] != anchor['index']]
        if anchor and len(terminal_os) == 2:
            group = {
                'anchor': anchor,
                'N': atom,
                'O1': terminal_os[0],
                'O2': terminal_os[1]
            }
            no2_groups.append(group)
    return no2_groups

# Assemble atomic masses
def assemble_masses(atoms):
    """Create an array of atomic masses for the given atoms.

    Parameters
    ----------
    atoms : list
        List of atom dictionaries with atomic number (Z).

    Returns
    -------
    numpy.ndarray
        Array of atomic masses in atomic mass units (amu).
    """
    periodic = {
        1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999,
    }
    return np.array([periodic.get(a['Z'], 12.0) for a in atoms], float)

# Extract coordinates from atoms
def coords_from_atoms(atoms):
    """Extract 3D coordinates from a list of atoms.

    Parameters
    ----------
    atoms : list
        List of atom dictionaries with coordinates.

    Returns
    -------
    numpy.ndarray
        Array of coordinates.
    """
    return np.array([a['coords'] for a in atoms], float)

# Update atom coordinates
def apply_coords_to_atoms(atoms, new_coords):
    """Update the coordinates of atoms with new values.

    Parameters
    ----------
    atoms : list
        List of atom dictionaries.
    new_coords : numpy.ndarray
        New coordinates to apply.
    """
    for a, c in zip(atoms, new_coords):
        a['coords'] = tuple(map(float, c))

# Displace coordinates along a vibrational mode
def displace_along_mode(coords, mode_vec, amplitude_angstrom, masses_amu=None, already_mass_weighted=True):
    """Displace atomic coordinates along a vibrational mode.

    Parameters
    ----------
    coords : numpy.ndarray
        Original coordinates.
    mode_vec : numpy.ndarray
        Vibrational mode vector.
    amplitude_angstrom : float
        Displacement amplitude in Angstroms.
    masses_amu : numpy.ndarray, optional
        Atomic masses in atomic mass units.
    already_mass_weighted : bool, optional
        Whether the mode is mass-weighted (default: True).

    Returns
    -------
    numpy.ndarray
        Displaced coordinates.

    Raises
    ------
    ValueError
        If masses_amu is not provided for ase modes or if the mode vector has zero norm.
    """
    mode = np.array(mode_vec, dtype=float)
    if already_mass_weighted:
        if masses_amu is None:
            raise ValueError("masses_amu must be provided when using Gaussian16 modes")
        mode = mode / np.sqrt(masses_amu)[:, None]
    norm = np.linalg.norm(mode)
    if norm == 0.0:
        raise ValueError("Mode vector has zero norm after conversion.")
    mode /= norm
    displaced_coords = coords + amplitude_angstrom * mode
    return displaced_coords

# Calculate NO2 angle changes for a vibrational mode
def no2_angle_changes_for_mode(group, atoms, mode_vec, amplitude_angstrom=1, already_mass_weighted=True):
    """Compute the change in NO₂ angles when displaced along a vibrational mode.

    Parameters
    ----------
    group : dict
        NO₂ group dictionary with anchor, N, O1, and O2 atoms.
    atoms : list
        List of atom dictionaries.
    mode_vec : numpy.ndarray
        Vibrational mode vector.
    amplitude_angstrom : float, optional
        Displacement amplitude in Angstroms (default: 1).
    already_mass_weighted : bool, optional
        Whether the mode is mass-weighted (default: True).

    Returns
    -------
    dict
        Dictionary containing original and displaced angles and their differences.
    """
    coords_opt = coords_from_atoms(atoms)
    masses = assemble_masses(atoms)
    X = np.array(group['anchor']['coords'], float)
    Np = np.array(group['N']['coords'], float)
    u_axis = Np - X
    e1, e2 = plane_basis_from_axis(u_axis)
    O1 = np.array(group['O1']['coords'], float)
    O2 = np.array(group['O2']['coords'], float)
    theta1_opt = angle_in_fixed_plane(O1, origin=Np, u_axis=u_axis, e1=e1, e2=e2)
    theta2_opt = angle_in_fixed_plane(O2, origin=Np, u_axis=u_axis, e1=e1, e2=e2)
    coords_disp = displace_along_mode(
        coords_opt, mode_vec,
        amplitude_angstrom=amplitude_angstrom,
        masses_amu=masses,
        already_mass_weighted=already_mass_weighted
    )
    atoms_disp = [dict(a) for a in atoms]
    apply_coords_to_atoms(atoms_disp, coords_disp)
    Nd = np.array(next(a['coords'] for a in atoms_disp if a['index'] == group['N']['index']), float)
    O1d = np.array(next(a['coords'] for a in atoms_disp if a['index'] == group['O1']['index']), float)
    O2d = np.array(next(a['coords'] for a in atoms_disp if a['index'] == group['O2']['index']), float)
    theta1_disp = angle_in_fixed_plane(O1d, origin=Nd, u_axis=u_axis, e1=e1, e2=e2)
    theta2_disp = angle_in_fixed_plane(O2d, origin=Nd, u_axis=u_axis, e1=e1, e2=e2)
    return {
        'theta_O1_opt': theta1_opt,
        'theta_O2_opt': theta2_opt,
        'theta_O1_disp': theta1_disp,
        'theta_O2_disp': theta2_disp,
        'dtheta_O1': wrap_to_pi(theta1_disp - theta1_opt),
        'dtheta_O2': wrap_to_pi(theta2_disp - theta2_opt)
    }

# Plot NO2 angle changes vs frequency
def plot_no2_angle_vs_frequency(frequencies, normal_modes, atoms, no2_groups,
                                amplitude_angstrom=1,
                                already_mass_weighted=True,
                                molecule_name="molecule", dpi=100,
                                min_angle_deg=0.0, base_dir=None):
    """Plot the maximum NO₂ angle change vs vibrational frequency.

    Parameters
    ----------
    frequencies : list
        Vibrational frequencies.
    normal_modes : list
        Normal mode vectors.
    atoms : list
        List of atom dictionaries.
    no2_groups : list
        List of NO₂ group dictionaries.
    amplitude_angstrom : float, optional
        Displacement amplitude in Angstroms (default: 1).
    already_mass_weighted : bool, optional
        Whether modes are mass-weighted (default: True).
    molecule_name : str, optional
        Name of the molecule (default: "molecule").
    dpi : int, optional
        Resolution for the plot (default: 100).
    min_angle_deg : float, optional
        Minimum angle change for inclusion in degrees (default: 0.0).
    base_dir : str, optional
        Base directory for saving plots.
    """
    angle_changes = []
    freq_list = []
    for i, mode_vec in enumerate(normal_modes):
        if i >= len(frequencies):
            continue
        freq = frequencies[i]
        if freq < 0:
            continue
        mode_max_dtheta = 0
        for group in no2_groups:
            changes = no2_angle_changes_for_mode(
                group, atoms, mode_vec,
                amplitude_angstrom=amplitude_angstrom,
                already_mass_weighted=already_mass_weighted
            )
            dtheta = max(abs(changes['dtheta_O1']), abs(changes['dtheta_O2']))
            if np.isnan(dtheta):
                continue
            if dtheta > mode_max_dtheta:
                mode_max_dtheta = dtheta
        mode_max_dtheta_deg = np.degrees(mode_max_dtheta)
        if mode_max_dtheta_deg >= min_angle_deg:
            freq_list.append(freq)
            angle_changes.append(mode_max_dtheta_deg)
    if not freq_list:
        print(f"No valid angle changes above {min_angle_deg}° found for {molecule_name}")
        return
    plt.figure(figsize=(6, 4), dpi=dpi)
    plt.scatter(freq_list, angle_changes,
                c="tab:blue", alpha=0.7, edgecolors="k")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Δθ$_{NO₂}$ (degrees)")
    plt.title(f"NO₂ Angle Change vs Frequency\n{molecule_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(0, 200)
    plt.ylim(bottom=min_angle_deg * 1)
    plt.tight_layout()
    save_path = os.path.join(base_dir, "NO2_Angle", f"{molecule_name}_no2_angle.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # print(f"Saved NO2 Angle plot to: {save_path}")
    except Exception as e:
        print(f"Error saving NO2 Angle plot for {molecule_name}: {e}")
    plt.close()

# Plot chosen and correct frequency vs NO2 angle
def plot_chosen_and_correct_freq_angle(molecule_name, valid_modes, i50_values, dpi, base_dir):
    """Plot vibrational modes with NO₂ angle changes, highlighting chosen and correct frequencies.

    Parameters
    ----------
    molecule_name : str
        Name of the molecule.
    valid_modes : list
        List of (frequency, angle) tuples.
    i50_values : dict
        Dictionary of molecule data including correct frequencies.
    dpi : int
        Resolution for the plot.
    base_dir : str
        Base directory for saving plots.
    """
    if not valid_modes:
        print(f"No valid modes for {molecule_name}. Skipping plot.")
        return

    correct_freq = i50_values.get(molecule_name, (None, None, None))[2]
    if correct_freq is None:
        print(f"No correct frequency found in i50_values for {molecule_name}. Plotting without correct highlight.")

    freqs, angles = zip(*valid_modes)
    freqs = list(freqs)
    angles = list(angles)

    chosen_freq = max(freqs)
    chosen_idx = freqs.index(chosen_freq)
    chosen_angle = angles[chosen_idx]

    plt.figure(figsize=(8, 6), dpi=dpi)
    plt.scatter(freqs, angles, color='tab:blue', alpha=0.7, label='Valid Modes (>=11°)')
    plt.scatter(chosen_freq, chosen_angle, color='red', s=100, label=f'Chosen (freq={chosen_freq:.2f}, angle={chosen_angle:.2f}°)')

    if correct_freq is not None:
        closest_idx = np.argmin(np.abs(np.array(freqs) - correct_freq))
        closest_freq = freqs[closest_idx]
        closest_angle = angles[closest_idx]
        plt.scatter(closest_freq, closest_angle, color='green', s=100, label=f'Closest to Correct (freq={closest_freq:.2f}, angle={closest_angle:.2f}°)')
        plt.axvline(correct_freq, color='green', linestyle='--', label=f'Correct Freq={correct_freq:.2f}')

    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
    plt.ylabel("Δθ$_{NO₂}$ (degrees)", fontsize=14)
    plt.title(f"Frequency vs NO₂ Angle Change: {molecule_name}", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(base_dir, "NO2_Angle_Chosen_Correct", f"{molecule_name}_chosen_correct_angle.png")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # print(f"Saved Chosen and Correct Angle plot to: {save_path}")
    except Exception as e:
        print(f"Error saving Chosen and Correct Angle plot for {molecule_name}: {e}")
    plt.close()

# Main execution block
if __name__ == "__main__":
    # Configuration parameters
    dpi = 100
    bwidth, gwidth = 0.5, 2.5
    lower_freq = 0
    upper_freq = 110
    bond_threshold = 1.6
    mad = 4

    # Create directory structure for plots
    base_dir = create_plot_directories()
    
    # Enable/disable specific plots
    plot_enabled = {
        'dos': False,
        'box_dos': False,
        'be_dos': False,
        'first_convolution': False,
        'second_convolution': False,
        'no2_angle': False,
        'max_min_normalised': True,
        'chosen_correct_angle': False,
    }

    # Read and parse .txt and .xyz files
    freq_data, subdir = get_file_data_from_txt_xyz(subdir="UPPUMPING/OPTIMISED_STRUCTURES/MEDIUM_MODEL")

    # Initialize dictionaries for storing results
    area_by_molecule = {}
    molecule_bounds = {}
    molecule_omega_max = {}
    original_displacement_frequencies = {}

    # Process each molecule
    for filename, atom_count, frequencies, atoms, normal_modes in freq_data:
        if not frequencies or atom_count is None or not normal_modes:
            print(f"Skipping {filename}, missing frequency, atom count, or normal modes data.")
            continue

        molecule_name = os.path.basename(filename).split(".")[0]

        # Identify NO2 groups
        no2_groups = identify_no2_groups(atoms, bond_threshold)

        # Determine omega_max based on NO2 angle changes
        if not no2_groups:
            print(f"No NO2 groups in {molecule_name}, using default omega_max")
            omega_max = 200
        else:
            min_angle_deg = mad
            valid_modes = []
            for i, mode_vec in enumerate(normal_modes):
                if i >= len(frequencies):
                    print(f"Warning: Mode index {i} exceeds frequency count in {filename}")
                    continue
                freq = frequencies[i]
                if isinstance(freq, complex):
                    freq = freq.real
                if freq < 0:
                    continue
                if not (lower_freq <= freq <= upper_freq):
                    continue
                mode_max_dtheta = 0
                for group in no2_groups:
                    changes = no2_angle_changes_for_mode(
                        group, atoms, mode_vec,
                        amplitude_angstrom=1,
                        already_mass_weighted=True
                    )
                    dtheta = max(abs(changes['dtheta_O1']), abs(changes['dtheta_O2']))
                    if np.isnan(dtheta):
                        continue
                    if dtheta > mode_max_dtheta:
                        mode_max_dtheta = dtheta
                mode_max_dtheta_deg = np.degrees(mode_max_dtheta)
                if mode_max_dtheta_deg >= min_angle_deg:
                    valid_modes.append((freq, mode_max_dtheta_deg))
            if valid_modes:
                omega_max, angle_at_omega_max = max(valid_modes, key=lambda x: x[0])
                print(f"Chosen ω_max = {omega_max:.1f} cm^-1 "
                      f"(Δθ = {angle_at_omega_max:.2f}°) for {molecule_name}")
            else:
                print(f"No NO2 angle change > {min_angle_deg}° in bounds for {molecule_name}, using default ω_max")
                omega_max = 200

        molecule_omega_max[molecule_name] = omega_max
        lower_bound = 1.0 * omega_max
        upper_bound = 4.0 * omega_max
        molecule_bounds[molecule_name] = {'lower': lower_bound, 'upper': upper_bound}

        # Generate and plot DOS-related data
        freq_axis, dos, max_freq = generate_dos(frequencies, filename, atom_count, bwidth, gwidth, frequencies)
        if freq_axis is None:
            print(f"Skipping {filename}, DOS generation failed.")
            continue

        freq_axis_box, box_dos = generate_box_dos(freq_axis, dos, omega_max)
        be_dos = bose_einstein_dos(box_dos, freq_axis)
        first_convolved_dos, freq_axis_fc, first_conv_no_projection, _ = first_convolution(freq_axis, be_dos, omega_max, bwidth)
        second_convolved_dos, second_conv_no_projection = second_convolution(first_convolved_dos, freq_axis_fc, be_dos, freq_axis, omega_max)

        # Generate plots if enabled
        if plot_enabled['dos']:
            plot_dos(freq_axis, dos, filename, dpi, base_dir)
        if plot_enabled['box_dos']:
            plot_box_dos(freq_axis_box, box_dos, filename, dpi, base_dir)
        if plot_enabled['be_dos']:
            plot_be_dos(freq_axis, be_dos, filename, atom_count, dpi, base_dir)
        if plot_enabled['first_convolution']:
            plot_first_convolution(freq_axis_fc, first_convolved_dos, first_conv_no_projection, be_dos, filename, atom_count, dpi, base_dir)
        if plot_enabled['second_convolution']:
            plot_second_convolution(second_convolved_dos, freq_axis, be_dos, filename, lower_bound, upper_bound, second_conv_no_projection, omega_max, dpi, base_dir)
        if plot_enabled['no2_angle'] and no2_groups:
            plot_no2_angle_vs_frequency(frequencies, normal_modes, atoms, no2_groups,
                                        amplitude_angstrom=1,
                                        molecule_name=molecule_name,
                                        dpi=dpi, min_angle_deg=0.0, base_dir=base_dir)


        # Calculate area under the second convolved DOS within bounds
        integration_mask = (freq_axis >= lower_bound) & (freq_axis <= upper_bound)
        if not np.any(integration_mask):
            print(f"Warning: No data points within bounds [{lower_bound}, {upper_bound}] for {molecule_name}")
            area = 0.0
        else:
            area = np.trapz(second_convolved_dos[integration_mask], freq_axis[integration_mask])

        if molecule_name not in area_by_molecule:
            area_by_molecule[molecule_name] = []
        area_by_molecule[molecule_name].append(area)

    # Generate normalised integral plot and summary table
    print_summary_table(original_displacement_frequencies, molecule_omega_max, area_by_molecule, i50_values, freq_data, molecule_bounds)
    if plot_enabled['max_min_normalised']:
        plot_max_min_normalised(i50_values, area_by_molecule, molecule_omega_max, freq_data, molecule_bounds, dpi, base_dir)
