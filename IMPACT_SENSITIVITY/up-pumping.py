import os
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

atomic_masses = {6: 12.01077, 1: 1.00794, 7: 14.0067, 8: 15.9994}  # CHNO
bohr_to_angstrom = 0.529177

i50_values = {
    "TET-1": (2, 1, 76.5324),
    "NP1": (2.5, 2, 66.1026),
    "HNB": (2.75, 3, 83.0775),
    "PETN": (3, 4, 56.2282),
    "CL20": (3, 5, 81.0393),
    "BP-1": (4.5, 6, 78.9906),
    "TET-2": (5, 7, 72.5534),
    "CH-PETN": (6, 8, 45.1597),
    "HMX": (8, 9, 108.3263),
    "TET-3": (8, 10, 86.7160),
    "RDX": (13, 11, 77.8322),
    "TNT-1": (14, 12, 86.9846),
    "PCA": (16, 13, 85.0698),
    "TET-4": (15, 14, 99.7584),
    "BP-2": (20, 15, 92.2816),
    "ADNP-5": (23, 16, 103.7134),
    "TNT": (24.5, 17, 49.0603),
    "DNP-35": (25, 18, 69.3667),
    "DNP-13": (25, 19, 52.6453),
    "TNT-2": (26.8, 20, 89.2217),
    "BP-3": (30, 21, 35.4174),
    "FOX7": (31, 22, 65.2550),
    "DNP-34": (40, 23, 56.9645),
    "TET-5": (40, 24, 96.6349),
    "ADNP-4": (41, 25, 105.6198),
    "nitrotriazolone": (73, 26, 43.2361),
    "nitroguanidine": (80, 27, 108.1432),
    "TET-6": (100, 28, 87.0307),
    "TATB": (120, 29, 84.7751),
}

sorted_i50 = dict(sorted(i50_values.items(), key=lambda item: item[1]))

def create_plot_directories(script_name=None):
    if script_name is None:
        script_name = os.path.splitext(os.path.basename(__file__))[0]
    
    figures_dir = os.path.join(os.getcwd(), "FIGURES")
    base_dir = os.path.join(figures_dir, f"{script_name}")
    
    subdirs = [
        "DOS", "Box_DOS", "BE_Scaled_DOS", "First_Convolution",
        "Second_Convolution", "NO2_Angle", "Max_Min_Normalised"
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

def read_log_files():
    subdir = os.path.join(os.getcwd(), "PM7")
    log_files = []
    if os.path.exists(subdir):
        for root, subdirs, files in os.walk(subdir):
            for f in files:
                if f.endswith(".log"):
                    log_files.append(os.path.join(root, f))
    else:
        print(f"Error: Directory {subdir} does not exist")
    return log_files, subdir

def get_file_data(log_files):
    file_data = []
    for log_file in log_files:
        fname = os.path.basename(log_file)
        with open(log_file, "r") as f:
            lines = f.readlines()
        frequencies = []
        normal_modes = []
        atoms = None
        atom_count = None
        std_indices = [i for i, l in enumerate(lines) if "Standard orientation:" in l]
        if std_indices:
            last_std = std_indices[-1]
            atom_start = last_std + 5
            atoms = []
            for line in lines[atom_start:]:
                if "----" in line: break
                parts = line.split()
                if len(parts) < 6: continue
                try:
                    center_num = int(parts[0])
                    atomic_num = int(parts[1])
                    x, y, z = map(float, parts[3:6])
                    atoms.append({'index': center_num, 'Z': atomic_num, 'coords': (x, y, z)})
                except (ValueError, IndexError) as e:
                    print(f"Error parsing atom coordinates in {fname} at line {line}: {e}")
                    continue
            atom_count = len(atoms)
        if atom_count is None:
            print(f"Error: No atoms found in {fname}")
            continue
        i = 0
        while i < len(lines):
            if "Frequencies --" in lines[i]:
                freq_parts = lines[i].split()[2:]
                num_modes_this_block = len(freq_parts)
                try:
                    frequencies.extend(float(f) for f in freq_parts if float(f) > 0)
                except ValueError as e:
                    print(f"Error parsing frequencies in {fname}: {e}")
                    i += 1
                    continue
                j = i + 1
                while j < len(lines) and not ("Atom  AN" in lines[j] or "AN" in lines[j]):
                    j += 1
                if j >= len(lines):
                    print(f"Warning: No eigenvector data found after frequencies in {fname}")
                    i += 1
                    continue
                j += 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                vectors = [np.zeros((atom_count, 3)) for _ in range(num_modes_this_block)]
                atom_idx = 0
                expected_columns = 2 + 3 * num_modes_this_block
                while j < len(lines) and atom_idx < atom_count:
                    line = lines[j].strip()
                    if not line or "----" in line:
                        j += 1
                        continue
                    parts = line.split()
                    if len(parts) < expected_columns:
                        print(f"Error: Expected {expected_columns} columns, got {len(parts)} in {fname} at line {j+1}: {line}")
                        break
                    try:
                        atom_num = int(parts[0]) - 1
                        if atom_num != atom_idx:
                            print(f"Warning: Atom index mismatch in {fname}, expected {atom_idx+1}, got {atom_num+1}")
                        for m in range(num_modes_this_block):
                            offset = 2 + m * 3
                            dx = float(parts[offset])
                            dy = float(parts[offset + 1])
                            dz = float(parts[offset + 2])
                            vectors[m][atom_idx] = [dx, dy, dz]
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing normal mode in {fname} at line {j+1}: {e}")
                        break
                    atom_idx += 1
                    j += 1
                if atom_idx != atom_count:
                    print(f"Error: Parsed {atom_idx} atoms, expected {atom_count} in {fname}")
                    vectors = []
                normal_modes.extend(vectors)
                i = j
            else:
                i += 1
        if len(frequencies) != len(normal_modes):
            print(f"Error: freqs {len(frequencies)} modes {len(normal_modes)} in {fname}")
            continue
        if frequencies:
            file_data.append((fname, atom_count, frequencies, atoms, normal_modes))
        else:
            print(f"Error: No valid frequencies found in {fname}")
    return file_data

def gaussian_broadening(histogram, bwidth=0.5, gwidth=2.5):
    len_bins = len(histogram)
    dos = np.zeros(len_bins)
    sigma = gwidth / 2.354
    if gwidth < bwidth:
        dos[:] = histogram / bwidth
    else:
        for i in range(-int(3.0 * gwidth / bwidth), int(3.0 * gwidth / bwidth) + 1):
            weight = math.exp(-((i * bwidth) ** 2) / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
            for h in range(max(i, 0), min(len_bins + i - 1, len_bins - 1) + 1):
                dos[h] += histogram[h - i] * weight
    return dos

def integrate_region(dos, min=0, max=1000):
    region = [dos[min:max]]
    print(np.trapz(dos[880:910]))
    return

def histogram(freqs, bwidth, len_bins, base):
    histogram = np.zeros(len_bins)
    for freq in freqs:
        bins = int((freq - base) / bwidth)
        if 0 <= bins < len_bins:
            histogram[bins] += 1.0
    return histogram

def normalise_dos(dos, atom_count, bwidth, molecule_name):
    area = np.trapz(dos, dx=bwidth)
    normalisation_factor = 3 * atom_count
    if area > 0:
        dos *= (normalisation_factor / area)
    return dos

def generate_dos(freqs, log_file, atom_count, bwidth, gwidth, filtered_freqs):
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

def plot_dos(freq_axis, dos, filename, atom_count, dpi, base_dir):
    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.plot(freq_axis, dos, label="", color="#000000")
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    plt.xlim(left=0)
    plt.ylabel("g(ω)", fontsize=24)
    plt.tick_params(axis="both", labelsize=16)
    # plt.show()
    molecule_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(base_dir, "DOS", f"{molecule_name}_dos.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved DOS plot to: {save_path}")
    except Exception as e:
        print(f"Error saving DOS plot for {molecule_name}: {e}")
    plt.close()


def generate_box_dos(freq_axis, dos, omega_max):
    box_dos = np.copy(dos)
    box_area = 6
    intensity = box_area / omega_max
    box_dos[freq_axis <= omega_max] = intensity
    return freq_axis, box_dos

def plot_box_dos(filtered_freq_axis, filtered_dos, filename, dpi, base_dir):
    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.plot(filtered_freq_axis, filtered_dos, label="")
    plt.xlabel("Wavenumber (cm⁻¹)", fontsize=24)
    plt.xlim(left=0)
    plt.ylabel("g(ω)", fontsize=24)
    plt.tick_params(axis='both', labelsize=18)
    # plt.show()
    molecule_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(base_dir, "Box_DOS", f"{molecule_name}_box_dos.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved Box DOS plot to: {save_path}")
    except Exception as e:
        print(f"Error saving Box DOS plot for {molecule_name}: {e}")
    plt.close()

def bose_einstein_dos(box_dos, freq_axis):
    be_dos = np.copy(box_dos)
    hbar, k, t = scipy.constants.hbar, scipy.constants.k, 300
    for i, w in enumerate(freq_axis):
        if w > 5:
            n = (1.0 / (np.exp(hbar * (w*(2*math.pi) * (1E9 * 29.979245)) / (k * t)) - 1.0))
            be_dos[i] = box_dos[i] * n
    return be_dos

def plot_be_dos(freq_axis, be_dos, filename, atom_count, dpi, base_dir):
    plt.figure(figsize=(12, 9), dpi=dpi)
    plt.plot(freq_axis, be_dos, label="", color="#000000")
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=24)
    plt.xlim(left=0)
    plt.ylabel("g(ω)", fontsize=24)
    plt.tick_params(axis="both", labelsize=16)
    # plt.show()
    molecule_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(base_dir, "BE_Scaled_DOS", f"{molecule_name}_be_dos.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved BE Scaled DOS plot to: {save_path}")
    except Exception as e:
        print(f"Error saving BE Scaled DOS plot for {molecule_name}: {e}")
    plt.close()

def first_convolution(freq_axis, be_dos, omega_max, bwidth):
    conv_freq_axis = np.linspace(0, 2 * np.max(freq_axis), len(freq_axis) * 2 - 1)
    convolved_raw = np.convolve(be_dos, be_dos, mode="full")
    f_interp = interp1d(conv_freq_axis, convolved_raw, bounds_error=False, fill_value=0)
    first_convolved_dos = f_interp(freq_axis)
    first_conv_no_projection = first_convolved_dos.copy()
    first_convolved_dos[be_dos <= 1E-6] = 0
    return first_convolved_dos, freq_axis, first_conv_no_projection, be_dos

def plot_first_convolution(freq_axis, first_convolved_dos, be_dos, filename, atom_count, dpi, base_dir):
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
    # plt.show()
    molecule_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(base_dir, "First_Convolution", f"{molecule_name}_first_convolution.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved First Convolution plot to: {save_path}")
    except Exception as e:
        print(f"Error saving First Convolution plot for {molecule_name}: {e}")
    plt.close()

def second_convolution(first_convolved_dos, first_conv_freq_axis, be_dos, original_freq_axis, omega_max):
    conv_freq_axis = np.linspace(0, 2*np.max(original_freq_axis), len(original_freq_axis)*2 - 1)
    convolved_raw = np.convolve(first_convolved_dos, be_dos, mode='full')
    f_interp = interp1d(conv_freq_axis, convolved_raw, bounds_error=False, fill_value=0)
    second_convolved_dos = f_interp(original_freq_axis)
    second_conv_no_projection = second_convolved_dos.copy()
    second_convolved_dos[be_dos == 0] = 0
    return second_convolved_dos, second_conv_no_projection

def plot_second_convolution(second_convolved_dos, freq_axis, be_dos, filename, lower_bound, upper_bound, second_conv_no_projection, omega_max, dpi, base_dir):
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
    # plt.show()
    molecule_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(base_dir, "Second_Convolution", f"{molecule_name}_second_convolution.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved Second Convolution plot to: {save_path}")
    except Exception as e:
        print(f"Error saving Second Convolution plot for {molecule_name}: {e}")
    plt.close()

def plot_max_min_normalised(i50_values, integral_values, molecule_omega_max, freq_data, molecule_bounds, dpi, base_dir):
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

    avg_normalized_integrals = []
    i50_for_plot = []
    omega_max_for_plot = []
    valid_molecule_names = []

    for name in sorted_molecule_names:
        values = np.array(integral_values[name])
        if len(values) == 0 or name not in freq_counts or freq_counts[name] == 0:
            print(f"Skipping {name}: No valid frequency count or integral values")
            continue
        avg_area = np.mean(values)
        normalized_value = avg_area / freq_counts[name]
        avg_normalized_integrals.append(normalized_value)
        i50_for_plot.append(i50_values[name][0])
        omega_max_for_plot.append(molecule_omega_max.get(name, 0))
        valid_molecule_names.append(name)

    if not valid_molecule_names:
        print("No valid molecules for plot")
        return {}

    avg_normalized_integrals = np.array(avg_normalized_integrals)
    i50_for_plot = np.array(i50_for_plot)
    omega_max_for_plot = np.array(omega_max_for_plot)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=dpi)
    ax.plot(
        i50_for_plot,
        avg_normalized_integrals,
        'o',
        markersize=6,
        color='tab:green'
    )

    for x, y, label in zip(i50_for_plot, avg_normalized_integrals, valid_molecule_names):
        number = i50_values[label][1]
        ax.text(x, y, f"{number}", ha='right', va='bottom', rotation=0, fontsize=18)

    inv_i50 = 1 / i50_for_plot
    coeffs = np.polyfit(inv_i50, avg_normalized_integrals, deg=1)
    a, b = coeffs
    predicted = a * inv_i50 + b
    r_squared = r2_score(avg_normalized_integrals, predicted)

    x_fit = np.linspace(min(i50_for_plot), max(i50_for_plot), 300)
    y_fit = a / x_fit + b
    ax.plot(x_fit, y_fit, '--', color='gray', label=f"(R² = {r_squared:.3f})")
    ax.tick_params(axis='both', labelsize=18)

    ax.set_xlabel("I50 Value /Nm", fontsize=24)
    ax.set_ylabel("Up-pumped metric /arb.", fontsize=24)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    # plt.show()
    save_path = os.path.join(base_dir, "Max_Min_Normalised", "max_min_normalised.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved Max Min Normalised plot to: {save_path}")
    except Exception as e:
        print(f"Error saving Max Min Normalised plot: {e}")
    plt.close()

def print_summary_table(original_displacement_frequencies, molecule_omega_max, area_by_molecule, i50_values, freq_data, molecule_bounds):
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
            normalized_area = avg_area / peak_count if peak_count > 0 else avg_area
            table_data.append((i50_tuple[0], molecule, omega_max, lower_bound, upper_bound, avg_area, peak_count, normalized_area))

    table_data.sort(key=lambda x: i50_values[x[1]][1])

    for i50, molecule, omega_max, lower_bound, upper_bound, avg_area, peak_count, normalized_area in table_data:
        print(f"{molecule:<20} {omega_max:>15.2f} {upper_bound:>20.2f} {i50:>10} {peak_count:>15} {avg_area:>20.2e} {normalized_area:>15.2e}")

def distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n

def plane_basis_from_axis(u):
    u = unit(u)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, u) * u
    e1 = unit(e1)
    e2 = np.cross(u, e1)
    return e1, e2

def angle_in_fixed_plane(point, origin, u_axis, e1, e2):
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

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def identify_no2_groups(atoms, bond_threshold):
    no2_groups = []
    for atom in atoms:
        if atom['Z'] != 7:
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
                    anchor_candidates.append(o)
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

def assemble_masses(atoms):
    periodic = {
        1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999,
    }
    return np.array([periodic.get(a['Z'], 12.0) for a in atoms], float)

def coords_from_atoms(atoms):
    return np.array([a['coords'] for a in atoms], float)

def apply_coords_to_atoms(atoms, new_coords):
    for a, c in zip(atoms, new_coords):
        a['coords'] = tuple(map(float, c))

def displace_along_mode(coords, mode_vec, amplitude_angstrom, masses_amu=None, already_mass_weighted=False):
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

def no2_angle_changes_for_mode(group, atoms, mode_vec, amplitude_angstrom=1, already_mass_weighted=False):
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

def plot_no2_angle_vs_frequency(frequencies, normal_modes, atoms, no2_groups,
                                amplitude_angstrom=1,
                                already_mass_weighted=False,
                                molecule_name="molecule", dpi=100,
                                min_angle_deg=0.0, base_dir=None):
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
    # plt.show()
    save_path = os.path.join(base_dir, "NO2_Angle", f"{molecule_name}_no2_angle.png")
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved NO2 Angle plot to: {save_path}")
    except Exception as e:
        print(f"Error saving NO2 Angle plot for {molecule_name}: {e}")
    plt.close()

def plot_chosen_and_correct_freq_angle(molecule_name, valid_modes, i50_values, dpi, base_dir):
    if not valid_modes:
        print(f"No valid modes for {molecule_name}. Skipping plot.")
        return

    correct_freq = i50_values.get(molecule_name, (None, None, None))[2]
    if correct_freq is None:
        print(f"No correct frequency found in i50_values for {molecule_name}. Plotting without correct highlight.")

    freqs, angles = zip(*valid_modes)
    freqs = list(freqs)
    angles = list(angles)

    # Chosen: max frequency
    chosen_freq = max(freqs)
    chosen_idx = freqs.index(chosen_freq)
    chosen_angle = angles[chosen_idx]

    plt.figure(figsize=(8, 6), dpi=dpi)
    plt.scatter(freqs, angles, color='tab:blue', alpha=0.7, label='Valid Modes (>=11°)')
    plt.scatter(chosen_freq, chosen_angle, color='red', s=100, label=f'Chosen (freq={chosen_freq:.2f}, angle={chosen_angle:.2f}°)')

    if correct_freq is not None:
        # Find closest mode to correct_freq
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
    # plt.show()

    save_path = os.path.join(base_dir, "NO2_Angle_Chosen_Correct", f"{molecule_name}_chosen_correct_angle.png")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved Chosen and Correct Angle plot to: {save_path}")
    except Exception as e:
        print(f"Error saving Chosen and Correct Angle plot for {molecule_name}: {e}")
    plt.close()


if __name__ == "__main__":
    dpi = 100
    bwidth, gwidth = 0.5, 2.5
    lower_freq = 0
    upper_freq = 110
    bond_threshold = 1.6
    mad = 5

    base_dir = create_plot_directories()
    plot_enabled = {
        'dos': True,
        'box_dos': True,
        'be_dos': True,
        'first_convolution': True,
        'second_convolution': True,
        'no2_angle': True,
        'max_min_normalised': True,
        'chosen_correct_angle': True
    }

    # Read log files and get frequency data
    log_files, subdir = read_log_files()
    freq_data = get_file_data(log_files)

    # Initialize dictionaries
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

        # Calculate omega_max
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
                if freq < 0:
                    continue
                if not (lower_freq <= freq <= upper_freq):
                    continue
                mode_max_dtheta = 0
                for group in no2_groups:
                    changes = no2_angle_changes_for_mode(
                        group, atoms, mode_vec,
                        amplitude_angstrom=1,
                        already_mass_weighted=False
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
        upper_bound = 3.0 * omega_max
        molecule_bounds[molecule_name] = {'lower': lower_bound, 'upper': upper_bound}

        # Generate DOS
        freq_axis, dos, max_freq = generate_dos(frequencies, filename, atom_count, bwidth, gwidth, frequencies)
        if freq_axis is None:
            print(f"Skipping {filename}, DOS generation failed.")
            continue

        freq_axis_box, box_dos = generate_box_dos(freq_axis, dos, omega_max)
        be_dos = bose_einstein_dos(box_dos, freq_axis)
        first_convolved_dos, freq_axis_fc, first_conv_no_projection, _ = first_convolution(freq_axis, be_dos, omega_max, bwidth)
        second_convolved_dos, second_conv_no_projection = second_convolution(first_convolved_dos, freq_axis_fc, be_dos, freq_axis, omega_max)

        if plot_enabled['dos']:
            plot_dos(freq_axis, dos, filename, atom_count, dpi, base_dir)
        if plot_enabled['box_dos']:
            plot_box_dos(freq_axis_box, box_dos, filename, dpi, base_dir)
        if plot_enabled['be_dos']:
            plot_be_dos(freq_axis, be_dos, filename, atom_count, dpi, base_dir)
        if plot_enabled['first_convolution']:
            plot_first_convolution(freq_axis_fc, first_convolved_dos, be_dos, filename, atom_count, dpi, base_dir)
        if plot_enabled['second_convolution']:
            plot_second_convolution(second_convolved_dos, freq_axis, be_dos, filename, lower_bound, upper_bound, second_conv_no_projection, omega_max, dpi, base_dir)
        if plot_enabled['no2_angle'] and no2_groups:
            plot_no2_angle_vs_frequency(frequencies, normal_modes, atoms, no2_groups,
                                        amplitude_angstrom=1,
                                        molecule_name=molecule_name,
                                        dpi=dpi, min_angle_deg=0.0, base_dir=base_dir)
        if plot_enabled['chosen_correct_angle'] and no2_groups and valid_modes:
            plot_chosen_and_correct_freq_angle(molecule_name, valid_modes, i50_values, dpi, base_dir)

        # Calculate area under the curve within bounds
        integration_mask = (freq_axis >= lower_bound) & (freq_axis <= upper_bound)
        if not np.any(integration_mask):
            print(f"Warning: No data points within bounds [{lower_bound}, {upper_bound}] for {molecule_name}")
            area = 0.0
        else:
            area = np.trapz(second_convolved_dos[integration_mask], freq_axis[integration_mask])

        if molecule_name not in area_by_molecule:
            area_by_molecule[molecule_name] = []
        area_by_molecule[molecule_name].append(area)

    # --- Modified: Call max_min_normalised with base_dir ---
    if plot_enabled['max_min_normalised']:
        plot_max_min_normalised(i50_values, area_by_molecule, molecule_omega_max, freq_data, molecule_bounds, dpi, base_dir)
    print_summary_table(original_displacement_frequencies, molecule_omega_max, area_by_molecule, i50_values, freq_data, molecule_bounds)
