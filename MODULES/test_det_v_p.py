"""
Calculate density and Kamlet-Jacobs detonation properties.

This script reads a CSV containing molecule IDs, SMILES strings, and enthalpies
of formation. For each molecule, it reads the corresponding optimised XYZ
structure, predicts crystal density using a trained density model, calculates
Kamlet-Jacobs detonation velocity and pressure, optionally writes density
scaling plots, and appends the results back to the input CSV.
"""
import os
import csv
import re as _re
import joblib
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import Counter
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Lipinski,
    Crippen,
    rdMolDescriptors,
)

# ARGPARSE

def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the input CSV path, optimised
        structure directory name, and plot-control option.
    """
    parser = argparse.ArgumentParser(
        description="Calculate density and Kamlet-Jacobs detonation properties"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input CSV file containing CID, SMILES, and Hf /kJmol-1 columns"
    )

    parser.add_argument(
        "--outdir", "-dir",
        type=str,
        default="HADES",
        help=(
            "Name of subdirectory inside OPTIMISED_STRUCTURES containing "
            "optimised molecule folders. Default: HADES"
        )
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not create density scaling plots"
    )

    return parser.parse_args()

def find_id_column(headers):
    """Find molecule ID column using preferred fallback order."""
    id_column_priority = ["CID", "FILENAME", "molecule", "MOLECULE"]

    for col in id_column_priority:
        if col in headers:
            return col

    raise ValueError(
        "[ERROR] No molecule ID column found. "
        f"Expected one of {id_column_priority}. "
        f"Available columns: {headers}"
    )


def read_csv(csv_path):
    """Read molecule data from the input CSV."""
    data = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"[ERROR] No headers found in CSV: {csv_path}")

        # Strip whitespace from headers
        reader.fieldnames = [
            h.strip() if h is not None else h
            for h in reader.fieldnames
        ]

        headers = reader.fieldnames
        id_column = find_id_column(headers)

        required_cols = ["SMILES", "Hf /kJmol-1"]
        for col in required_cols:
            if col not in headers:
                raise ValueError(
                    f"[ERROR] Required column '{col}' not found in CSV. "
                    f"Available columns: {headers}"
                )

        print(f"Using molecule ID column: {id_column}")

        for row in reader:
            mol_id = row.get(id_column, "").strip()
            eof_val = row.get("Hf /kJmol-1", "").strip()

            if not mol_id:
                print(f"[WARNING] Missing molecule ID in column {id_column}")
                continue

            if not eof_val:
                print(f"[WARNING] Missing EOF for {id_column} {mol_id}")
                continue

            mol = {
                "smiles": row["SMILES"].strip(),
                "eof": float(eof_val),
                "cid": mol_id,          # keep this key so the rest of the script works
                "id_column": id_column, # useful for writing/debugging
            }

            data.append(mol)

    return data


# LOAD DENSITY MODEL
def load_density_model(model_path):
    """Load the trained density model from disk.

    The model file may either contain a dictionary with a fitted model and
    feature names, or a fitted model object directly.

    Parameters
    ----------
    model_path : str
        Path to the saved joblib density model.

    Returns
    -------
    tuple[object, list[str] or None]
        Fitted density model and optional feature-name list.
    """
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        model = bundle["model"]
        feature_names = bundle.get("feature_names")

        print("\nLoaded density model")
        print("Model type:", type(model))

        return model, feature_names

    # fallback if you ever save only the model directly
    print("\nLoaded density model")
    print("Model type:", type(bundle))

    return bundle, None


# BUILD DENSITY MODEL FEATURES
def build_density_model_features(smiles, base_density):
    """
    Build the same feature vector used during training.

    Feature order must match:

        [
            "base_density",
            "base_density_sq",
            "n_N",
            "n_O",
            "n_C",
            "n_heavy",
            "n_heavy_sq",
            "N_fraction",
            "N_fraction_sq",
            "O_fraction",
            "has_aromatic",
            "has_nitro",
            "ring_count",
            "TPSA",
            "MolLogP",
            "MolMR",
            "HeavyAtomCount",
            "NumHDonors",
            "NumHAcceptors",
        ]
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)

    # Same hand-crafted features as training script
    n_N = len(_re.findall(r'N(?![a-z])', smiles))
    n_O = len(_re.findall(r'O(?![a-z])', smiles))
    n_C = len(_re.findall(r'C(?![a-z])', smiles))

    n_heavy = n_N + n_O + n_C

    N_frac = n_N / max(n_heavy, 1)
    O_frac = n_O / max(n_heavy, 1)

    has_aromatic = int("c" in smiles)

    # This only detects nitro written as N(=O)=O.
    has_nitro = int(
        bool(_re.search(r'N\(=O\)=O', smiles))
    )

    ring_count = (
        smiles.count("1") +
        smiles.count("2") +
        smiles.count("3")
    )

    # Same selected RDKit descriptors as training script
    TPSA = rdMolDescriptors.CalcTPSA(mol_h)
    MolLogP = Crippen.MolLogP(mol_h)
    MolMR = Crippen.MolMR(mol_h)
    HeavyAtomCount = Lipinski.HeavyAtomCount(mol_h)
    NumHDonors = Lipinski.NumHDonors(mol_h)
    NumHAcceptors = Lipinski.NumHAcceptors(mol_h)

    X = np.array([[
        base_density,
        base_density ** 2,

        n_N,
        n_O,
        n_C,

        n_heavy,
        n_heavy ** 2,

        N_frac,
        N_frac ** 2,

        O_frac,

        has_aromatic,
        has_nitro,

        ring_count,

        TPSA,
        MolLogP,
        MolMR,
        HeavyAtomCount,
        NumHDonors,
        NumHAcceptors,
    ]], dtype=float)

    return X


# CALCULATE DENSITY USING ML MODEL
def calc_density(
    xyz_path,
    data,
    density_model,
    density_feature_names=None,
):
    """Calculate molecular density using an optimised XYZ and ML model.

        The function calculates molecular weight from SMILES, molecular volume from
        the optimised XYZ structure, forms the trained model feature vector, predicts
        density, and stores density-related values in the molecule dictionary.

        Parameters
        ----------
        xyz_path : str
            Path to the optimised XYZ file.
        data : dict
            Molecule dictionary containing at least `smiles` and `cid`.
        density_model : object
            Fitted model with a `.predict()` method.
        density_feature_names : list[str] or None, optional
            Optional list of expected model feature names.

        Returns
        -------
        dict or None
            Updated molecule dictionary if successful. Returns None if parsing,
            volume calculation, feature generation, or prediction fails.
        """
    smiles = data.get("smiles")
    cid = data.get("cid")

    # SMILES → molecular weight
    mol_smiles = Chem.MolFromSmiles(smiles)

    if mol_smiles is None:
        print(f"[ERROR] Invalid SMILES for {cid}")
        return None

    mol_smiles_h = Chem.AddHs(mol_smiles)
    mw = Descriptors.MolWt(mol_smiles_h)

    # XYZ → molecular volume
    mol_xyz = Chem.MolFromXYZFile(xyz_path)

    if mol_xyz is None:
        print(f"[ERROR] Failed to read {xyz_path}")
        return None

    try:
        Chem.SanitizeMol(mol_xyz)
    except Exception as e:
        print(f"[ERROR] Failed to sanitise {xyz_path}: {e}")
        return None

    try:
        volume = AllChem.ComputeMolVolume(mol_xyz)
    except Exception as e:
        print(f"[ERROR] Failed to compute volume for {cid}: {e}")
        return None

    if volume is None or volume <= 0:
        print(f"[ERROR] Invalid volume for {cid}")
        return None

    # Base molecular density
    base_density = mw / volume

    # Build ML feature array
    X = build_density_model_features(
        smiles,
        base_density,
    )

    if X is None:
        print(f"[ERROR] Could not build density features for {cid}")
        return None

    # check against saved feature names
    if density_feature_names is not None:
        if X.shape[1] != len(density_feature_names):
            print(
                f"[ERROR] Feature mismatch for {cid}: "
                f"X has {X.shape[1]} features, "
                f"model expects {len(density_feature_names)}"
            )
            return None

    # Predict density
    try:
        density = float(density_model.predict(X)[0])
    except Exception as e:
        print(f"[ERROR] Density prediction failed for {cid}: {e}")
        return None

    # Store results
    data["density"] = density
    data["base_density"] = base_density
    data["mw"] = mw
    data["volume"] = volume

    return data


# CALCULATE GAS PRODUCTS
def calc_gas_products(data):
    """Calculate detonation products and heat-release parameters.

    Product stoichiometry follows the oxygen-priority scheme used in the script:
    hydrogen is oxidised to water first, remaining oxygen forms carbon dioxide,
    and any remaining carbon is assigned as solid carbon. If oxygen is
    insufficient for all hydrogen, excess hydrogen is assigned as hydrogen gas.

    Parameters
    ----------
    data : dict
        Molecule dictionary containing SMILES, molecular weight, CID, and
        enthalpy of formation.

    Returns
    -------
    dict
        Updated molecule dictionary containing gas products, gas moles per gram,
        average gas molecular mass, reaction product enthalpy, and Q.
    """
    smiles = data.get("smiles")

    if smiles is None:
        print(f"[ERROR] No SMILES for {data.get('cid')}")
        return data

    mw = data.get("mw")

    if mw is None:
        print(f"[ERROR] No molecular weight for {data.get('cid')}")
        return data

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print(f"[ERROR] Invalid SMILES for {data.get('cid')}")
        return data

    mol = Chem.AddHs(mol)

    counts = Counter(
        atom.GetSymbol()
        for atom in mol.GetAtoms()
    )

    C = counts.get("C", 0)
    H = counts.get("H", 0)
    N = counts.get("N", 0)
    O = counts.get("O", 0)

    """
    Original KJ paper gives the product formation as:

    Ca Hb Nc Od -->
        c/2 N2
        + b/2 H2O
        + (d/2 - b/4) CO2
        + (a - d/2 + b/4) C
    """

    O_required_for_H2O = H / 2

    if O >= O_required_for_H2O:
        H2O = H / 2
        O_remaining = O - H2O
        CO2 = O_remaining / 2
        C_solid = C - CO2
        H2 = 0
    else:
        H2O = O
        CO2 = 0
        C_solid = C
        H_excess = H - (2 * H2O)
        H2 = H_excess / 2

    products = {
        "N2": N / 2,
        "H2O": H2O,
        "CO2": CO2,
        "C": C_solid,
        "H2": H2,
    }

    gas_prods = [
        "CO2",
        "N2",
        "H2O",
        "H2",
    ]

    gas_prods_masses = {
        "CO2": 44,
        "N2": 28,
        "H2O": 18,
        "H2": 2,
    }

    gas_eofs = {
        "CO2": -393.477,
        "N2": 0,
        "H2O": -241.808,
        "H2": 0,
    }

    total_gas = sum(
        products.get(species, 0)
        for species in gas_prods
    )

    if total_gas == 0:
        print(f"[ERROR] No gas products for {data.get('cid')}")
        return data

    eod = sum(
        products.get(species, 0) * gas_eofs[species]
        for species in gas_prods
    )

    gas_mols_per_g = total_gas / mw

    average_mass = (
        sum(
            products.get(species, 0) *
            gas_prods_masses[species]
            for species in gas_prods
        )
        / total_gas
    )

    """
    q = chemical energy of detonation reaction.

    Original KJ-style expression used here:

        Q = -[Hf(products) - Hf(explosive)] / formula weight

    Your original code keeps the final *1000 scaling, so this
    is preserved here.
    """

    q = ((-(eod - data.get("eof")) / 4.184) / mw) * 1000

    data["Q"] = q
    data["M"] = average_mass
    data["N"] = gas_mols_per_g
    data["eod"] = eod
    data["products"] = products
    data["mw"] = mw

    return data


# CALCULATE PHI
def calc_phi(data):
    """Calculate the Kamlet-Jacobs phi term.

    Phi is defined as:

    `phi = N * sqrt(M) * sqrt(Q)`

    where `N` is moles of gaseous product per gram of explosive, `M` is average
    molecular mass of gaseous products, and `Q` is the chemical energy of
    detonation.

    Parameters
    ----------
    data : dict
        Molecule dictionary containing `N`, `M`, `Q`, and `cid`.

    Returns
    -------
    dict
        Updated molecule dictionary containing `phi`.
    """

    N = data.get("N")
    M = data.get("M")
    Q = data.get("Q")

    if N is None or M is None or Q is None:
        print(f"[ERROR] Missing N, M, or Q for {data.get('cid')}")
        return data

    if Q <= 0:
        print(f"[WARNING] Non-positive Q for {data.get('cid')}: {Q}")
        data["phi"] = np.nan
        return data

    phi = N * np.sqrt(M) * np.sqrt(Q)

    data["phi"] = phi

    return data


# DETONATION VELOCITY
def det_v(data):
    """Calculate Kamlet-Jacobs detonation velocity.

    The velocity is calculated as:

    `D / km s-1 = 1.01 * sqrt(phi) * (1 + 1.3 * rho)`

    Parameters
    ----------
    data : dict
        Molecule dictionary containing `phi`, `density`, and `cid`.

    Returns
    -------
    dict
        Updated molecule dictionary containing detonation velocity under the
        `d` key.
    """

    phi = data.get("phi")
    rho = data.get("density")

    if phi is None or rho is None:
        print(f"[ERROR] Missing phi or density for {data.get('cid')}")
        return data

    if np.isnan(phi):
        data["d"] = np.nan
        return data

    d = 1.01 * np.sqrt(phi) * (1 + (1.3 * rho))

    data["d"] = d

    return data


# DETONATION PRESSURE
def det_p(data):
    """Calculate Kamlet-Jacobs detonation pressure.

    The pressure is calculated as:

    `P / GPa = 1.558 * phi * rho^2`

    Parameters
    ----------
    data : dict
        Molecule dictionary containing `phi`, `density`, and `cid`.

    Returns
    -------
    dict
        Updated molecule dictionary containing detonation pressure under the
        `p` key.
    """

    phi = data.get("phi")
    rho = data.get("density")

    if phi is None or rho is None:
        print(f"[ERROR] Missing phi or density for {data.get('cid')}")
        return data

    if np.isnan(phi):
        data["p"] = np.nan
        return data

    p = 1.558 * phi * (rho ** 2)

    data["p"] = p

    return data


# WRITE RESULTS BACK TO CSV
def write_to_csv(csv_path, results):
    """Write calculated density and detonation results back to the CSV.

    Existing rows are matched by CID. New result columns are added if they do
    not already exist.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file to update.
    results : list[dict]
        List of processed molecule dictionaries containing calculated density
        and Kamlet-Jacobs values.
    """
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"[ERROR] No headers found in CSV: {csv_path}")

        reader.fieldnames = [
            h.strip() if h is not None else h
            for h in reader.fieldnames
        ]

        fieldnames = reader.fieldnames
        id_column = find_id_column(fieldnames)

        rows = list(reader)

    new_fields = [
        "density / gcm-3",
        "base_density / gcm-3",
        "det_velocity / kms-1",
        "det_pressure / Gpa",
        "Q",
    ]

    for field in new_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    results_map = {
        str(mol["cid"]): mol
        for mol in results
    }

    for row in rows:
        mol_id = row.get(id_column, "").strip()

        if mol_id in results_map:
            mol = results_map[mol_id]

            row["density / gcm-3"] = mol.get("density")
            row["base_density / gcm-3"] = mol.get("base_density")
            row["det_velocity / kms-1"] = mol.get("d")
            row["det_pressure / Gpa"] = mol.get("p")
            row["Q"] = mol.get("Q")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# CREATE SCALING PLOTS
def create_scaling_plot(data, output_dir=None, show=False):
    """Create density-scaling plots for detonation velocity and pressure.

    The density is scaled using a factor gamma:

    - gamma = 0 gives rho = 0
    - gamma = 1 gives the original predicted density

    The scaled density is then used to recalculate detonation velocity and
    pressure.

    Parameters
    ----------
    data : dict
        Molecule dictionary containing `cid`, `density`, and `phi`.
    output_dir : str or None, optional
        Directory where plots should be saved. If None, plots are not saved.
    show : bool, optional
        Whether to display plots interactively. Default is False.

    Returns
    -------
    list[str] or None
        Paths to saved plot files if successful. Returns None if density or phi
        is missing or invalid.
    """

    cid = data.get("cid", "unknown")

    rho0 = data.get("density")
    phi = data.get("phi")

    if rho0 is None or phi is None:
        print(f"[WARNING] Missing density or phi for CID {cid}")
        return None

    if np.isnan(phi):
        print(f"[WARNING] Cannot create scaling plot because phi is NaN for CID {cid}")
        return None

    gamma = np.arange(0, 1.001, 0.001)

    # Scale density, not final D/P
    scaled_density = rho0 * gamma

    scaled_det_v = (
        1.01 *
        np.sqrt(phi) *
        (1 + (1.3 * scaled_density))
    )

    scaled_det_p = (
        1.558 *
        phi *
        (scaled_density ** 2)
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    # Detonation velocity plot
    plt.figure(figsize=(7, 5))

    plt.plot(
        gamma,
        scaled_det_v,
    )

    plt.xlabel("Density scaling factor, γ")
    plt.ylabel("Detonation velocity / km s$^{-1}$")
    plt.title(f"Detonation velocity vs scaled density: {cid}")
    plt.grid(True)
    plt.tight_layout()

    if output_dir is not None:
        v_path = os.path.join(
            output_dir,
            f"{cid}_det_velocity_density_scaling.png",
        )

        plt.savefig(
            v_path,
            dpi=300,
        )

        saved_files.append(v_path)

    if show:
        plt.show()

    plt.close()

    # Detonation pressure plot
    plt.figure(figsize=(7, 5))

    plt.plot(
        gamma,
        scaled_det_p,
    )

    plt.xlabel("Density scaling factor, γ")
    plt.ylabel("Detonation pressure / GPa")
    plt.title(f"Detonation pressure vs scaled density: {cid}")
    plt.grid(True)
    plt.tight_layout()

    if output_dir is not None:
        p_path = os.path.join(
            output_dir,
            f"{cid}_det_pressure_density_scaling.png",
        )

        plt.savefig(
            p_path,
            dpi=300,
        )

        saved_files.append(p_path)

    if show:
        plt.show()

    plt.close()

    return saved_files


# MAIN
if __name__ == "__main__":

    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))

    # Input CSV
    csv_path = os.path.abspath(args.input)

    # Optimised structure directory
    xyz_dir = os.path.abspath(
        os.path.join(
            parent_dir,
            "OPTIMISED_STRUCTURES",
            args.outdir,
        )
    )

    # Density model
    model_path = os.path.join(
        script_dir,
        "TOOLS",
        "direct_gbt_density_model.pkl",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Density model not found: {model_path}"
        )

    print(f"Input CSV: {csv_path}")
    print(f"Optimised structure directory: {xyz_dir}")
    print(f"Density model: {model_path}")

    density_model, density_feature_names = load_density_model(
        model_path,
    )

    # Read molecules
    data = read_csv(csv_path)

    results = []

    # Process molecules
    for mol in tqdm(data, desc="Processing molecules"):

        cid = mol["cid"]

        xyz_path = os.path.join(
            xyz_dir,
            f"{cid}",
            f"{cid}.xyz",
        )

        if not os.path.exists(xyz_path):
            print(f"[WARNING] Missing XYZ for {cid}: {xyz_path}")
            continue

        mol = calc_density(
            xyz_path,
            mol,
            density_model,
            density_feature_names,
        )

        if mol is None:
            continue

        mol = calc_gas_products(mol)
        mol = calc_phi(mol)
        mol = det_v(mol)
        mol = det_p(mol)

        if not args.no_plots:
            plot_dir = os.path.join(
                xyz_dir,
                cid,
            )

            create_scaling_plot(
                mol,
                output_dir=plot_dir,
                show=False,
            )

        results.append(mol)

    # Write results
    write_to_csv(
        csv_path,
        results,
    )

    print("\nDone.")
    print(f"Processed molecules: {len(results)}")
    print(f"Updated CSV: {csv_path}")