import os
import csv
import joblib
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# =========================================================
# DENSITY MODEL FEATURE DEFINITIONS
# =========================================================

# These are the features used by the density-model training script.
# The feature order must remain identical to the order used for training.
DEFAULT_DENSITY_FEATURE_NAMES = [
    "MolecularDensity",
    "NumHAcceptors",
    "NumHDonors",
    "Phi",
    "NumAromaticRings",
    "TPSA",
    "MolLogP",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "NumAliphaticRings",
]

DESCRIPTOR_LIBRARY = dict(Descriptors.descList)


# =========================================================
# READ CSV
# =========================================================

def read_csv(csv_path):
    data = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            eof_val = row["Hf /kJmol-1"].strip()

            if not eof_val:
                print(f"[WARNING] Missing EOF for CID {row['CID']}")
                continue

            mol = {
                "smiles": row["SMILES"],
                "eof": float(eof_val),
                "cid": row["CID"],
            }

            data.append(mol)

    return data


# =========================================================
# LOAD DENSITY MODEL
# =========================================================

def load_density_model(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[ERROR] Density model not found: {model_path}"
        )

    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        if "model" not in bundle:
            raise KeyError(
                "[ERROR] Model bundle does not contain a 'model' entry."
            )

        model = bundle["model"]
        feature_names = bundle.get("feature_names")
    else:
        # Fallback for a model saved directly rather than inside a bundle.
        model = bundle
        feature_names = None

    if feature_names is None:
        print(
            "[WARNING] No feature_names were stored with the model. "
            "Using the feature order from the current training script."
        )
        feature_names = DEFAULT_DENSITY_FEATURE_NAMES.copy()
    else:
        feature_names = list(feature_names)

    unsupported_features = [
        feature_name
        for feature_name in feature_names
        if (
            feature_name != "MolecularDensity"
            and feature_name not in DESCRIPTOR_LIBRARY
        )
    ]

    if unsupported_features:
        raise ValueError(
            "[ERROR] The density model contains unsupported features: "
            f"{unsupported_features}"
        )

    model_feature_count = getattr(model, "n_features_in_", None)

    if (
        model_feature_count is not None
        and model_feature_count != len(feature_names)
    ):
        raise ValueError(
            "[ERROR] Saved feature-name count does not match the model: "
            f"{len(feature_names)} names, but the model expects "
            f"{model_feature_count} features."
        )

    print("\nLoaded density model")
    print("Model path:", model_path)
    print("Model type:", type(model))
    print("Model features:")

    for index, feature_name in enumerate(feature_names, start=1):
        print(f"  {index:2d}. {feature_name}")

    return model, feature_names


# =========================================================
# BUILD DENSITY MODEL FEATURES
# =========================================================

def build_density_model_features(
    smiles,
    molecular_density,
    feature_names,
):
    """
    Build one prediction row using the feature names saved with the model.

    The training script uses MolecularDensity first, followed by the selected
    RDKit descriptors. Iterating over the stored names guarantees that the
    prediction feature order is identical to the training feature order.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    feature_values = []

    for feature_name in feature_names:
        if feature_name == "MolecularDensity":
            feature_value = molecular_density
        else:
            descriptor_function = DESCRIPTOR_LIBRARY.get(feature_name)

            if descriptor_function is None:
                print(
                    f"[ERROR] RDKit descriptor is unavailable: "
                    f"{feature_name}"
                )
                return None

            try:
                feature_value = float(descriptor_function(mol))
            except Exception as exc:
                print(
                    f"[ERROR] Could not calculate {feature_name} "
                    f"for {smiles}: {exc}"
                )
                return None

        if not np.isfinite(feature_value):
            print(
                f"[ERROR] Non-finite density feature "
                f"{feature_name} for {smiles}: {feature_value}"
            )
            return None

        feature_values.append(float(feature_value))

    return np.asarray([feature_values], dtype=float)


# =========================================================
# CALCULATE DENSITY USING ML MODEL
# =========================================================

def calc_density(
    xyz_path,
    data,
    density_model,
    density_feature_names,
):
    smiles = data.get("smiles")
    cid = data.get("cid")

    # --------------------------------------------------
    # SMILES -> molecular weight
    # --------------------------------------------------
    mol_smiles = Chem.MolFromSmiles(smiles)

    if mol_smiles is None:
        print(f"[ERROR] Invalid SMILES for {cid}")
        return None

    # This matches the molecular-weight calculation in the training script.
    mw = float(Descriptors.MolWt(mol_smiles))

    # --------------------------------------------------
    # XYZ -> molecular volume
    # --------------------------------------------------
    if not os.path.isfile(xyz_path):
        print(f"[ERROR] XYZ file not found for {cid}: {xyz_path}")
        return None

    mol_xyz = Chem.MolFromXYZFile(xyz_path)

    if mol_xyz is None:
        print(f"[ERROR] Failed to read {xyz_path}")
        return None

    try:
        volume = float(AllChem.ComputeMolVolume(mol_xyz))
    except Exception as exc:
        print(f"[ERROR] Failed to compute volume for {cid}: {exc}")
        return None

    if not np.isfinite(volume) or volume <= 0.0:
        print(f"[ERROR] Invalid volume for {cid}: {volume}")
        return None

    # --------------------------------------------------
    # Molecular density used as the first model feature
    # --------------------------------------------------
    molecular_density = mw / volume

    if not np.isfinite(molecular_density) or molecular_density <= 0.0:
        print(
            f"[ERROR] Invalid molecular density for {cid}: "
            f"{molecular_density}"
        )
        return None

    # --------------------------------------------------
    # Build the feature row in the saved model order
    # --------------------------------------------------
    X = build_density_model_features(
        smiles,
        molecular_density,
        density_feature_names,
    )

    if X is None:
        print(f"[ERROR] Could not build density features for {cid}")
        return None

    if X.shape[1] != len(density_feature_names):
        print(
            f"[ERROR] Feature mismatch for {cid}: "
            f"X has {X.shape[1]} features, but the model bundle "
            f"contains {len(density_feature_names)} feature names."
        )
        return None

    model_feature_count = getattr(density_model, "n_features_in_", None)

    if (
        model_feature_count is not None
        and X.shape[1] != model_feature_count
    ):
        print(
            f"[ERROR] Feature mismatch for {cid}: "
            f"X has {X.shape[1]} features, but the model expects "
            f"{model_feature_count}."
        )
        return None

    # --------------------------------------------------
    # Predict density
    # --------------------------------------------------
    try:
        density = float(density_model.predict(X)[0])
    except Exception as exc:
        print(f"[ERROR] Density prediction failed for {cid}: {exc}")
        return None

    if not np.isfinite(density):
        print(f"[ERROR] Non-finite density prediction for {cid}: {density}")
        return None

    # --------------------------------------------------
    # Store results
    # --------------------------------------------------
    data["density"] = density
    data["base_density"] = molecular_density
    data["mw"] = mw
    data["volume"] = volume

    return data


# =========================================================
# CALCULATE GAS PRODUCTS
# =========================================================

def calc_gas_products(data):
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


# =========================================================
# CALCULATE PHI
# =========================================================

def calc_phi(data):
    """
    Phi = N * sqrt(M) * sqrt(Q)

    Where:
        N = moles of gaseous product per gram of explosive
        M = average molecular mass of gaseous products
        Q = chemical energy of detonation reaction
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


# =========================================================
# DETONATION VELOCITY
# =========================================================

def det_v(data):
    """
    D / km s-1 = 1.01 * sqrt(phi) * (1 + 1.3*rho)
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


# =========================================================
# DETONATION PRESSURE
# =========================================================

def det_p(data):
    """
    P / GPa = 1.558 * phi * rho^2
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


# =========================================================
# WRITE RESULTS BACK TO CSV
# =========================================================

def write_to_csv(csv_path, results):
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames if reader.fieldnames else []

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
        mol["cid"]: mol
        for mol in results
    }

    for row in rows:
        cid = row["CID"]

        if cid in results_map:
            mol = results_map[cid]

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


# =========================================================
# CREATE SCALING PLOTS
# =========================================================

def create_scaling_plot(data, output_dir=None, show=False):
    """
    Creates scaling plots for detonation velocity and pressure
    by scaling the density using gamma.

    gamma = 0 gives rho = 0
    gamma = 1 gives the original predicted density
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

    # --------------------------------------------------
    # Detonation velocity plot
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Detonation pressure plot
    # --------------------------------------------------
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


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --------------------------------------------------
    # Input CSV and XYZ directory
    # --------------------------------------------------
    csv_path = os.path.join(script_dir, "..", "large_data.csv",)
    xyz_dir = os.path.join(script_dir, "..", "OPTIMISED_STRUCTURES", "LARGE_DATASET",)

    # csv_path = os.path.join(script_dir, "..", "30_bench.csv")
    # xyz_dir = os.path.join(script_dir, "..", "OPTIMISED_STRUCTURES", "30_MOL")

    # csv_path = os.path.join(script_dir, "..", "bak_30_bench.csv")
    # xyz_dir = os.path.join(script_dir, "..", "OPTIMISED_STRUCTURES", "DET_V_P_TEST")

    # --------------------------------------------------
    # Density model
    # --------------------------------------------------
    model_path = os.path.join(
        script_dir,
        "TOOLS", 
        "test_direct_gbt_density_model.pkl",
    )

    density_model, density_feature_names = load_density_model(
        model_path,
    )

    # --------------------------------------------------
    # Read molecules
    # --------------------------------------------------
    data = read_csv(csv_path)

    results = []

    # --------------------------------------------------
    # Process molecules
    # --------------------------------------------------
    for mol in tqdm(data, desc="Processing molecules"):

        cid = mol["cid"]

        xyz_path = os.path.join(
            xyz_dir,
            f"{cid}",
            f"{cid}.xyz",
        )

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

    # --------------------------------------------------
    # Write results
    # --------------------------------------------------
    write_to_csv(
        csv_path,
        results,
    )

    print("\nDone.")
    print(f"Processed molecules: {len(results)}")
    print(f"Updated CSV: {csv_path}")