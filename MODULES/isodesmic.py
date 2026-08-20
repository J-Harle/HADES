"""
Calculate enthalpies of formation using isodesmic reactions.

This script estimates target-molecule gas-phase enthalpies of formation using
MACE thermochemical energies and a reference set of molecules with known
experimental enthalpies of formation.

The workflow is:

1. Read a target CSV containing CID and SMILES columns.
2. Read MACE thermochemistry files for the target molecules.
3. Read an isodesmic reference CSV.
4. Represent molecules using atom-count and bond-count feature vectors.
5. Search for atom- and bond-balanced isodesmic reactions.
6. Filter reactions by reaction energy.
7. Calculate weighted-average enthalpies of formation.
8. Write the results back to the input CSV.
"""

from collections import Counter
from itertools import combinations_with_replacement
from rdkit import Chem
from math import comb
from tqdm import tqdm
import pandas as pd
import os
import argparse
import numpy as np

def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the input CSV path, optimised
        structure directory name, and optional row limit.
    """
    parser = argparse.ArgumentParser(
        description="Calculate enthalpy of formation using isodesmic reactions"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input CSV containing CID and SMILES columns"
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
        "--nrows", "-n",
        type=int,
        default=None,
        help="Number of rows/targets to process. Default: process all rows."
    )

    return parser.parse_args()


script_dir = os.path.dirname(os.path.abspath(__file__))

EV_TO_HARTREE = 1.0 / 27.211386245988
HARTREE_TO_KJMOL = 2625.49962

TARGET_ID_COLUMN_CANDIDATES = [
    "CID",
    "molecule",
    "MOLECULE",
    "FILENAME",
]


def find_target_id_column(df):
    """
    Find the column used to identify target molecule folders.

    Priority:
        CID -> molecule -> MOLECULE -> FILENAME

    Returns
    -------
    str
        Name of the matched column.
    """
    # Strip whitespace from column names
    df.columns = [str(col).strip() for col in df.columns]

    for col in TARGET_ID_COLUMN_CANDIDATES:
        if col in df.columns:
            return col

    raise KeyError(
        "[ERROR] No valid target ID column found. "
        f"Expected one of: {TARGET_ID_COLUMN_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def find_smiles_column(df):
    """
    Find the SMILES column robustly.
    """
    df.columns = [str(col).strip() for col in df.columns]

    for col in ["SMILES", "smiles", "Smiles"]:
        if col in df.columns:
            return col

    raise KeyError(
        "[ERROR] No SMILES column found. "
        f"Available columns: {list(df.columns)}"
    )


# ENERGY PARSING
def read_mace_energy_from_thermo(optimised_dir, cid):
    """Read a MACE enthalpy value from a target thermochemistry file.

    The function searches for a file named `{cid}_thermo.txt` inside the target
    molecule directory and extracts the first line containing `Enthalpy H`.

    Parameters
    ----------
    optimised_dir : str
        Directory containing optimised molecule subdirectories.
    cid : str
        Compound identifier used as the molecule folder name.

    Returns
    -------
    float or None
        MACE enthalpy in Hartree. Returns None if the thermochemistry file is
        missing or the enthalpy value cannot be parsed.
    """
    thermo_path = os.path.join(
        optimised_dir,
        f"{cid}",
        f"{cid}_thermo.txt",
    )

    # print(thermo_path)

    if not os.path.exists(thermo_path):
        return None

    with open(thermo_path, "r") as f:
        for line in f:
            if "Enthalpy H" in line:
                try:
                    energy_eV = float(line.split(":")[1].strip().split()[0])
                    return energy_eV * EV_TO_HARTREE
                except:
                    return None
    return None


# READ REFERENCE CSV
def read_reference_csv(csv_path):
    """Read and process the isodesmic reference molecule CSV.

    Each valid reference molecule is converted into a canonical SMILES string and
    an atom/bond-count dictionary. Experimental enthalpies are converted from
    kJ mol-1 to Hartree, and MACE energies are converted from eV to Hartree.

    Parameters
    ----------
    csv_path : str
        Path to the reference CSV file.

    Returns
    -------
    dict[str, dict]
        Dictionary keyed by reference molecule name. Each value contains the
        canonical SMILES, experimental enthalpy, MACE energy, and atom/bond
        feature dictionary.
    """
    df = pd.read_csv(csv_path)
    ref_data = {}

    for _, row in df.iterrows():

        name = str(row["MOL"])
        smi = row["SMILES"]

        if pd.isna(smi):
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.AddHs(mol)

        counts = Counter()

        for atom in mol.GetAtoms():
            counts[atom.GetSymbol()] += 1

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetSymbol()
            a2 = bond.GetEndAtom().GetSymbol()
            a, b = sorted([a1, a2])
            counts[f"{a}-{b} {bond.GetBondType()}"] += 1

        exp = row.get("exp e /kJ mol")
        mace = row.get("mace e /eV")

        if pd.isna(exp) or pd.isna(mace):
            continue

        ref_data[name] = {
            "SMILES": canonical_smi,
            "exp_energy": float(exp) / HARTREE_TO_KJMOL,
            "MACE_energy_Ha": float(mace) * EV_TO_HARTREE,
            "atom_bond_dict": dict(counts)
        }

    print(f"Loaded {len(ref_data)} reference molecules")
    return ref_data


# TARGETS
def read_target_mol(csv_path, optimised_dir, nrows=None):
    """Read target molecules and their MACE thermochemical energies.

    Parameters
    ----------
    csv_path : str
        Path to the target CSV file containing `CID` and `SMILES` columns.
    optimised_dir : str
        Directory containing optimised target molecule folders.
    nrows : int or None, optional
        Number of rows to read from the target CSV. If None, all rows are read.

    Returns
    -------
    list[dict]
        List of target molecule dictionaries containing CID, SMILES, atom/bond
        feature dictionary, and MACE energy in Hartree.
    """
    df = pd.read_csv(csv_path, nrows=nrows)
    target_id_col = find_target_id_column(df)
    smiles_col = find_smiles_column(df)

    print(f"[INFO] Using target ID column: {target_id_col}")
    print(f"[INFO] Using SMILES column: {smiles_col}")

    targets = []

    for _, row in df.iterrows():

        cid = str(row[target_id_col]).strip()
        smiles = row[smiles_col]

        if cid == "" or cid.lower() == "nan":
            print("[SKIPPED] Missing target identifier")
            continue

        if pd.isna(smiles):
            print(f"[SKIPPED] {target_id_col} {cid} — missing SMILES")
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[SKIPPED] {target_id_col} {cid} — invalid SMILES")
            continue

        mol = Chem.AddHs(mol)

        counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetSymbol()
            a2 = bond.GetEndAtom().GetSymbol()
            a, b = sorted([a1, a2])
            counts[f"{a}-{b} {bond.GetBondType()}"] += 1

        mace_energy = read_mace_energy_from_thermo(optimised_dir, cid)

        if mace_energy is None:
            print(f"[SKIPPED] {target_id_col} {cid} — no thermo energy")
            continue

        targets.append({
            "CID": cid,
            "target_id_column": target_id_col,
            "SMILES": smiles,
            "atom_bond_dict": dict(counts),
            "MACE_energy_Ha": mace_energy
        })

    print(f"Loaded {len(targets)} targets")
    return targets

# DICT OPS (kept for bond table/debug)
def add_dicts(a, b):
    """Add values from two dictionaries.

    Parameters
    ----------
    a : dict
        First dictionary.
    b : dict
        Second dictionary.

    Returns
    -------
    dict
        Dictionary containing the sum of matching values from both inputs.
    """
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out

def subtract_dicts(a, b):
    """Subtract values in one dictionary from another.

    Parameters
    ----------
    a : dict
        Dictionary to subtract from.
    b : dict
        Dictionary containing values to subtract.

    Returns
    -------
    dict
        Dictionary containing `a - b` for matching keys.
    """
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) - v
    return out

def clean_dict(d):
    """Remove zero-valued entries from a dictionary.

    Parameters
    ----------
    d : dict
        Input dictionary.

    Returns
    -------
    dict
        Dictionary containing only non-zero values.
    """
    return {k: v for k, v in d.items() if v != 0}


# FEATURE SPACE
def build_feature_index(json_data, targets):
    """Build a shared atom/bond feature index.

    The feature index contains every atom-count and bond-count key found in the
    reference molecules and target molecules.

    Parameters
    ----------
    json_data : dict[str, dict]
        Reference molecule data.
    targets : list[dict]
        Target molecule data.

    Returns
    -------
    tuple[list[str], dict[str, int]]
        Sorted feature list and dictionary mapping each feature to its vector
        index.
    """
    features = set()

    for d in json_data.values():
        features.update(d["atom_bond_dict"].keys())

    for t in targets:
        features.update(t["atom_bond_dict"].keys())

    feature_list = sorted(features)
    feature_index = {k: i for i, k in enumerate(feature_list)}

    return feature_list, feature_index


def dict_to_vector(d, feature_index):
    """Convert an atom/bond dictionary into a numerical feature vector.

    Parameters
    ----------
    d : dict
        Atom/bond-count dictionary.
    feature_index : dict[str, int]
        Mapping from feature names to vector positions.

    Returns
    -------
    numpy.ndarray
        Integer feature vector.
    """
    vec = np.zeros(len(feature_index), dtype=np.int16)

    for k, v in d.items():
        if k in feature_index:
            vec[feature_index[k]] = v

    return vec


# VECTORISE
def vectorise_reference(json_data, feature_index):
    """Vectorise all reference molecules.

    Parameters
    ----------
    json_data : dict[str, dict]
        Reference molecule data.
    feature_index : dict[str, int]
        Mapping from feature names to vector positions.

    Returns
    -------
    tuple[list[str], numpy.ndarray]
        Reference molecule names and corresponding feature vectors.
    """
    names = []
    vectors = []

    for name, d in json_data.items():
        names.append(name)
        vectors.append(dict_to_vector(d["atom_bond_dict"], feature_index))

    return names, np.array(vectors)


def vectorise_targets(targets, feature_index):
    """Vectorise all target molecules.

    Parameters
    ----------
    targets : list[dict]
        Target molecule data.
    feature_index : dict[str, int]
        Mapping from feature names to vector positions.

    Returns
    -------
    numpy.ndarray
        Array of target feature vectors.
    """
    return np.array([
        dict_to_vector(t["atom_bond_dict"], feature_index)
        for t in targets
    ])


# BUILD COMBO CACHE (VECTOR)
def build_combo_cache(names, vectors, max_r=None):
    """Build cached reference-molecule combination vectors.

    Combinations with replacement are generated for reference molecules up to
    size `max_r`. Each combination is represented by the sum of its feature
    vectors.

    Parameters
    ----------
    names : list[str]
        Reference molecule names.
    vectors : numpy.ndarray
        Reference molecule feature vectors.
    max_r : int
        Maximum combination size.

    Returns
    -------
    tuple[numpy.ndarray, list[list[str]]]
        Array of summed combination vectors and matching lists of molecule
        names for each combination.
    """
    combo_vecs = []
    combo_names = []

    print("[CACHE] Building combinations...")

    n = len(names)

    for r in range(1, max_r + 1):

        total = comb(n + r - 1, r)

        iterator = combinations_with_replacement(range(n), r)

        for idxs in tqdm(iterator, total=total, desc=f"r={r}"):

            vec_sum = np.sum(vectors[list(idxs)], axis=0)

            combo_vecs.append(vec_sum)
            combo_names.append([names[i] for i in idxs])

    print(f"[CACHE] Built {len(combo_vecs)} combos")

    return np.array(combo_vecs), combo_names


# LOOKUP
def build_reverse_lookup(json_data, feature_index):
    """Build a vector-to-name lookup for reference molecules.

    Parameters
    ----------
    json_data : dict[str, dict]
        Reference molecule data.
    feature_index : dict[str, int]
        Mapping from feature names to vector positions.

    Returns
    -------
    dict[tuple[int, ...], str]
        Dictionary mapping feature-vector tuples to reference molecule names.
    """
    lookup = {}

    for name, d in json_data.items():
        vec = dict_to_vector(d["atom_bond_dict"], feature_index)
        lookup[tuple(vec.tolist())] = name

    return lookup


# VECTORISED ISODESMIC SEARCH
def find_isodesmics(combo_vecs, combo_names, target_vec, target_smiles, 
                    combo_lookup=None, max_product_r = None):
    """Find atom- and bond-balanced isodesmic reactions for a target molecule.

        The search checks whether a reference-molecule reactant combination can be
        balanced by the target molecule plus an optional reference-molecule product
        combination.

        Parameters
        ----------
        combo_vecs : numpy.ndarray
            Cached reference-combination feature vectors.
        combo_names : list[list[str]]
            Reference molecule names corresponding to each cached combination.
        target_vec : numpy.ndarray
            Target molecule feature vector.
        target_smiles : str
            Target molecule SMILES string. This is used as the product-side target
            identifier.
        combo_lookup : dict[tuple[int, ...], int] or None, optional
            Lookup mapping combination vectors to indices in `combo_vecs`. If None,
            the lookup is built inside the function.
        max_product_r : int or None, optional
            Maximum allowed number of extra reference products.

        Returns
        -------
        list[dict]
            List of candidate reactions. Each reaction dictionary contains
            `reactants` and `products`.
        """    
    # Build lookup if not provided
    if combo_lookup is None:
        combo_lookup = {
            tuple(vec.tolist()): i
            for i, vec in enumerate(combo_vecs)
        }

    # Compute leftover vectors
    leftover = combo_vecs - target_vec
    valid_mask = np.all(leftover >= 0, axis=1)

    valid_idx = np.where(valid_mask)[0]
    valid_leftover = leftover[valid_mask]

    solutions = []

    # Main loop
    for i, vec in zip(valid_idx, valid_leftover):

        reactants = combo_names[i]

        # ---- Case 1: perfect balance (no extra products)
        if np.all(vec == 0):
            solutions.append({
                "reactants": reactants,
                "products": [target_smiles]
            })
            continue

        # ---- Case 2: match leftover to product combinations
        key = tuple(vec.tolist())

        if key not in combo_lookup:
            continue

        prod_idx = combo_lookup[key]
        product_combo = combo_names[prod_idx]

        # ---- Control explosion: limit product size
        if len(product_combo) > max_product_r:
            continue

        products = [target_smiles] + product_combo

        solutions.append({
            "reactants": reactants,
            "products": products
        })

    return solutions


# ENERGY CALCULATIONS
def calculate_hr(solutions, target_energy, json_data, target_smiles):
    """Calculate reaction energies for candidate isodesmic reactions.

    The reaction energy is calculated as:

    Delta_E = E_products - E_reactants

    Reference molecule energies are taken from `json_data`, while the target
    molecule energy is supplied separately.

    Parameters
    ----------
    solutions : list[dict]
        Candidate reaction dictionaries to update in place.
    target_energy : float
        Target molecule MACE energy in Hartree.
    json_data : dict[str, dict]
        Reference molecule data containing MACE energies.
    target_smiles : str
        Target molecule SMILES string used to identify the target product.
    """
    for sol in solutions:

        E_reactants = sum(json_data[r]["MACE_energy_Ha"] for r in sol["reactants"])

        E_products = 0.0
        for p in sol["products"]:
            if p in json_data:
                E_products += json_data[p]["MACE_energy_Ha"]
            else:
                E_products += target_energy

        sol["Delta_E_Ha"] = E_products - E_reactants


def calculate_hf(solutions, json_data, target_smiles):
    """Calculate target enthalpies of formation from reaction energies.

    The target enthalpy is calculated from the known experimental enthalpies of
    the reference reactants and non-target products:

    Hf_target = sum(Hf_reactants) - sum(Hf_other_products) - Delta_E

    Parameters
    ----------
    solutions : list[dict]
        Candidate reaction dictionaries to update in place.
    json_data : dict[str, dict]
        Reference molecule data containing experimental enthalpies.
    target_smiles : str
        Target molecule SMILES string used to identify the target product.
    """
    for sol in solutions:

        reactants = sol["reactants"]
        products = sol["products"]

        other_products = [p for p in products if p != target_smiles]

        sum_reactants = sum(json_data[r]["exp_energy"] for r in reactants)
        sum_products = sum(json_data[p]["exp_energy"] for p in other_products)

        sol["Hf_target"] = sum_reactants - sum_products - sol["Delta_E_Ha"]


def filter_and_average(solutions):
    """Filter candidate reactions and calculate weighted-average enthalpy.

    Reactions are retained if their absolute reaction energy is less than
    0.005 Hartree. The retained enthalpies are averaged using inverse reaction
    energy magnitude as weights.

    Parameters
    ----------
    solutions : list[dict]
        Candidate reaction dictionaries containing `Delta_E_Ha` and
        `Hf_target`.

    Returns
    -------
    tuple[float or None, float or None, list[dict]]
        Weighted mean enthalpy in Hartree, enthalpy spread in Hartree, and the
        filtered reactions. If no reactions pass filtering, returns
        `(None, None, [])`.
    """
    filtered = [
        s for s in solutions
        if abs(s["Delta_E_Ha"]) < 0.005 # 0.005
    ]

    if not filtered:
        print("\n[DEBUG] No reactions passed filtering.")
        print("ΔE values (kJ/mol):")

        for s in solutions:
            dE_kj = s["Delta_E_Ha"] * HARTREE_TO_KJMOL
            print(f"{dE_kj:.2f}")

        return None, None, []

    weights = np.array([1 / (abs(s["Delta_E_Ha"]) + 1e-6) for s in filtered])
    hf_values = np.array([s["Hf_target"] for s in filtered])

    weighted_mean = np.sum(weights * hf_values) / np.sum(weights)
    spread = np.max(hf_values) - np.min(hf_values)

    return weighted_mean, spread, filtered


# CSV UPDATE
def update_csv_with_results(csv_path, results):
    """Update the target CSV with enthalpy results.

    Adds or updates the following columns:

    - `Hf /kJmol-1`
    - `Hf_spread /kJmol-1`

    Parameters
    ----------
    csv_path : str
        Path to the target CSV file to update.
    results : dict[str, dict]
        Results keyed by CID. Each value should contain `Hf_kJmol` and
        `Hf_spread_kJmol`.
    """
    df = pd.read_csv(csv_path)

    target_id_col = find_target_id_column(df)

    print(f"[INFO] Updating CSV using target ID column: {target_id_col}")

    df[target_id_col] = df[target_id_col].astype(str).str.strip()

    hf_col = "Hf /kJmol-1"
    spread_col = "Hf_spread /kJmol-1"

    if hf_col not in df.columns:
        df[hf_col] = None

    if spread_col not in df.columns:
        df[spread_col] = None

    for i, row in df.iterrows():

        cid = str(row[target_id_col]).strip()

        if cid in results:
            df.at[i, hf_col] = results[cid]["Hf_kJmol"]
            df.at[i, spread_col] = results[cid]["Hf_spread_kJmol"]

    df.to_csv(csv_path, index=False)

# MAIN
if __name__ == "__main__":

    args = parse_args()

    max_r = 3          # max reactant combo size
    max_p = 3          # max product combo size
    save_every = 1000

    csv_path = os.path.abspath(args.input)

    optimised_dir = os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            # "OPTIMISED_STRUCTURES",
            args.outdir,
        )
    )

    ref_csv_path = os.path.join(
        script_dir,
        "TOOLS",
        "EOF",
        "isodesmic.csv",
    )

    print(f"Input CSV: {csv_path}")
    print(f"Optimised structure directory: {optimised_dir}")
    print(f"Reference CSV: {ref_csv_path}")

    # Load data
    json_data = read_reference_csv(ref_csv_path)

    targets = read_target_mol(
        csv_path,
        optimised_dir,
        nrows=None,
    )

    # Build feature space
    feature_list, feature_index = build_feature_index(
        json_data,
        targets,
    )

    # Vectorise
    names, ref_vecs = vectorise_reference(
        json_data,
        feature_index,
    )

    target_vecs = vectorise_targets(
        targets,
        feature_index,
    )

    # Build combination cache
    combo_vecs, combo_names = build_combo_cache(
        names,
        ref_vecs,
        max_r=max_r,
    )

    # FAST lookup for combo vectors
    print("[LOOKUP] Building combo lookup...")

    combo_lookup = {
        tuple(vec.tolist()): i
        for i, vec in enumerate(combo_vecs)
    }

    results = {}

    # Main loop
    for idx, (target, target_vec) in enumerate(
        zip(targets, target_vecs),
        start=1,
    ):

        cid = target["CID"]
        print(f"\n=== CID {cid} ({idx}/{len(targets)}) ===")

        solutions = find_isodesmics(
            combo_vecs=combo_vecs,
            combo_names=combo_names,
            target_vec=target_vec,
            target_smiles=target["SMILES"],
            combo_lookup=combo_lookup,
            max_product_r=max_p,
        )

        if not solutions:
            print("No reactions found")
            results[cid] = {
                "Hf_kJmol": None,
                "Hf_spread_kJmol": None,
            }

        else:
            calculate_hr(
                solutions,
                target["MACE_energy_Ha"],
                json_data,
                target["SMILES"],
            )

            calculate_hf(
                solutions,
                json_data,
                target["SMILES"],
            )

            mean_hf, spread, filtered = filter_and_average(
                solutions,
            )

            if not filtered:
                print("No good reactions after filtering")

                results[cid] = {
                    "Hf_kJmol": None,
                    "Hf_spread_kJmol": None,
                }

            else:
                print(f"Valid reactions: {len(filtered)}")
                print(
                    f"Mean Hf: {mean_hf * HARTREE_TO_KJMOL:.2f} kJ/mol"
                )

                results[cid] = {
                    "Hf_kJmol": mean_hf * HARTREE_TO_KJMOL,
                    "Hf_spread_kJmol": spread * HARTREE_TO_KJMOL,
                }

        # Periodic save to free up RAM
        if idx % save_every == 0:
            print(f"\n[CHECKPOINT] Writing results at step {idx}...")
            update_csv_with_results(csv_path, results)
            print("[CHECKPOINT] CSV updated.")

    # Final save
    update_csv_with_results(csv_path, results)

    print("\nDone.")
    print(f"Processed targets: {len(results)}")
    print(f"Updated CSV: {csv_path}")
