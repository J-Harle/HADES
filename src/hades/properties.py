"""
High-level HADES property API.

This module provides calculate_properties(), which accepts one SMILES
string and returns oxygen balance, enthalpy of formation, density and
Kamlet-Jacobs detonation properties.
"""

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from .oxygen_balance import (
    atom_counts_from_smiles,
    oxygen_balance,
)

from .isodesmic import (
    HARTREE_TO_KJMOL,
    build_combo_cache,
    calculate_hf,
    calculate_hr,
    dict_to_vector,
    filter_and_average,
    find_isodesmics,
    read_mace_energy_from_thermo,
    read_reference_csv,
    vectorise_reference,
)

from .det_v_p import (
    calc_density,
    calc_gas_products,
    calc_phi,
    det_p,
    det_v,
    load_density_model,
)


# ============================================================
# PACKAGE PATHS
# ============================================================

PACKAGE_DIRECTORY = Path(__file__).resolve().parent

REFERENCE_CSV = (
    PACKAGE_DIRECTORY
    / "TOOLS"
    / "EOF"
    / "isodesmic.csv"
)

DENSITY_MODEL_PATH = (
    PACKAGE_DIRECTORY
    / "TOOLS"
    / "DENSITY"
    / "gbt_density_model.pkl"
)


# Change this if the calculation method changes and old cached
# results should no longer be reused.
CACHE_VERSION = 1


# ============================================================
# IN-MEMORY CACHES
# ============================================================

_REFERENCE_DATA = None
_DENSITY_MODEL = None
_DENSITY_FEATURE_NAMES = None

_REACTION_CACHE = None
_REACTION_CACHE_SIGNATURE = None
_REACTION_FEATURES = None

_PROPERTY_CACHE = {}


# ============================================================
# MOLECULE HELPERS
# ============================================================

def canonicalise_smiles(smiles):
    """Validate and canonicalise a SMILES string."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    return Chem.MolToSmiles(mol, canonical=True)


def molecule_identifier(smiles):
    """Generate a stable directory-safe identifier from a SMILES."""
    digest = sha256(smiles.encode("utf-8")).hexdigest()[:16]
    return f"HADES_{digest}"


def atom_bond_counts(smiles):
    """
    Generate the atom/bond representation expected by the
    isodesmic reaction code.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    counts = Counter()

    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1

    for bond in mol.GetBonds():
        atom_1 = bond.GetBeginAtom().GetSymbol()
        atom_2 = bond.GetEndAtom().GetSymbol()

        atom_1, atom_2 = sorted((atom_1, atom_2))

        key = f"{atom_1}-{atom_2} {bond.GetBondType()}"
        counts[key] += 1

    return dict(counts)


def finite_or_none(value):
    """Convert NumPy values and non-finite numbers for JSON output."""
    if value is None:
        return None

    value = float(value)

    if not math.isfinite(value):
        return None

    return value


# ============================================================
# LOAD STATIC HADES DATA
# ============================================================

def get_reference_data():
    """Load the isodesmic reference database once."""
    global _REFERENCE_DATA
    global _REACTION_FEATURES

    if _REFERENCE_DATA is None:
        if not REFERENCE_CSV.is_file():
            raise FileNotFoundError(
                f"Isodesmic reference CSV not found: {REFERENCE_CSV}"
            )

        _REFERENCE_DATA = read_reference_csv(str(REFERENCE_CSV))

        _REACTION_FEATURES = set()

        for reference in _REFERENCE_DATA.values():
            _REACTION_FEATURES.update(
                reference["atom_bond_dict"].keys()
            )

    return _REFERENCE_DATA


def get_density_model():
    """Load the density model once."""
    global _DENSITY_MODEL
    global _DENSITY_FEATURE_NAMES

    if _DENSITY_MODEL is None:
        if not DENSITY_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Density model not found: {DENSITY_MODEL_PATH}"
            )

        (
            _DENSITY_MODEL,
            _DENSITY_FEATURE_NAMES,
        ) = load_density_model(str(DENSITY_MODEL_PATH))

    return _DENSITY_MODEL, _DENSITY_FEATURE_NAMES


# ============================================================
# ISODESMIC REACTION CACHE
# ============================================================

def get_reaction_cache(target_counts, max_reactants):
    """
    Build and cache the reference combinations used to find
    isodesmic reactions.

    The cache is rebuilt only if a target introduces an atom or
    bond feature that was not previously present.
    """
    global _REACTION_CACHE
    global _REACTION_CACHE_SIGNATURE
    global _REACTION_FEATURES

    reference_data = get_reference_data()

    new_features = set(target_counts) - _REACTION_FEATURES

    if new_features:
        _REACTION_FEATURES.update(new_features)

    feature_list = sorted(_REACTION_FEATURES)

    signature = (
        int(max_reactants),
        tuple(feature_list),
    )

    if (
        _REACTION_CACHE is None
        or signature != _REACTION_CACHE_SIGNATURE
    ):
        feature_index = {
            feature: index
            for index, feature in enumerate(feature_list)
        }

        names, reference_vectors = vectorise_reference(
            reference_data,
            feature_index,
        )

        combo_vectors, combo_names = build_combo_cache(
            names,
            reference_vectors,
            max_r=max_reactants,
        )

        combo_lookup = {
            tuple(vector.tolist()): index
            for index, vector in enumerate(combo_vectors)
        }

        _REACTION_CACHE = {
            "feature_index": feature_index,
            "combo_vectors": combo_vectors,
            "combo_names": combo_names,
            "combo_lookup": combo_lookup,
        }

        _REACTION_CACHE_SIGNATURE = signature

    return _REACTION_CACHE


# ============================================================
# MACE CALCULATION
# ============================================================

def ensure_mace_calculation(
    smiles,
    molecule_id,
    work_directory,
    n_conformers,
    mmff_max_iters,
    force,
):
    """
    Generate, optimise and calculate thermochemistry for one molecule.

    Existing XYZ and thermochemistry files are reused unless force=True.
    """
    molecule_directory = work_directory / molecule_id

    xyz_path = molecule_directory / f"{molecule_id}.xyz"
    thermo_path = molecule_directory / f"{molecule_id}_thermo.txt"

    if (
        not force
        and xyz_path.is_file()
        and thermo_path.is_file()
    ):
        return xyz_path

    # Imported here so importing hades itself does not immediately
    # initialise Torch and MACE.
    from .create_object import (
        create_ase_objs,
        optimise_and_write_single,
    )

    atoms_list = create_ase_objs(
        [smiles],
        n_conformers=n_conformers,
        mmff_max_iters=mmff_max_iters,
    )

    if not atoms_list or atoms_list[0] is None:
        raise RuntimeError(
            f"Conformer generation failed for {smiles}"
        )

    optimise_and_write_single(
        (
            atoms_list[0],
            smiles,
            molecule_id,
            str(work_directory),
        )
    )

    if not xyz_path.is_file():
        raise RuntimeError(
            f"MACE optimisation did not produce an XYZ file: {xyz_path}"
        )

    if not thermo_path.is_file():
        raise RuntimeError(
            "MACE vibrational thermochemistry failed for "
            f"{smiles}. Expected: {thermo_path}"
        )

    return xyz_path


# ============================================================
# ENTHALPY OF FORMATION
# ============================================================

def calculate_isodesmic_eof(
    smiles,
    molecule_id,
    work_directory,
    max_reactants,
    max_products,
):
    """Calculate the isodesmic formation enthalpy."""
    reference_data = get_reference_data()
    target_counts = atom_bond_counts(smiles)

    target_energy = read_mace_energy_from_thermo(
        str(work_directory),
        molecule_id,
    )

    if target_energy is None:
        raise RuntimeError(
            f"Could not read MACE thermochemical energy for {molecule_id}"
        )

    reaction_cache = get_reaction_cache(
        target_counts,
        max_reactants=max_reactants,
    )

    target_vector = dict_to_vector(
        target_counts,
        reaction_cache["feature_index"],
    )

    solutions = find_isodesmics(
        combo_vecs=reaction_cache["combo_vectors"],
        combo_names=reaction_cache["combo_names"],
        target_vec=target_vector,
        target_smiles=smiles,
        combo_lookup=reaction_cache["combo_lookup"],
        max_product_r=max_products,
    )

    if not solutions:
        raise RuntimeError(
            f"No isodesmic reactions found for {smiles}"
        )

    calculate_hr(
        solutions=solutions,
        target_energy=target_energy,
        json_data=reference_data,
        target_smiles=smiles,
    )

    calculate_hf(
        solutions=solutions,
        json_data=reference_data,
        target_smiles=smiles,
    )

    mean_hf, spread, filtered = filter_and_average(solutions)

    if mean_hf is None or not filtered:
        raise RuntimeError(
            "No isodesmic reactions passed the reaction-energy "
            f"filter for {smiles}"
        )

    return {
        "hf_kj_mol": mean_hf * HARTREE_TO_KJMOL,
        "hf_spread_kj_mol": spread * HARTREE_TO_KJMOL,
        "n_isodesmic_reactions": len(filtered),
    }


# ============================================================
# PUBLIC API
# ============================================================

def calculate_properties(
    smiles,
    work_dir="HADES_CALCULATIONS",
    max_reactants=3,
    max_products=3,
    n_conformers=10,
    mmff_max_iters=25,
    force=False,
):
    """
    Calculate HADES properties for one SMILES string.

    Parameters
    ----------
    smiles : str
        Molecular SMILES string.

    work_dir : str or pathlib.Path
        Persistent directory for structures, thermochemistry and
        cached property results.

    max_reactants : int
        Maximum number of reference reactants in the isodesmic search.

    max_products : int
        Maximum number of additional reference products.

    n_conformers : int
        Number of initial RDKit conformers.

    mmff_max_iters : int
        Maximum MMFF optimisation iterations per conformer.

    force : bool
        Recalculate the molecule even when cached results exist.

    Returns
    -------
    dict
        Calculated molecular properties.
    """
    canonical_smiles = canonicalise_smiles(smiles)
    molecule_id = molecule_identifier(canonical_smiles)

    work_directory = Path(work_dir).expanduser().resolve()
    work_directory.mkdir(parents=True, exist_ok=True)

    memory_cache_key = (
        canonical_smiles,
        str(work_directory),
        max_reactants,
        max_products,
    )

    if not force and memory_cache_key in _PROPERTY_CACHE:
        return dict(_PROPERTY_CACHE[memory_cache_key])

    molecule_directory = work_directory / molecule_id
    result_path = molecule_directory / "properties.json"

    # Reuse a previous completed calculation.
    if not force and result_path.is_file():
        with result_path.open("r") as handle:
            cached_result = json.load(handle)

        if (
            cached_result.get("cache_version") == CACHE_VERSION
            and cached_result.get("smiles") == canonical_smiles
            and cached_result.get("max_reactants") == max_reactants
            and cached_result.get("max_products") == max_products
        ):
            _PROPERTY_CACHE[memory_cache_key] = cached_result
            return dict(cached_result)

    xyz_path = ensure_mace_calculation(
        smiles=canonical_smiles,
        molecule_id=molecule_id,
        work_directory=work_directory,
        n_conformers=n_conformers,
        mmff_max_iters=mmff_max_iters,
        force=force,
    )

    # --------------------------------------------------------
    # Oxygen balance
    # --------------------------------------------------------

    mol = Chem.MolFromSmiles(canonical_smiles)
    counts = atom_counts_from_smiles(canonical_smiles)
    molecular_weight = float(Descriptors.MolWt(mol))

    ob = oxygen_balance(
        C=counts.get("C", 0),
        H=counts.get("H", 0),
        O=counts.get("O", 0),
        mol_weight=molecular_weight,
    )

    # --------------------------------------------------------
    # Isodesmic enthalpy of formation
    # --------------------------------------------------------

    eof_result = calculate_isodesmic_eof(
        smiles=canonical_smiles,
        molecule_id=molecule_id,
        work_directory=work_directory,
        max_reactants=max_reactants,
        max_products=max_products,
    )

    # --------------------------------------------------------
    # Density and Kamlet-Jacobs properties
    # --------------------------------------------------------

    density_model, density_feature_names = get_density_model()

    detonation_data = {
        "cid": molecule_id,
        "smiles": canonical_smiles,
        "eof": eof_result["hf_kj_mol"],
    }

    detonation_data = calc_density(
        xyz_path=str(xyz_path),
        data=detonation_data,
        density_model=density_model,
        density_feature_names=density_feature_names,
    )

    if detonation_data is None:
        raise RuntimeError(
            f"Density prediction failed for {canonical_smiles}"
        )

    detonation_data = calc_gas_products(detonation_data)
    detonation_data = calc_phi(detonation_data)
    detonation_data = det_v(detonation_data)
    detonation_data = det_p(detonation_data)

    result = {
        "cache_version": CACHE_VERSION,
        "molecule_id": molecule_id,
        "smiles": canonical_smiles,
        "oxygen_balance_percent": finite_or_none(ob),
        "hf_kj_mol": finite_or_none(
            eof_result["hf_kj_mol"]
        ),
        "hf_spread_kj_mol": finite_or_none(
            eof_result["hf_spread_kj_mol"]
        ),
        "n_isodesmic_reactions": int(
            eof_result["n_isodesmic_reactions"]
        ),
        "molecular_weight_g_mol": finite_or_none(
            detonation_data.get("mw")
        ),
        "molecular_volume_angstrom3": finite_or_none(
            detonation_data.get("volume")
        ),
        "base_density_g_cm3": finite_or_none(
            detonation_data.get("base_density")
        ),
        "density_g_cm3": finite_or_none(
            detonation_data.get("density")
        ),
        "q_cal_g": finite_or_none(
            detonation_data.get("Q")
        ),
        "detonation_velocity_km_s": finite_or_none(
            detonation_data.get("d")
        ),
        "detonation_pressure_gpa": finite_or_none(
            detonation_data.get("p")
        ),
        "max_reactants": int(max_reactants),
        "max_products": int(max_products),
    }

    molecule_directory.mkdir(parents=True, exist_ok=True)

    with result_path.open("w") as handle:
        json.dump(result, handle, indent=4)

    _PROPERTY_CACHE[memory_cache_key] = result

    return dict(result)
