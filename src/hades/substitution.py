"""
Generate substituted molecules for the HADES workflow.

This script reads molecular cores, functional groups, and heteroatom fragments
from text files, randomly substitutes cores with those groups, filters duplicate
molecules using canonical SMILES, calculates synthetic accessibility scores, and
writes the generated molecules to a CSV file.
"""

import os
import random
import re
import csv
import argparse
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Contrib.SA_Score import sascorer
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate substituted molecules for HADES"
    )

    parser.add_argument(
        "--input", "-i",
        dest="output",
        type=str,
        default="hades_out.csv",
        help=(
            "Output CSV file path for generated molecules. "
            "This uses -i so the generated file can be passed directly "
            "as the input to the rest of the HADES workflow."
        )
    )

    parser.add_argument(
        "--output", "-o",
        dest="output",
        type=str,
        default=None,
        help=(
            "Optional alias for the output CSV file path. "
            "If provided, this overrides the default output path."
        )
    )

    parser.add_argument(
        "--num-molecules", "-n",
        type=int,
        default=1000,
        help="Number of unique molecules to generate. Default: 1000"
    )

    return parser.parse_args()

RDLogger.DisableLog("rdApp.*")

script_dir = os.path.dirname(os.path.abspath(__file__))
txt_path = os.path.join(script_dir, "TOOLS", "SUBSTITUTION")


# READ INPUT FILES
def read_data_files(txt_path):
    """Read molecular cores, functional groups, and heteroatoms from text files.

    The expected files are `cores.txt`, `functional_groups.txt`, and
    `heteroatoms.txt`. Each valid line should have the format:

    `name: SMILES`

    Blank lines and lines starting with `#` are ignored.

    Parameters
    ----------
    txt_path : str
        Directory containing the substitution input text files.

    Returns
    -------
    tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]
        Tuple containing lists of core molecules, functional groups, and
        heteroatom fragments. Each entry is stored as `(name, smiles)`.
    """
    cores = []
    func_gs = []
    heteroatoms = []

    with open(os.path.join(txt_path, "cores.txt"), "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split(":")

            if len(parts) == 2:
                name = parts[0].strip()
                smiles = parts[1].strip()

                mol = Chem.MolFromSmiles(smiles)

                if mol:
                    cores.append((name, smiles))

    with open(os.path.join(txt_path, "functional_groups.txt"), "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split(":")

            if len(parts) == 2:
                name = parts[0].strip()
                smiles = parts[1].strip()

                func_gs.append((name, smiles))

    with open(os.path.join(txt_path, "heteroatoms.txt"), "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split(":")

            if len(parts) == 2:
                name = parts[0].strip()
                smiles = parts[1].strip()

                heteroatoms.append((name, smiles))

    return cores, func_gs, heteroatoms


# CANONICAL SMILES FOR DUPLICATE CHECKING

def canonicalise_smiles(smiles):
    """Convert a SMILES string into a canonical RDKit SMILES.

    This is used for duplicate checking. Different non-canonical SMILES
    representations of the same molecule should produce the same canonical
    SMILES.

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    str or None
        Canonical SMILES string if successful. Returns None if the molecule
        cannot be parsed or sanitised.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)

        canonical_smiles = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True
        )

        return canonical_smiles

    except Exception:
        return None


# SYNTHETIC ACCESSIBILITY SCORE
def calculate_sascore(smiles):
    """Calculate the RDKit synthetic accessibility score.

    Lower values are generally easier to synthesise. Higher values are generally
    harder to synthesise. The usual range is approximately 1 to 10.

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    float or None
        Synthetic accessibility score if successful. Returns None if the
        molecule cannot be parsed, sanitised, or scored.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
        sascore = sascorer.calculateScore(mol)
        return sascore

    except Exception:
        return None


# MARK CORE ATOMS WITH WILDCARDS
def mark_core_atoms(core_smiles, num_sites, max_attempts=10):
    """Add wildcard attachment points to random heavy atoms in a core molecule.

    The function randomly selects heavy atoms in the input molecule and attaches
    dummy atoms, represented as wildcard atoms, to create substitution sites.

    Parameters
    ----------
    core_smiles : str
        SMILES string for the core molecule.
    num_sites : int
        Number of substitution sites to add.
    max_attempts : int, optional
        Maximum number of attempts to generate a valid marked molecule.
        Default is 10.

    Returns
    -------
    str or None
        SMILES string containing wildcard attachment sites if successful.
        Returns None if marking fails.
    """
    attempt = 0

    while attempt < max_attempts:
        attempt += 1

        try:
            mol = Chem.MolFromSmiles(core_smiles)

            if mol is None:
                return None

            Chem.Kekulize(mol, clearAromaticFlags=False)

            mol = Chem.AddHs(mol)
            rw_mol = Chem.RWMol(mol)

            heavy_atoms = [
                atom.GetIdx()
                for atom in rw_mol.GetAtoms()
                if atom.GetAtomicNum() > 1
            ]

            if not heavy_atoms:
                return None

            attach_indices = random.sample(
                heavy_atoms,
                min(num_sites, len(heavy_atoms))
            )

            for idx in attach_indices:
                dummy = Chem.Atom(0)
                new_idx = rw_mol.AddAtom(dummy)
                rw_mol.AddBond(idx, new_idx, Chem.BondType.SINGLE)

            mol_no_H = Chem.RemoveHs(rw_mol.GetMol())
            Chem.SanitizeMol(mol_no_H)

            smiles_noH = Chem.MolToSmiles(mol_no_H, canonical=False)

            # Convert bare * into (*) for consistency.
            smiles_noH = re.sub(r"(?<!\()\*(?!\))", "(*)", smiles_noH)

            return smiles_noH

        except Exception:
            continue

    return None


# SUBSTITUTE WILDCARDS
def substitute_wildcards_rdkit(core_smiles, func_groups):
    """Replace wildcard atoms in a molecule with random substituents.

    Each wildcard atom in the core molecule is replaced with a randomly chosen
    functional group or heteroatom fragment. Functional groups must themselves
    contain a wildcard atom defining their attachment point.

    Parameters
    ----------
    core_smiles : str
        SMILES string containing one or more wildcard atoms.
    func_groups : list[tuple[str, str]]
        Available substituents stored as `(name, smiles)` pairs.

    Returns
    -------
    tuple[str or None, list[str] or None]
        Final substituted SMILES string and list of attached substituent names.
        Returns `(None, None)` if the input molecule has no valid wildcard atoms.
        Returns `(None, attached_names)` if final sanitisation fails.
    """
    mol = Chem.MolFromSmiles(core_smiles)

    if mol is None:
        return None, None

    rw_mol = Chem.RWMol(mol)

    dummy_atoms = [
        atom.GetIdx()
        for atom in rw_mol.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]

    if not dummy_atoms:
        return None, None

    attached_names = []

    for dummy_idx in sorted(dummy_atoms, reverse=True):
        name, fg_smiles = random.choice(func_groups)

        # Try original functional group SMILES first, then uppercase as a fallback.
        for attempt in [fg_smiles, fg_smiles.upper()]:
            fg_mol = Chem.MolFromSmiles(attempt)

            if fg_mol is None:
                continue

            fg_dummy_atoms = [
                atom.GetIdx()
                for atom in fg_mol.GetAtoms()
                if atom.GetAtomicNum() == 0
            ]

            if not fg_dummy_atoms:
                continue

            fg_dummy_idx = fg_dummy_atoms[0]

            combo = Chem.CombineMols(rw_mol, fg_mol)
            combo_rw = Chem.RWMol(combo)

            fg_offset = rw_mol.GetNumAtoms()
            fg_dummy_global = fg_offset + fg_dummy_idx

            core_neighs = [
                neighbour.GetIdx()
                for neighbour in combo_rw.GetAtomWithIdx(
                    dummy_idx
                ).GetNeighbors()
            ]

            fg_neighs = [
                fg_offset + neighbour.GetIdx()
                for neighbour in fg_mol.GetAtomWithIdx(
                    fg_dummy_idx
                ).GetNeighbors()
            ]

            for c_idx in core_neighs:
                for f_idx in fg_neighs:
                    combo_rw.AddBond(c_idx, f_idx, Chem.BondType.SINGLE)

            combo_rw.BeginBatchEdit()
            combo_rw.RemoveAtom(max(dummy_idx, fg_dummy_global))
            combo_rw.RemoveAtom(min(dummy_idx, fg_dummy_global))
            combo_rw.CommitBatchEdit()

            try:
                Chem.SanitizeMol(combo_rw)

                final_smiles = Chem.MolToSmiles(
                    combo_rw,
                    canonical=False
                )

                if final_smiles != core_smiles and "*" not in final_smiles:
                    rw_mol = combo_rw
                    attached_names.append(name)
                    break

            except Exception:
                continue

    try:
        Chem.SanitizeMol(rw_mol)

        final_smiles = Chem.MolToSmiles(
            rw_mol,
            canonical=False
        )

        return final_smiles, attached_names

    except Exception:
        return None, attached_names


# WRITE OUTPUT CSV
def write_to_csv(data, output_csv):
    """Write generated molecule data to a CSV file.

    The output contains one row per generated molecule and dynamically adds
    substitution columns according to the largest number of substitutions found
    in the generated dataset.

    Parameters
    ----------
    data : list[tuple[str, str, str, float, list[str]]]
        Generated molecule data. Each row should contain the CID, core name,
        canonical SMILES, synthetic accessibility score, and list of
        substitutions.
    output_csv : str
        Path to the output CSV file.
    """
    max_subs = max(len(row[4]) for row in data)

    headers = (
        ["CID", "Core", "SMILES", "sascore"]
        + [f"Substitution {i + 1}" for i in range(max_subs)]
    )

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for cid, core, canonical_smiles, sascore, subs in data:
            padded_subs = subs + [""] * (max_subs - len(subs))

            if sascore is None:
                sascore_value = ""
            else:
                sascore_value = f"{sascore:.4f}"

            writer.writerow(
                [cid, core, canonical_smiles, sascore_value] + padded_subs
            )

    print(f"\nSaved {len(data)} unique molecules to {output_csv}")


# MAIN GENERATION LOOP
def main():

    args = parse_args()
    iteration_count = args.num_molecules

    if args.output is None:
        output_csv = "hades_out.csv"
    else:
        output_csv = args.output

    max_attempts = 5
    max_total_attempts = iteration_count * 500

    cores, func_gs, heteroatoms = read_data_files(txt_path)

    all_substituents = func_gs + heteroatoms

    if not cores:
        raise ValueError("No valid cores were found.")

    if not all_substituents:
        raise ValueError("No valid functional groups or heteroatoms were found.")

    generated_unique = 0
    total_attempts = 0
    duplicate_count = 0
    failed_count = 0

    results = []

    seen_canonical_smiles = set()

    pbar = tqdm(
        total=iteration_count,
        desc="Generating unique molecules",
        unit="mol"
    )

    while generated_unique < iteration_count:
        total_attempts += 1

        if total_attempts > max_total_attempts:
            print(f"\nStopped after {max_total_attempts} total attempts.")
            print(
                f"Requested {iteration_count} unique molecules, "
                f"but only generated {generated_unique}.")
            print(
                "This probably means the generator is now producing mostly "
                "duplicates, or the accessible chemical space is too small.")
            break

        core_name, core_smiles = random.choice(cores)

        attempts = 0
        success = False

        while attempts < max_attempts and not success:
            attempts += 1

            current_smiles = core_smiles
            substitution_count = random.randint(1, 4)

            substitutions_done = []
            success = True

            for _ in range(substitution_count):
                marked_smiles = mark_core_atoms(
                    current_smiles,
                    num_sites=1
                )

                if marked_smiles is None:
                    success = False
                    break

                substituted_smiles, group_names = substitute_wildcards_rdkit(
                    marked_smiles,
                    all_substituents
                )

                if substituted_smiles is None:
                    success = False
                    break

                current_smiles = substituted_smiles

                if group_names:
                    substitutions_done.extend(group_names)

        if not success:
            failed_count += 1
            continue

        canonical_smiles = canonicalise_smiles(current_smiles)

        if canonical_smiles is None:
            failed_count += 1
            continue

        if canonical_smiles in seen_canonical_smiles:
            duplicate_count += 1
            continue

        sascore = calculate_sascore(canonical_smiles)

        if sascore is None:
            failed_count += 1
            continue

        seen_canonical_smiles.add(canonical_smiles)

        generated_unique += 1
        cid = f"CID{generated_unique}"

        results.append(
            (
                cid,
                core_name,
                canonical_smiles,
                sascore,
                substitutions_done
            )
        )

        pbar.update(1)

    pbar.close()

    print("\nGeneration complete.")
    print(f"Requested unique molecules: {iteration_count}")
    print(f"Unique molecules generated: {generated_unique}")
    print(f"Duplicate molecules rejected: {duplicate_count}")
    print(f"Failed molecules rejected: {failed_count}")

    if results:
        write_to_csv(results, output_csv)
    else:
        print("No molecules generated successfully.")


if __name__ == "__main__":
    main()
