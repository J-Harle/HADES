import os
import random
import re
import csv
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

script_dir = os.path.dirname(os.path.abspath(__file__))
txt_path = os.path.join(script_dir, "SUBSTITUTION")
output_csv = os.path.join(script_dir, "..", "hades_test_out.csv")


def read_data_files(txt_path):
    """Read core, functional group, and heteroatom definitions from text files.

    Each file should contain lines in the format `name: SMILES`, with optional
    comment lines starting with '#'. Invalid or empty lines are ignored.
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


def break_aromaticity(mol):
    """Convert all aromatic atoms and bonds in a molecule into non-aromatic single bonds,
    while recording which bonds were originally aromatic.

    Returns:
        tuple[rdkit.Chem.RWMol, list[tuple[int, int]]]:
            - The modified molecule with all aromaticity removed.
            - A list of (begin_idx, end_idx) pairs for aromatic bonds.
    """
    rw_mol = Chem.RWMol(mol)
    aromatic_bonds = []

    for bond in rw_mol.GetBonds():
        if bond.GetIsAromatic():
            aromatic_bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)

    for atom in rw_mol.GetAtoms():
        if atom.GetIsAromatic():
            atom.SetIsAromatic(False)

    return rw_mol, aromatic_bonds


def restore_aromaticity(mol, aromatic_bonds):
    """Restore aromatic bonds to a molecule based on a list of saved bond indices."""
    rw_mol = Chem.RWMol(mol)

    for b1, b2 in aromatic_bonds:
        bond = rw_mol.GetBondBetweenAtoms(b1, b2)
        if bond is not None:
            bond.SetIsAromatic(True)
            bond.SetBondType(Chem.BondType.SINGLE)

    for atom in rw_mol.GetAtoms():
        neighs = [n for n in atom.GetNeighbors()]
        if any(
            rw_mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetIsAromatic()
            for n in neighs
        ):
            atom.SetIsAromatic(True)

    try:
        Chem.SanitizeMol(rw_mol)
    except Exception:
        Chem.SetAromaticity(rw_mol)

    return rw_mol


def mark_core_atoms(core_smiles, num_sites, max_attempts=10):
    """Mark random attachment sites in a core molecule by adding dummy atoms (*)."""
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            mol = Chem.MolFromSmiles(core_smiles)
            if mol is None:
                return None

            mol = Chem.AddHs(mol)
            rw_mol, aromatic_bonds = break_aromaticity(mol)

            heavy_atoms = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() > 1]
            if not heavy_atoms:
                return None

            attach_indices = random.sample(heavy_atoms, min(num_sites, len(heavy_atoms)))
            for idx in attach_indices:
                dummy = Chem.Atom(0)
                new_idx = rw_mol.AddAtom(dummy)
                rw_mol.AddBond(idx, new_idx, Chem.BondType.SINGLE)

            mol_noH = Chem.RemoveHs(rw_mol)
            # Restore aromaticity before SMILES export
            mol_noH = restore_aromaticity(mol_noH, aromatic_bonds)

            smiles_noH = Chem.MolToSmiles(mol_noH, canonical=False)
            smiles_noH = re.sub(r'(?<!\()\*(?!\))', '(*)', smiles_noH)
            return smiles_noH

        except Exception as e:
            print(f"Marking attempt failed: {e}")
            continue
    return None


def substitute_wildcards_rdkit(core_smiles, func_groups):
    """Replace wildcard atoms (*) in a core with random functional groups."""
    mol = Chem.MolFromSmiles(core_smiles)
    if mol is None:
        return None, None

    rw_mol = Chem.RWMol(mol)
    dummy_atoms = [a.GetIdx() for a in rw_mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy_atoms:
        return None, None

    attached_names = []

    for dummy_idx in sorted(dummy_atoms, reverse=True):
        name, fg_smiles = random.choice(func_groups)
        attached_names.append(name)

        for attempt in [fg_smiles, fg_smiles.upper()]:
            fg_mol = Chem.MolFromSmiles(attempt)
            if fg_mol is None:
                continue

            fg_dummy = [a.GetIdx() for a in fg_mol.GetAtoms() if a.GetAtomicNum() == 0]
            if not fg_dummy:
                print(f"Functional group {name} has no dummy attachment site. Skipping.")
                continue
            fg_dummy_idx = fg_dummy[0]

            combo = Chem.CombineMols(rw_mol, fg_mol)
            combo_rw = Chem.RWMol(combo)
            fg_offset = rw_mol.GetNumAtoms()
            fg_dummy_global = fg_offset + fg_dummy_idx

            core_neighs = [n.GetIdx() for n in combo_rw.GetAtomWithIdx(dummy_idx).GetNeighbors()]
            fg_neighs = [fg_offset + n.GetIdx() for n in fg_mol.GetAtomWithIdx(fg_dummy_idx).GetNeighbors()]

            for c_idx in core_neighs:
                for f_idx in fg_neighs:
                    combo_rw.AddBond(c_idx, f_idx, Chem.BondType.SINGLE)

            combo_rw.BeginBatchEdit()
            combo_rw.RemoveAtom(max(dummy_idx, fg_dummy_global))
            combo_rw.RemoveAtom(min(dummy_idx, fg_dummy_global))
            combo_rw.CommitBatchEdit()

            try:
                Chem.SanitizeMol(combo_rw)
                final_smiles = Chem.MolToSmiles(combo_rw, canonical=False)

                if final_smiles == core_smiles or "*" in final_smiles:
                    print(f"Substitution with {name} yielded no change, retrying with uppercase form.")
                    continue
                else:
                    rw_mol = combo_rw
                    break

            except Exception as e:
                print(f"Sanitization failed for {name} ({attempt}): {e}")
                continue

    try:
        Chem.SanitizeMol(rw_mol)
        final_smiles = Chem.MolToSmiles(rw_mol, canonical=False)
        return final_smiles, ", ".join(attached_names)
    except Exception as e:
        print(f"Final sanitization failed: {e}")
        return None, ", ".join(attached_names)


def write_to_csv(data, output_csv):
    """Write generated molecule SMILES strings to a CSV file."""
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CID", "SMILES"])
        writer.writerows(data)
    print(f"\nSaved {len(data)} molecules to {output_csv}")


if __name__ == "__main__":
    """Main program entry point."""
    iteration_count = 2000
    max_attempts = 5

    cores, func_gs, heteroatoms = read_data_files(txt_path)
    all_substituents = func_gs + heteroatoms

    core_name, core_smiles = random.choice(cores)
    print(f"\nUsing core: {core_name} ({core_smiles})\n")

    generated = 0
    results = []

    while generated < iteration_count:
        attempts = 0
        success = False

        while attempts < max_attempts and not success:
            attempts += 1
            current_smiles = core_smiles
            substitution_count = random.randint(1, 6)
            success = True

            for i in range(substitution_count):
                marked_smiles = mark_core_atoms(current_smiles, num_sites=1)
                if marked_smiles is None:
                    success = False
                    break
                substituted_smiles, group_name = substitute_wildcards_rdkit(marked_smiles, all_substituents)
                if substituted_smiles is None:
                    success = False
                    break
                current_smiles = substituted_smiles

            if not success:
                print(f"Attempt #{attempts} failed.\n")

        if success:
            generated += 1
            cid = f"C{generated:03d}"
            results.append((cid, current_smiles))
        else:
            print(f"Failed to generate molecule after {max_attempts} attempts.\n")

    if results:
        write_to_csv(results, output_csv)
    else:
        print("No molecules generated successfully.")
