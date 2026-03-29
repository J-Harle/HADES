import os
import random
import re
import csv
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

script_dir = os.path.dirname(os.path.abspath(__file__))
txt_path = os.path.join(script_dir, "SUBSTITUTION")
output_csv = os.path.join(script_dir, "..", "hades_out.csv")


def read_data_files(txt_path):
    """Read core, functional group, and heteroatom definitions from text files."""
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
    """Convert all aromatic atoms and bonds in a molecule into non-aromatic single bonds."""
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
            mol_noH = restore_aromaticity(mol_noH, aromatic_bonds)

            smiles_noH = Chem.MolToSmiles(mol_noH, canonical=False)
            smiles_noH = re.sub(r'(?<!\()\*(?!\))', '(*)', smiles_noH)

            # print(f"Marked core: {smiles_noH}")
            return smiles_noH

        except Exception as e:
            # print(f"\nMarking attempt failed: {e}")
            continue
    return None


def substitute_wildcards_rdkit(core_smiles, func_groups):

    mol = Chem.MolFromSmiles(core_smiles)
    if mol is None:
        return None, None

    rw_mol = Chem.RWMol(mol)
    dummy_atoms = [a.GetIdx() for a in rw_mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy_atoms:
        return None, None

    attached_names = []

    for dummy_idx in sorted(dummy_atoms, reverse=True):  # reverse need to stop atom index shifting
        name, fg_smiles = random.choice(func_groups)
        attached_names.append(name)

        # print(f"\nAttaching functional group '{name}' ({fg_smiles})")

        for attempt in [fg_smiles, fg_smiles.upper()]:  # This will try to attach the aromatic version, if it fails, the non-aromatic version
            fg_mol = Chem.MolFromSmiles(attempt)
            if fg_mol is None:
                continue

            fg_dummy = [a.GetIdx() for a in fg_mol.GetAtoms() if a.GetAtomicNum() == 0]  # check to see if there are any dummy atoms
            if not fg_dummy:
                print(f"\nFunctional group '{name}' has no dummy site. Skipping.")
                continue
            fg_dummy_idx = fg_dummy[0]

            combo = Chem.CombineMols(rw_mol, fg_mol)  # combine the two molecules. will be in the form mol.func group (not connected)
            combo_rw = Chem.RWMol(combo)  # make it editable
            fg_offset = rw_mol.GetNumAtoms()  # how many atoms were added?
            fg_dummy_global = fg_offset + fg_dummy_idx  # increase the atom index by the nymber of atoms in the core mol to get the global index of the fg dummy

            core_neighs = [n.GetIdx() for n in combo_rw.GetAtomWithIdx(dummy_idx).GetNeighbors()]  # the nummy atom in the core mol
            fg_neighs = [fg_offset + n.GetIdx() for n in fg_mol.GetAtomWithIdx(fg_dummy_idx).GetNeighbors()]  # the dummy atom in the func group

            for c_idx in core_neighs:
                for f_idx in fg_neighs:
                    combo_rw.AddBond(c_idx, f_idx, Chem.BondType.SINGLE)  # connect the two molecules with a sinbgle bond

            combo_rw.BeginBatchEdit()  # edit molecules in "read only mode"
            combo_rw.RemoveAtom(max(dummy_idx, fg_dummy_global))  # remove the highest idx dummy first to avoid reindexing issues
            combo_rw.RemoveAtom(min(dummy_idx, fg_dummy_global))  # then remove the other dummy (if num_sites > 1)
            combo_rw.CommitBatchEdit()  # finish editing, save changes

            try:
                Chem.SanitizeMol(combo_rw)
                final_smiles = Chem.MolToSmiles(combo_rw, canonical=False)

                if final_smiles == core_smiles or "*" in final_smiles:
                    # print(f"Substitution with '{name}' yielded no change, retrying with uppercase form.")
                    continue
                else:
                    rw_mol = combo_rw
                    # print(f"\nMolecule after substitution: {final_smiles}")
                    break

            except Exception as e:
                print(f"\nSanitization failed for '{name}' ({attempt}): {e}")
                continue

    try:
        Chem.SanitizeMol(rw_mol)
        final_smiles = Chem.MolToSmiles(rw_mol, canonical=False)
        # print(f"\nFinal substituted molecule: {final_smiles}\n")
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
    iteration_count = 10000
    max_attempts = 5

    cores, func_gs, heteroatoms = read_data_files(txt_path)
    all_substituents = func_gs + heteroatoms

    generated = 0
    results = []

    pbar = tqdm(total=iteration_count, desc="Generating molecules", unit="mol")

    while generated < iteration_count:
        core_name, core_smiles = random.choice(cores)
        # print(f"\nUsing core: {core_name} ({core_smiles})\n")

        attempts = 0
        success = False

        while attempts < max_attempts and not success:
            attempts += 1
            current_smiles = core_smiles
            substitution_count = random.randint(1, 4)
            success = True

            # print(f"\n=== Attempt {attempts}: performing {substitution_count} substitutions ===")

            for i in range(substitution_count):
                # print(f"\n--- Substitution step {i + 1} ---")
                marked_smiles = mark_core_atoms(current_smiles, num_sites=1)
                if marked_smiles is None:
                    success = False
                    break

                substituted_smiles, group_name = substitute_wildcards_rdkit(
                    marked_smiles, all_substituents
                )
                if substituted_smiles is None:
                    success = False
                    break

                current_smiles = substituted_smiles
                # print(f"\nMolecule after step {i + 1}: {current_smiles}")

            # if not success:
                # print(f"Attempt #{attempts} failed.\n")

        if success:
            generated += 1
            cid = f"C{generated:03d}"
            results.append((cid, current_smiles))
            pbar.update(1)
        else:
            print(f"Failed to generate molecule after {max_attempts} attempts.\n")

    pbar.close()

    if results:
        write_to_csv(results, output_csv)
    else:
        print("No molecules generated successfully.")
