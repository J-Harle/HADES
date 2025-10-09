import os
import urllib.request
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms, units
from ase.io import read, write
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import AllChem
from mace.calculators import MACECalculator


def get_mace_calculator():
    """
    Ensure MACE model is available in CALCULATORS directory and return calculator.
    """
    parent_dir = os.getcwd()
    calc_dir = os.path.join(parent_dir, "CALCULATORS")
    os.makedirs(calc_dir, exist_ok=True)

    model_file = os.path.join(calc_dir, "MACE-OFF23_medium.model")
    if not os.path.exists(model_file):
        print(f"Downloading MACE model to {model_file}")
        url = "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_medium.model?raw=true"
        urllib.request.urlretrieve(url, model_file)
        print("Download complete")

    return MACECalculator(
        model_paths=[model_file],
        dispersion=False,
        default_dtype="float64",
        device="cpu"  # change to "cuda" if GPU available
    )


def create_ase_objs(smiles_list, max_attempts=5):
    """
    Convert SMILES strings to ASE Atoms objects using RDKit embedding.
    """
    atoms_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        success = False
        for attempt in range(max_attempts):
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
                success = True
                break

        if not success or mol.GetNumConformers() == 0:
            print(f"Embedding failed for SMILES: {smiles}")
            atoms_list.append(None)
            continue

        xyz_block = Chem.MolToXYZBlock(mol)
        atoms = read(StringIO(xyz_block), format="xyz")
        atoms_list.append(atoms)

    return atoms_list


def optimise_and_write_single(args):
    """
    Run optimisation for a single molecule, then perform a single point energy calculation
    and write the result to the comment line of the .xyz file.
    """
    atoms, smiles, mol_id, base_dir = args
    parent_dir = os.getcwd()
    optimised_dir = os.path.join(parent_dir, base_dir)
    os.makedirs(optimised_dir, exist_ok=True)

    calc = get_mace_calculator()
    atoms.calc = calc

    mol_dir = os.path.join(optimised_dir, str(mol_id))
    os.makedirs(mol_dir, exist_ok=True)

    log_file = os.path.join(mol_dir, f"{mol_id}.log")
    dyn = BFGS(atoms, logfile=log_file)
    dyn.run(fmax=0.001)

    spe = atoms.get_potential_energy()

    xyz_path = os.path.join(mol_dir, f"{mol_id}.xyz")
    with open(xyz_path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"SMILES: {smiles}  SPE (eV): {spe:.8f}\n")
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{symbol:2} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    return mol_id, xyz_path, spe


def optimise_and_write_parallel(atoms_list, smiles_list, id_list, base_dir="OPTIMISED_STRUCTURES", ncores=20):
    """
    Parallelised using ProcessPoolExecutor.
    Skips molecules whose embedding failed (atoms=None).
    """
    tasks = [
        (atoms, smiles, mol_id, base_dir)
        for atoms, smiles, mol_id in zip(atoms_list, smiles_list, id_list)
        if atoms is not None
    ]

    results = []
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        futures = {executor.submit(optimise_and_write_single, t): t[2] for t in tasks}
        for future in as_completed(futures):
            mol_id = futures[future]
            try:
                mol_id, xyz_path, spe = future.result()
                print(f"Optimised ID: {mol_id} → {xyz_path} | SPE: {spe:.6f} eV")
                results.append((mol_id, xyz_path, spe))
            except Exception as e:
                print(f"ID: {mol_id} failed: {e}")

    return results

def get_bonding_info(smiles_list):
    """
    Get bonding information from SMILES strings.
    """
    feature_vectors = {}
    for name, smiles in SMILES_dict.items():
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        atom_counter = Counter()
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            label = f"{symbol}_arom" if atom.GetIsAromatic() and symbol == "C" else symbol
            atom_counter[label] += 1

        bond_counter = Counter()
        for bond in mol.GetBonds():
            atoms = sorted([bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()])
            bond_type = str(bond.GetBondType())
            bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
            bond_counter[bond_label] += 1

        atom_vector = [atom_counter.get(a, 0) for a in atom_labels]
        bond_vector = [bond_counter.get(b, 0) for b in bond_labels]
        feature_vectors[name] = atom_vector + bond_vector

    return feature_vectors

if __name__ == "__main__":
    smiles_dict = {
        # "methylnitrite": "CON=O",
        # "methylnitrate": "CO[N+](=O)[O-]",
        # "dinitromethane": "C([N+](=O)[O-])[N+](=O)[O-]",
        # "ethylnitrite": "CCON=O",
        # "ethylnitrate": "CCO[N+](=O)[O-]",
        # "propylnitrite": "CCCON=O",
        # "methyl2nitro2propane": "CC(C)(C)[N+](=O)[O-]",
        # "nbutylnitrite": "CCCCON=O",
        # "tbutylnitrite": "CC(C)(C)ON=O",
        # "furazan24dimethanoldinitrate": "O=[N+]([O-])OCC1=COC=C1CO[N+](=O)O",
        # "nitrosobenzene": "C1=CC=C(C=C1)N=O",
        # "nitromethylbenzene": "C1=CC=C(C=C1)C[N+](=O)[O-]",
        # "dinitromethylbenzene": "C1=CC=C(C=C1)C([N+](=O)[O-])[N+](=O)[O-]",
        # "nitro1piperidine": "C1CCN(CC1)[N+](=O)[O-]",
        "nitromethane": "C[N+](=O)[O-]",
        "azido11dinitroethane": "CC(N=[N+]=[N-])([N+](=O)[O-])[N+](=O)[O-]",
        "hexanitroethane": "C(C([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-])([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]",
        "nitroglycerin": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",
        "dinitroso14piperazine": "C1CN(CCN1N=O)N=O",
        "dinitro14piperazine": "C1CN(CCN1[N+](=O)[O-])[N+](=O)[O-]",
        "nitrobenzene": "C1=CC=C(C=C1)[N+](=O)[O-]",
        "nitro2phenol": "C1=CC=C(C(=C1)[N+](=O)[O-])O",
        "nitro3phenol": "C1=CC(=CC(=C1)O)[N+](=O)[O-]",
        "nitro4phenol": "C1=CC(=CC=C1[N+](=O)[O-])O",
        "mnitroaniline": "C1=CC(=CC(=C1)[N+](=O)[O-])N",
        "pnitroaniline": "C1=CC(=CC=C1N)[N+](=O)[O-]",
        "azidobenzene": "C1=CC=C(C=C1)N=[N+]=[N-]",
        "azido1nitro4benzene": "C1=CC(=CC=C1N=[N+]=[N-])[N+](=O)[O-]",
        "methyl1nitro4benzene": "CC1=CC=C(C=C1)[N+](=O)[O-]",
        "methyl1dinitro24benzene": "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]",
        "azidomethylbenzene": "C1=CC=C(C=C1)CN=[N+]=[N-]",
        "azido3ethyl3pentane": "CCC(CC)(CC)N=[N+]=[N-]",
        "tnt": "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "dinitro22adamantane": "C1C2CC3CC1CC(C2)C3([N+](=O)[O-])[N+](=O)[O-]",
        "azido1adamantane": "C1C2CC3CC1CC(C2)(C3)N=[N+]=[N-]",
        "hns": "C1=C(C=C(C(=C1[N+](=O)[O-])/C=C/C2=C(C=C(C=C2[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
    }

    smiles_list = list(smiles_dict.values())
    id_list = list(smiles_dict.keys())

    atoms_list = create_ase_objs(smiles_list)
    results = optimise_and_write_parallel(atoms_list, smiles_list, id_list, ncores=20)
