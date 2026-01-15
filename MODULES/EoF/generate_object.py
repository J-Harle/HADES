import os
import numpy as np
import pandas as pd
import re
import urllib.request
import gc
from io import StringIO
from ase import Atoms, units
from ase.io import read, write
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import AllChem
from mace.calculators import MACECalculator
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import multiprocessing
import torch
import json  


# Suppress non-critical warnings from PyTorch and cuequivariance
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*cuequivariance.*")

# Restrict PyTorch thread usage to prevent oversubscription in parallel execution
torch.set_num_threads(1)
torch.set_num_interop_threads(1)




def read_csv(filename):
    df = pd.read_csv(filename)
    df.columns = [c.strip().upper() for c in df.columns]
    if not {"SMILES", "CID"}.issubset(df.columns):
        raise ValueError("CSV must contain 'SMILES' and 'CID' columns")
    return df[["CID", "SMILES"]]


def create_ase_objs(smiles_list, max_attempts=10):
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


def get_mace_calculator():
    parent_dir = os.getcwd()
    calc_dir = os.path.join(parent_dir, "CALCULATORS")
    os.makedirs(calc_dir, exist_ok=True)

    model_file = os.path.join(calc_dir, "MACE-OFF23_small.model")
    if not os.path.exists(model_file):
        print(f"Downloading MACE model to {model_file}")
        url = "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model?raw=true"
        urllib.request.urlretrieve(url, model_file)
        print("Download complete")

    return MACECalculator(
        model_paths=[model_file],
        dispersion=False,
        default_dtype="float64",
        device="cpu"
    )


def optimise_and_write_single(args):
    atoms, smiles, mol_id, exp_energy, flag, base_dir = args
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

    print(f"[PID {os.getpid()}] Finished optimisation for {mol_id} → {xyz_path} | SPE: {spe:.6f} eV")
    
    return {
        "ID": mol_id,
        "SMILES": smiles,
        "MACE_energy_Ha": float(spe) / 27.211386245988,
        "Exp_energy": exp_energy,
        "flag": flag,
        "xyz_path": xyz_path
    }



def optimise_and_write_parallel(atoms_list, smiles_list, id_list,
                                exp_energy_list, flag_list,
                                base_dir="OPTIMISED_STRUCTURES/",
                                ncores=38):
    tasks = [
        (atoms, smiles, mol_id, exp_energy, flag, base_dir)
        for atoms, smiles, mol_id, exp_energy, flag
        in zip(atoms_list, smiles_list, id_list, exp_energy_list, flag_list)
        if atoms is not None
    ]

    results = []
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        futures = {executor.submit(optimise_and_write_single, t): t[2] for t in tasks}
        for future in as_completed(futures):
            mol_id = futures[future]
            try:
                result = future.result()
                print(f"Optimised CID: {mol_id} → {result['xyz_path']}")
                results.append(result)
            except Exception as e:
                # keep the error message but continue with other molecules
                print(f"CID: {mol_id} failed: {repr(e)}")

        # executor.shutdown is called automatically by the context manager

    gc.collect()
    for child in multiprocessing.active_children():
        child.join(timeout=1)

    print("All optimisations finished and resources cleaned up.")
    return results


def write_json(results, output_path="eof.json"):
    json_dict = {
        entry["ID"]: {
            "SMILES": entry["SMILES"],
            "MACE_energy_Ha": entry["MACE_energy_Ha"],
            "Exp_energy": entry["Exp_energy"],
            "flag": entry["flag"]
        }
        for entry in results
    }

    with open(output_path, "w") as f:
        json.dump(json_dict, f, indent=4)

if __name__ == "__main__":

    F = "fitting"
    T = "test"
    # Example input: name : SMILES dictionary, Exp energy, flag
    molecules = {

# Byrd 2006 (In kcals/mol)
        # "1_4-dinitropiperazine": ("C1CN(CCN1[N+](=O)[O-])[N+](=O)[O-]", float(13.9 / 627.5095), F),
        # "1_4-dinitrosopiperazine": ("C1CN(CCN1N=O)N=O", float(46.4 / 627.5095), F),
        # "1_azido_4-nitrobenzene": ("[O-][N+](=O)C1=CC=C(C=C1)N=[N+]=[N-]", float(93.1 / 627.5095), F),
        # "2_4-dinitrotoluene": ("CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]", float(8.3 / 627.5095), F),
        "2-2_dinitroadamantane": ("C1C2CC3CC1CC(C2)C3([N+](=O)[O-])[N+](=O)[O-]", float(-36.88 / 627.5095), F),
        "2-azido-2-phenylpropane": ("CC(C)(N=[N+]=[N-])c1ccccc1", float(87.4 / 627.5095), F),
        "3-azido-3-ethylpentane": ("CCC(CC)(N=[N+]=[N-])CC", float(40.6 / 627.5095), F),
        # "2-nitrophenol": ("C1=CC=C(C(=C1)[N+](=O)[O-])O", float(-31.62 / 627.5095), F),        
        "3-nitrophenol": ("C1=CC(=CC(=C1)O)[N+](=O)[O-]", float(-26.12 / 627.5095), F),
        "4-nitrophenol": ("C1=CC(=CC=C1[N+](=O)[O-])O", float(-27.41 / 627.5095), F),
        "4-nitrotoluene": ("CC1=CC=C(C=C1)[N+](=O)[O-]", float(7.38 / 627.5095), F),
        "azidoadamantane": ("C1C2CC3CC1CC(C2)(C3)N=[N+]=[N-]", float(51.6 / 627.5095), F),
        "azidomethylbenzene": ("C1=CC=C(C=C1)CN=[N+]=[N-]", float(99.5 / 627.5095), F),
        "azidotrinitromethane": ("[N-]=[N+]=NC([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]", float(84.2 / 627.5095), F),
        "nitrobenzene": ("C1=CC=C(C=C1)[N+](=O)[O-]", float(62.1 / 2625.5,), F), 
        "nitroglycerine": ("C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", float(-66.71 / 627.5095), F),
        "nitromethane": ("C[N+](=O)[O-]", float(-19.3 / 627.5095), F),
        "p-nitroaniline": ("C1=CC(=CC=C1N)[N+](=O)[O-]", float(13.2 / 627.5095), F),
        "rdx": ("C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]", float(45.8 / 627.5095), F),
        "ttt": ("C1N(CN(CN1N=O)N=O)N=O", float(94.3 / 627.5095), F),
        "furazan34dimethanoldinitrate": ("C(C1=NON=C1CO[N+](=O)[O-])O[N+](=O)[O-]", float(2.6 / 627.5095), F),
        "methylnitrate": ("CO[N+](=O)[O-]", float(-29.2 / 627.5095), F),
        "nbutylnitrite": ("CCCCON=O", float(-34.8 / 627.5095), F),

        # "ethylnitrite": ("CCON=O", float(-37.0 / 627.5095), T), 
        "methylnitrite": ("CON=O", float(-15.64 / 627.5095), T),
        "1_3dimethyl2nitrobenzene": ("CC1=C(C(=CC=C1)C)[N+](=O)[O-]", float(2.1 / 627.5095), T),
        "dinitromethane": ("C([N+](=O)[O-])[N+](=O)[O-]", float(-14.1 / 627.5095), T),
        # "dinitromethylbenzene": ("C1=CC=C(C=C1)C([N+](=O)[O-])[N+](=O)[O-]", float(8.3 / 627.5095), T),
        "2methyl2nitropropane": ("CC(C)(C)[N+](=O)[O-]", float(-42.32 / 627.5095), T),
        "methylnitrobenzene": ("c1ccccc1C([N+](=O)[O-])", float(7.38 / 627.5095), T),        
        "nitrosobenzene": ("C1=CC=C(C=C1)N=O", float(48.1 / 627.5095), T),
        # "tertbutylnitrite": ("CC(C)(C)ON=O", float(-41.0/ 627.5095), T),
        "tetranitromethane": ("C([N+](=O)[O-])([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]", float(19.7 / 627.5095), T), 
        "tnt": ("CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]", float(5.75 / 627.5095), T), 
        "hns": ("C1=C(C=C(C(=C1[N+](=O)[O-])/C=C/C2=C(C=C(C=C2[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]", float(56.98 / 627.5095), T),
        "m-nitroaniline": ("C1=CC(=CC(=C1)[N+](=O)[O-])N", float(14.9 / 627.5095), T),



# === From ATCT (https://atct.anl.gov/Thermochemical%20Data/version%201.220/index.php) === 
        # Substitution.py cores
        "methane" : ("C", float(-74.513 / 2625.5), F),
        "ethyne" : ("C#C", float(228.32 / 2625.5), F),
        "propene" : ("CC=C", float(20.06 / 2625.5), F),
        "propyne" : ("CC#C", float(185.56 / 2625.5), F),
        "benzene" : ("c1ccccc1", float(83.18 / 2625.5), F),
        "toluene" : ("Cc1ccccc1", float(50.07 / 2625.5), F),
        "phenol" : ("c1ccccc1O", float(-92.94 / 2625.5), F),
        "napthalene" : ("c1ccc2ccccc2c1", float(147.68 / 2625.5), F),

        "ethane" : ("CC", float(-84.02 / 2625.5), T),
        "ethene" : ("C=C", float(52.39 / 2625.5), T),
        "propane" : ("CCC", float(-105.00 / 2625.5), T),


        # Nitroso compounds
        "fulminic_acid": ("[O-][N+]#C", float(169.29 / 2625.5), F),

        # "nitrosomethane": ("CN=O", float(71.06 / 2625.5), T),

        # Benzene derivatives
        "benzaldehyde": ("c1c(C=O)cccc1", float(-36.96 / 2625.5), F),
        "phenylethene": ("c1ccccc1C=C", float(148.56 / 2625.5), F),
        "phenylacetylene": ("c1ccccc1C#C", float(317.64 / 2625.5), F),
        "acetophenone": ("c1ccccc1C(=O)C", float(-83.90 / 2625.5), F),
        "ethylbenzene": ("c1ccccc1CC", float(29.96 / 2625.5), F),
        "anisole": ("c1ccccc1OC", float(-70.82 / 2625.5), F),
        "aniline": ("c1ccccc1N", float(86.82 / 2625.5), F),
        "biphenyl": ("c1ccccc1c2ccccc2", float(178.8 / 2625.5), F),
        "benzoic_acid": ("c1ccccc1C(=O)O", float(-294.13 / 2625.5), F),
        # "cyclohexene": ("C1CCC=CC1", float(-4.28 / 2625.5), F),
        "benzophenone": ("c1ccccc1C(=O)c2ccccc2", float(51.5 / 2625.5), F),
        "2_3dimethylbenzaldehyde": ("Cc1cc(C)ccc1C=O", float(-93.7 / 2625.5), F),

        # Nitrite
        "dinitrogen_pentoxide": ("[O-][N+](=O)O[N+]([O-])=O", float(14.79 / 2625.5,), F),
        "dioxohydrazine": ("O=NN=O", float( 171.18 / 2625.5), F),
        "hydrazine": ("NN", float(97.64 / 2625.5), F),
        "peroxynitrous_acid": ("O=N(=O)OO", float(-11.54 / 2625.5), F),
        # "propylnitrite": ("CCC[N+](=O)[O-]", float(-120.8 / 2625.5), T),
                          
        # Nitro
        "nitrosyl_hydride": ("N=O", float(106.97 / 2625.5), F),
        "nitrous_acid": ("O=NO", float(-79.113 / 2625.5), F),
        "nitrous_oxide": ("[N-]=[N+]=O", float(82.594 / 2625.5), F),
        "dioxodiazoxane": ("O=N-O-N=O", float(87.9 / 2625.5), F),
        "dinitrogen_tetroxide": ("O=N(=O)[N+](=O)[O-]", float(10.90 / 2625.5), F),
        "formaldoxime": ("C=NO", float(19.9 / 2625.5), F),
        "nitrosoamine": ("NN=O", float(76.11 / 2625.5), F),   
        # "ethylnitrate": ("CCO[N+](=O)[O-]", float(-98.69 / 2625.5), F),
        
        "n-hexane": ("CCCCCC", float(-166.75 / 2625.5), F),
        "n-pentane": ("CCCCC", float(-146.04 / 2625.5), F),
        "cyclopentane": ("C1CCCC1", float(-76.25 / 2625.5), F),
        "neopentane": ("CC(C)(C)C", float(-167.10 / 2625.5), F),
        "isopentane": ("CC(C)CC", float(-152.95 / 2625.5), F),
        "3-methylpentane": ("CCC(C)CC", float(-171.39 / 2625.5), F),
        "2-methylpentane": ("CC(C)CCC", float(-173.69 / 2625.5), F),
        "methylcyclopentane": ("C1CCC(C)C1", float(-106.12 / 2625.5), F),
        "2-2-dimethylbutane": ("CCC(C)(C)C", float(-185.69 / 2625.5), F),
        "n-heptane": ("CCCCCCC", float(-187.23 / 2625.5), F),
        # "cycloheptane": ("C1CCCCCC1", float(-117.99 / 2625.5), F),
        "n-octane": ("CCCCCCCC", float(-207.82 / 2625.5), F),

        # Toluenes
        "o-tolualdehyde": ("Cc1ccccc1C=O", float(-66.9 / 2625.5), F),
        "m-tolualdehyde": ("Cc1cccc(C=O)c1", float(-72.0 / 2625.5), F),
        "p-tolualdehyde": ("Cc1ccc(C=O)cc1", float(-72.9 / 2625.5), F),
                           
        # Amide
        "urea": ("C(N)(N)=O", float(-234.58 / 2625.5), F),
        "formamide": ("C(=O)N", float(-188.95 / 2625.5), F),
        "isocyanic_acid": ("N=C=O", float(-118.95 / 2625.5), F),
        "acetamide": ("CC(=O)N", float(-237.10 / 2625.5), F),
        "nitrosamide": ("NN=O", float(76.11 / 2625.5), F),

        # Hydroxides
        "methanol": ("CO", float(-200.92 / 2625.5), F),
        "ethanol": ("CCO", float(-235.04 / 2625.5), F),
        "water": ("O", float(-241.801 / 2625.5), F),
        # "hydrogen_peroxide": ("OO", float(-135.397 / 2625.5), F),
        "formic_acid": ("C(=O)O", float(-378.34 / 2625.5), F),
        "ethenol": ("C=CO", float(-119.67 / 2625.5), F),

        # Ester
        "methyl_formate": ("COC=O", float(-359.99 / 2625.5), F),
        # "methyl_acetate": ("CC(=O)OC", float(-382.1 / 2625.5), F),

        "ethyl_formate": ("CCOC=O", float(-393.9 / 2625.5), T),
 
        # Carboxylic acids
        "propionic_acid": ("CCC(=O)O", float(-433.1 / 2625.5), F),

        # "succinic_acid": ("OC(=O)CCC(=O)O", float(-823.7 / 2625.5), T),

    }

    id_list = list(molecules.keys())
    smiles_list = [molecules[k][0] for k in id_list]
    exp_energy_list = [molecules[k][1] for k in id_list]
    flag_list = [molecules[k][2] for k in id_list]

    atoms_list = create_ase_objs(smiles_list)

    results = optimise_and_write_parallel(
        atoms_list,
        smiles_list,
        id_list,
        exp_energy_list,
        flag_list=flag_list,
        base_dir="OPTIMISED_STRUCTURES/",
        ncores=38
    )


    write_json(results, output_path="eof.json")


    gc.collect()
    multiprocessing.active_children()
    print("All jobs complete. Exiting cleanly.")
