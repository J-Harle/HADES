import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from rdkit import Chem
from collections import Counter
import numpy as np
from sklearn.metrics import r2_score

# "NAME": "SMILES", EXPERIMENTAL ENERGY (Kcal/mol -> Ha), MACE ENERGIES (eV -> Ha), PAPER ENERGY (Ha)
# 29 mols in fitting set, 14 mols in test set
fitting_dict = { 

    "1_4-dinitropiperazine" : {
        "SMILES": "C1CN(CCN1[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 13.9 / 627.5095,
        "MACE energy": -18432.27954564 / 27.211386245988, # SMALL
        # "MACE energy": -18432.21454615 / 27.211386245988, # MEDIUM
        # "MACE energy": -18432.22057924 / 27.211386245988, # LARGE
    },

    "1_4-dinitrosopiperazine" : {
        "SMILES": "C1CN(CCN1N=O)N=O",
        "Experiment_Energy": 46.4 / 627.5095,
        "MACE energy": -14337.78974013 / 27.211386245988, # SMALL
        # "MACE energy": -14337.77507280 / 27.211386245988, # MEDIUM
        # "MACE energy": -14337.74581513 / 27.211386245988, # LARGE
    },

    "1_azido_4-nitrobenzene" : {
        "SMILES": "[O-][N+](=O)C1=CC=C(C=C1)N=[N+]=[N-]",
        "Experiment_Energy": 93.1 / 627.5095,
        "MACE energy": -16347.35684212 / 27.211386245988, # SMALL
        # "MACE energy": -16347.38932567 / 27.211386245988, # MEDIUM
        # "MACE energy": -16347.38105664 / 27.211386245988, # LARGE
    },

    "2_4-dinitrotoluene" : {
        "SMILES": "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 8.3 / 627.5095,
        "MACE energy": -18531.76437143 / 27.211386245988, # SMALL
        # "MACE energy": -18531.80614990 / 27.211386245988, # MEDIUM
        # "MACE energy": -18531.79536959 / 27.211386245988, # LARGE 
    },

    "2-2_dinitroadamantane" : {
        "SMILES": "C1C2CC3CC1CC(C2)C3([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": -36.88 / 627.5095,
        "MACE energy": -21776.71822474 / 27.211386245988, # SMALL
        # "MACE energy": -21776.75066370 / 27.211386245988, # MEDIUM
        # "MACE energy": -21776.67820297 / 27.211386245988, # LARGE
    },

    "2-azido-2-phenylpropane" : {
        "SMILES": "CC(C)(N=[N+]=[N-])c1ccccc1",
        "Experiment_Energy": 87.4 / 627.5095,
        "MACE energy": -13990.25785902 / 27.211386245988, # SMALL
        # "MACE energy": -13990.24436800 / 27.211386245988, # MEDIUM
        # "MACE energy": -13990.25179793/ 27.211386245988, # LARGE
    },

    "2-nitrophenol" : {
        "SMILES": "C1=CC=C(C(=C1)[N+](=O)[O-])O",
        "Experiment_Energy": -31.62 / 627.5095,
        "MACE energy": -13941.32814987 / 27.211386245988, # SMALL
        # "MACE energy": -13941.32303320 / 27.211386245988, # MEDIUM
        # "MACE energy": -13941.31257295 / 27.211386245988, # LARGE
    },

    "3-azido-3-ethylpentane" : {
        "SMILES": "CCC(CC)(N=[N+]=[N-])CC",
        "Experiment_Energy": 40.6 / 627.5095,
        "MACE energy": -11980.40877074 / 27.211386245988, # SMALL
        # "MACE energy": -11980.38169975 / 27.211386245988, # MEDIUM
        # "MACE energy": -11980.35961186 / 27.211386245988, # LARGE
    },

    "3-nitrophenol" : {
        "SMILES": "C1=CC(=CC(=C1)O)[N+](=O)[O-]",
        "Experiment_Energy": -26.12 / 627.5095,
        "MACE energy": -13941.14837698 / 27.211386245988, # SMALL
        # "MACE energy": -13941.16405897 / 27.211386245988, # MEDIUM
        # "MACE energy": -13941.16408860 / 27.211386245988, # LARGE
    },

    "4-nitrophenol" : {
        "SMILES": "C1=CC(=CC=C1[N+](=O)[O-])O",
        "Experiment_Energy": -27.41 / 627.5095,
        "MACE energy": -13941.21155117 / 27.211386245988, # SMALL
        # "MACE energy": -13941.21393956 / 27.211386245988, # MEDIUM
        # "MACE energy": -13941.20866514 / 27.211386245988, # LARGE
    },

    "4-nitrotoluene" : { # 1-methyl-4-nitrobenzene in paper
        "SMILES": "CC1=CC=C(C=C1)[N+](=O)[O-]",
        "Experiment_Energy": 7.38 / 627.5095,
        "MACE energy": -12963.36705153 / 27.211386245988, # SMALL
        # "MACE energy": -12963.36933490 / 27.211386245988, # MEDIUM
        # "MACE energy": -12963.36603891 / 27.211386245988, # LARGE
    },

    "azidoadamantane" : {
        "SMILES": "C1C2CC3CC1CC(C2)(C3)N=[N+]=[N-]",
        "Experiment_Energy": 51.6 / 627.5095,
        "MACE energy": -15094.40825911 / 27.211386245988, # SMALL
        # "MACE energy": -15094.34323450 / 27.211386245988, # MEDIUM
        # "MACE energy": -15094.35279234 / 27.211386245988, # LARGE
    },

    "azidobenzene" : {
        "SMILES": "C1=CC=C(C=C1)N=[N+]=[N-]",
        "Experiment_Energy": 93.0 / 627.5095,
        "MACE energy": -10778.75954720 / 27.211386245988, # SMALL
        # "MACE energy": -10778.77510022 / 27.211386245988, # MEDIUM
        # "MACE energy": -10778.76284051 / 27.211386245988, # LARGE
    },

    "azidomethylbenzene" : {
        "SMILES": "C1=CC=C(C=C1)CN=[N+]=[N-]",
        "Experiment_Energy": 99.5 / 627.5095,
        "MACE energy": -11849.08392266 / 27.211386245988, # SMALL
        # "MACE energy": -11849.07122388 / 27.211386245988, # MEDIUM
        # "MACE energy": -11849.04080786 / 27.211386245988, # LARGE
    },

    "azidotrinitromethane" : {
        "SMILES": "[N-]=[N+]=NC([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 84.2 / 627.5095,
        "MACE energy": -22261.83112985 / 27.211386245988, # SMALL
        # "MACE energy": -22261.71269254 / 27.211386245988, # MEDIUM
        # "MACE energy": -22261.64994125 / 27.211386245988, # LARGE
    },

    "dmno" : {
        "SMILES": "CNC[N+](=O)[O-]",
        "Experiment_Energy": -1.2 / 627.5095,
        "MACE energy": -9248.93904971 / 27.211386245988, # SMALL
        # "MACE energy": -9248.97452066 / 27.211386245988, # MEDIUM
        # "MACE energy": -9248.99317450 / 27.211386245988, # LARGE 
    },

    "dnpn" : {
        "SMILES": "CC(CN(CC(C)([N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": -31.7 / 627.5095,
        "MACE energy": -35804.52712936 / 27.211386245988, # SMALL
        # "MACE energy": -35804.36243221 / 27.211386245988, # MEDIUM
        # "MACE energy": -35804.50079690 / 27.211386245988, # LARGE
    },

    "hexanitroethane" : {
        "SMILES": "C(C([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-])([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 42.8 / 627.5095,
        "MACE energy": -35581.35362902 / 27.211386245988, # SMALL
        # "MACE energy": -35581.35398690 / 27.211386245988, # MEDIUM
        # "MACE energy": -35581.35016983 / 27.211386245988, # LARGE
    },

    "hns" : {
        "SMILES": "C1=C(C=C(C(=C1[N+](=O)[O-])/C=C/C2=C(C=C(C=C2[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 56.98 / 627.5095,
        "MACE energy": -48133.84497581 / 27.211386245988, # SMALL
        # "MACE energy": -48134.03278402 / 27.211386245988, # MEDIUM
        # "MACE energy": -48134.01260584 / 27.211386245988, # LARGE
    },

    "m-nitroaniline" : {
        "SMILES": "C1=CC(=CC(=C1)[N+](=O)[O-])N",
        "Experiment_Energy": 14.9 / 627.5095,
        "MACE energy": -13400.30744681 / 27.211386245988, # SMALL
        # "MACE energy": -13400.31265026 / 27.211386245988, # MEDIUM
        # "MACE energy": -13400.31309365 / 27.211386245988, # LARGE
    },

    "n-nitrobis2_2_2-trinitroethylamine" : {
        "SMILES": "O=N(=O)NCCNN(=O)=O",
        "Experiment_Energy": 21.42 / 627.5095,
        "MACE energy": -16324.43917466 / 27.211386245988, # SMALL
        # "MACE energy": -16324.42305633 / 27.211386245988, # MEDIUM
        # "MACE energy": -16324.41992595 / 27.211386245988, # LARGE
    },

    "nitrobenzene" : {
        "SMILES": "C1=CC=C(C=C1)[N+](=O)[O-]",
        "Experiment_Energy": 16.38 / 627.5095,
        "MACE energy": -11892.76308853 / 27.211386245988, # SMALL
        # "MACE energy": -11892.77009512 / 27.211386245988, # MEDIUM
        # "MACE energy": -11892.77026402 / 27.211386245988, # LARGE
    },

    "nitroglycerine" : {
        "SMILES": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",
        "Experiment_Energy": -66.71 / 627.5095,
        "MACE energy": -26090.87382331 / 27.211386245988, # SMALL
        # "MACE energy": -26090.85788416 / 27.211386245988, # MEDIUM
        # "MACE energy": -26090.91578679 / 27.211386245988, # LARGE
    },

    "nitromethane" : {
        "SMILES": "C[N+](=O)[O-]",
        "Experiment_Energy": -19.3 / 627.5095,
        "MACE energy": -6671.50635367 / 27.211386245988, # SMALL
        # "MACE energy": -6671.53523402 / 27.211386245988, # MEDIUM
        # "MACE energy": -6671.53647289 / 27.211386245988, # LARGE
    },

    "p-nitroaniline" : {
        "SMILES": "C1=CC(=CC=C1N)[N+](=O)[O-]",
        "Experiment_Energy": 13.2 / 627.5095,
        "MACE energy": -13400.36662025 / 27.211386245988, # SMALL
        # "MACE energy": -13400.38312690 / 27.211386245988, # MEDIUM
        # "MACE energy": -13400.37316586 / 27.211386245988, # LARGE
    },

    "rdx" : {
        "SMILES": "C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 45.8 / 627.5095,
        "MACE energy": -24437.11933344 / 27.211386245988, # SMALL
        # "MACE energy": -24436.67568444 / 27.211386245988, # MEDIUM
        # "MACE energy": -24436.90237240 / 27.211386245988, # LARGE
    },

    "tetranitromethane" : {
        "SMILES": "C([N+](=O)[O-])([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 19.7 / 627.5095,
        "MACE energy": -23374.80013130 / 27.211386245988, # SMALL
        # "MACE energy": -23374.74363261 / 27.211386245988, # MEDIUM
        # "MACE energy": -23374.87738962 / 27.211386245988, # LARGE 
    },

    "tnt" : {
        "SMILES": "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 5.75 / 627.5095,
        "MACE energy": -24100.03522379 / 27.211386245988, # SMALL
        # "MACE energy": -24100.07703916 / 27.211386245988, # MEDIUM
        # "MACE energy": -24100.04023681 / 27.211386245988, # LARGE
    },

    "ttt" : {
        "SMILES": "C1N(CN(CN1N=O)N=O)N=O",
        "Experiment_Energy": 94.3 / 627.5095,
        "MACE energy": -18295.03707206 / 27.211386245988, # SMALL
        # "MACE energy": -18295.01274327 / 27.211386245988, # MEDIUM
        # "MACE energy": -18295.05410931 / 27.211386245988, # LARGE
    },

    "furazan34dimethanoldinitrate" : {
        "SMILES": "C(C1=NON=C1CO[N+](=O)[O-])O[N+](=O)[O-]",
        "Experiment_Energy": 2.6 / 627.5095,
        # "MACE energy": -24508.03652046 / 27.211386245988, # SMALL
        # "MACE energy": -24507.86659948 / 27.211386245988, # MEDIUM
        "MACE energy": -24507.97131620 / 27.211386245988, # LARGE
    },

}



test_dict = {

    "1_3dimethyl2nitrobenzene" : {
        "SMILES": "CC1=C(C(=CC=C1)C)[N+](=O)[O-]",
        "Experiment_Energy": 2.1 / 627.5095,
        # "MACE energy": -14033.77625063 / 27.211386245988, # SMALL
        "MACE energy": -14033.63278412 / 27.211386245988, # MEDIUM
        # "MACE energy": -14033.61713095 / 27.211386245988, # LARGE
    },

    "dinitromethane" : {
        "SMILES": "C([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": -14.1 / 627.5095,
        # "MACE energy": -12239.60258122 / 27.211386245988, # SMALL
        "MACE energy": -12239.61913223 / 27.211386245988, # MEDIUM
        # "MACE energy": -12239.62267937 / 27.211386245988, # LARGE
    },

    "dinitromethylbenzene" : {
        "SMILES": "C1=CC=C(C=C1)C([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 8.3 / 627.5095,
        # "MACE energy": -18531.39293788 / 27.211386245988, # SMALL
        "MACE energy": -18531.43304428 / 27.211386245988, # MEDIUM
        # "MACE energy": -18531.42580448 / 27.211386245988, # LARGE
    },

    "ethylnitrate" : {
        "SMILES": "CCO[N+](=O)[O-]",
        "Experiment_Energy": -25.9 / 627.5095,
        # "MACE energy": -9789.23531244 / 27.211386245988, # SMALL
        "MACE energy": -9789.23505542 / 27.211386245988, # MEDIUM
        # "MACE energy": -9789.26673278 / 27.211386245988, # LARGE
    },

    "ethylnitrite" : {
        "SMILES": "CCON=O",
        "Experiment_Energy": -37.0 / 627.5095,
        # "MACE energy": -7741.90209714 / 27.211386245988, # SMALL
        "MACE energy": -7742.01150896 / 27.211386245988, # MEDIUM
        # "MACE energy": -7742.01470880 / 27.211386245988, # LARGE
    },

    "2methyl2nitropropane" : {
        "SMILES": "CC(C)(C)[N+](=O)[O-]",
        "Experiment_Energy": -42.32 / 627.5095,
        # "MACE energy": -9883.32776292 / 27.211386245988, # SMALL
        "MACE energy": -9883.29251470 / 27.211386245988, # MEDIUM
        # "MACE energy": -9883.32101707 / 27.211386245988, # LARGE
    },

    "methylnitrate" : {
        "SMILES": "CO[N+](=O)[O-]",
        "Experiment_Energy": -29.2 / 627.5095,
        # "MACE energy": -8718.63615026 / 27.211386245988, # SMALL
        "MACE energy": -8718.63486708 / 27.211386245988, # MEDIUM
        # "MACE energy": -8718.67110647 / 27.211386245988, # LARGE
    },

    "methylnitrite" : {
        "SMILES": "CON=O",
        "Experiment_Energy": -15.64 / 627.5095,
        # "MACE energy": -6671.32812737 / 27.211386245988, # SMALL
        "MACE energy": -6671.45085405 / 27.211386245988, # MEDIUM
        # "MACE energy": -6671.41629495 / 27.211386245988, # LARGE
    },

    "methylnitrobenzene" : {
        "SMILES": "c1ccccc1C([N+](=O)[O-])",
        "Experiment_Energy": 7.38 / 627.5095,
        # "MACE energy": -12963.24084391 / 27.211386245988, # SMALL
        "MACE energy": -12963.27153977 / 27.211386245988, # MEDIUM
        # "MACE energy": -12963.26666778 / 27.211386245988, # LARGE
    },

    "nbutylnitrite" : {
        "SMILES": "CCCCON=O",
        "Experiment_Energy": -34.8 / 627.5095,
        # "MACE energy": -9882.84097443 / 27.211386245988, # SMALL
        "MACE energy": -9882.96109645 / 27.211386245988, # MEDIUM
        # "MACE energy": -9882.90924521 / 27.211386245988, # LARGE
    },

    "1nitropiperidine" : {
        "SMILES": "O=NN1CCCCC1",
        "Experiment_Energy": -10.6 / 627.5095,
        # "MACE energy": -10380.24299052 / 27.211386245988, # SMALL
        "MACE energy": -10380.23132950 / 27.211386245988, # MEDIUM
        # "MACE energy": -10380.22238569 / 27.211386245988, # LARGE
    },

    "nitrosobenzene" : {
        "SMILES": "C1=CC=C(C=C1)N=O",
        "Experiment_Energy": 48.1 / 627.5095,
        # "MACE energy": -9844.75581343 / 27.211386245988, # SMALL
        "MACE energy": -9844.77533103 / 27.211386245988, # MEDIUM
        # "MACE energy": -9844.80180181 / 27.211386245988, # LARGE
    },

    "propylnitrite" : {
        "SMILES": "CCCO[N+](=O)[O-]",
        "Experiment_Energy": -28.4 / 627.5095,
        # "MACE energy": -10859.71196604 / 27.211386245988, # SMALL
        "MACE energy": -10859.71606568 / 27.211386245988, # MEDIUM
        # "MACE energy": -10859.74379917 / 27.211386245988, # LARGE
    },

    "tertbutylnitrite" : {
        "SMILES": "CC(C)(C)ON=O",
        "Experiment_Energy": -41.0/ 627.5095,
        # "MACE energy": -9882.97171893 / 27.211386245988, # SMALL
        "MACE energy": -9883.12009522 / 27.211386245988, # MEDIUM
        # "MACE energy": -9883.07147083 / 27.211386245988, # LARGE
    },

}

# === Dynamic atom and bond label generation ===
def get_dynamic_labels(*smiles_dicts):
    atom_labels = set()
    bond_labels = set()

    for smiles_dict in smiles_dicts:
        for name, data in smiles_dict.items():
            smiles = data["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: could not parse SMILES for {name}")
                continue
            mol = Chem.AddHs(mol)

            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if atom.GetIsAromatic():
                    label = f"{symbol}_arom" if symbol in ["C", "N", "O"] else symbol
                else:
                    label = symbol
                atom_labels.add(label)

            for bond in mol.GetBonds():
                atoms = sorted([
                    bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol()
                ])
                bond_type = str(bond.GetBondType())
                bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
                bond_labels.add(bond_label)

    return sorted(atom_labels), sorted(bond_labels)


# === Feature vector builder ===
def build_feature_vectors(SMILES_dict, atom_labels, bond_labels):
    feature_vectors = {}
    atom_counts_dict = {}

    for name, data in SMILES_dict.items():
        smiles = data["SMILES"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: could not parse SMILES for {name}")
            continue
        mol = Chem.AddHs(mol)

        atom_counter = Counter()
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if atom.GetIsAromatic():
                label = f"{symbol}_arom" if symbol in ["C", "N", "O"] else symbol
            else:
                label = symbol
            atom_counter[label] += 1

        bond_counter = Counter()
        for bond in mol.GetBonds():
            atoms = sorted([
                bond.GetBeginAtom().GetSymbol(),
                bond.GetEndAtom().GetSymbol()
            ])
            bond_type = str(bond.GetBondType())
            bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
            bond_counter[bond_label] += 1

        simple_atom_counter = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
        atom_counts_dict[name] = dict(simple_atom_counter)

        atom_vector = [atom_counter.get(a, 0) for a in atom_labels]
        bond_vector = [bond_counter.get(b, 0) for b in bond_labels]
        feature_vectors[name] = atom_vector + bond_vector

    return feature_vectors, atom_counts_dict


def fit_atomic_bond_contributions(fitting_dict, atom_labels, bond_labels, show_plot=True):
    """
    Fit a linear model to the difference between MACE and experimental ΔHf values
    using counts of atom and bond types as features.
    Optionally plot predicted vs experimental ΔHf for the fitting set.
    """

    names, X, y_corr, y_exp, y_mace = [], [], [], [], []

    for name, data in fitting_dict.items():
        try:
            mol = Chem.MolFromSmiles(data["SMILES"])
            if mol is None:
                print(f"Could not parse SMILES for {name}. Skipping.")
                continue

            mol = Chem.AddHs(mol)

            # Count atoms
            atom_counts = np.zeros(len(atom_labels))
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if atom.GetIsAromatic():
                    label = f"{symbol}_arom" if symbol in ["C", "N", "O"] else symbol
                else:
                    label = symbol
                if label in atom_labels:
                    atom_counts[atom_labels.index(label)] += 1

            # Count bonds
            bond_counts = np.zeros(len(bond_labels))
            for bond in mol.GetBonds():
                atoms = sorted([
                    bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol()
                ])
                bond_type = str(bond.GetBondType())
                bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
                if bond_label in bond_labels:
                    bond_counts[bond_labels.index(bond_label)] += 1

            features = np.concatenate([atom_counts, bond_counts])
            X.append(features)

            mace_e = data["MACE energy"]
            exp_e = data["Experiment_Energy"]
            y_corr.append(mace_e - exp_e)
            y_exp.append(exp_e)
            y_mace.append(mace_e)
            names.append(name)

        except Exception as e:
            print(f"Error processing {name}: {e}")

    X = np.array(X)
    y_corr = np.array(y_corr)
    y_exp = np.array(y_exp)
    y_mace = np.array(y_mace)

    if X.shape[0] == 0:
        raise ValueError("No valid data found in fitting_dict.")

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y_corr)
    coeffs = model.coef_
    intercept = model.intercept_

    # Predicted correction and ΔHf
    y_pred_corr = model.predict(X)
    y_pred_exp = y_mace - y_pred_corr

    r2_fit = r2_score(y_exp, y_pred_exp)

    print("\n===== Linear Fit Summary =====")
    print(f"R² (fit) = {r2_fit:.4f}")
    print(f"Intercept = {intercept:.6f} Ha")
    print("\nCoefficients:")
    for i, label in enumerate(atom_labels + bond_labels):
        print(f"{label:>8s}: {coeffs[i]: .6f} Ha per unit")

    # ===== Plotting =====
    if show_plot:
        errors = y_pred_exp - y_exp  # Prediction error

        fig, ax1 = plt.subplots(figsize=(10, 10))
        ax1.scatter(y_exp, y_pred_exp, c='black', s=50, alpha=0.7, label="Predicted vs Exp")
        lims = [
            min(np.min(y_exp), np.min(y_pred_exp)),
            max(np.max(y_exp), np.max(y_pred_exp))
        ]
        ax1.plot(lims, lims, 'k--', linewidth=1.5, label="y = x (perfect prediction)")
        ax1.set_xlabel("Experimental EoF (Ha)", fontsize=12)
        ax1.set_ylabel("Predicted EoF (Ha)", fontsize=12)
        ax1.set_title(f"Fitting Set (Large Model)\nR² = {r2_fit:.4f}", fontsize=13)
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.grid(alpha=0.3)

        # Second y-axis for errors (points only)
        ax2 = ax1.twinx()
        ax2.scatter(y_exp, errors, c='red', s=40, alpha=0.8, label="Prediction Error")
        ax2.set_ylabel("Prediction Error (Ha)", color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')

        # Annotate main scatter points
        for i, n in enumerate(names):
            ax1.text(y_exp[i], y_pred_exp[i], n, fontsize=8, ha='right', va='bottom', alpha=0.7)

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

        plt.tight_layout()
        plt.show()



        # Annotate molecules
        for i, n in enumerate(names):
            plt.text(y_exp[i], y_pred_exp[i], n, fontsize=8, ha='right', va='bottom', alpha=0.7)

        plt.tight_layout()
        plt.show()

    # Contributions per molecule
    contributions = {n: c for n, c in zip(names, y_pred_corr)}

    return coeffs, contributions, y_pred_exp, r2_fit


import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.metrics import r2_score

def predict_test_set(test_dict, atom_labels, bond_labels, coeffs, intercept=0.0, show_plot=True):
    """
    Apply fitted atomic/bond coefficients to predict experimental ΔHf for a test set,
    and optionally plot predicted vs experimental energies.
    """

    names, X_test, y_mace, y_exp = [], [], [], []

    for name, data in test_dict.items():
        try:
            mol = Chem.MolFromSmiles(data["SMILES"])
            if mol is None:
                print(f"Could not parse SMILES for {name}. Skipping.")
                continue

            mol = Chem.AddHs(mol)

            atom_counts = np.zeros(len(atom_labels))
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if atom.GetIsAromatic():
                    label = f"{symbol}_arom" if symbol in ["C", "N", "O"] else symbol
                else:
                    label = symbol
                if label in atom_labels:
                    atom_counts[atom_labels.index(label)] += 1

            bond_counts = np.zeros(len(bond_labels))
            for bond in mol.GetBonds():
                atoms = sorted([
                    bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol()
                ])
                bond_type = str(bond.GetBondType())
                bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
                if bond_label in bond_labels:
                    bond_counts[bond_labels.index(bond_label)] += 1

            features = np.concatenate([atom_counts, bond_counts])
            X_test.append(features)
            y_mace.append(data["MACE energy"])
            y_exp.append(data["Experiment_Energy"])
            names.append(name)

        except Exception as e:
            print(f"Error processing {name}: {e}")

    X_test = np.array(X_test)
    y_mace = np.array(y_mace)
    y_exp = np.array(y_exp)

    expected_len = len(atom_labels) + len(bond_labels)
    if X_test.shape[1] != expected_len:
        raise ValueError(
            f"Feature length mismatch: expected {expected_len}, got {X_test.shape[1]}. "
            "Ensure atom_labels and bond_labels are identical to those used for fitting."
        )

    y_corr_pred = X_test @ coeffs + intercept
    y_pred_exp = y_mace - y_corr_pred

    r2_test = r2_score(y_exp, y_pred_exp)

    print("\n===== Test Set Performance =====")
    print(f"R² (test) = {r2_test:.4f}")
    for n, y_p, y_e in zip(names, y_pred_exp, y_exp):
        print(f"{n:25s}  Pred: {y_p: .6f}   Exp: {y_e: .6f}   Δ = {y_p - y_e: .6f}")

    # ===== Plotting =====
    if show_plot:
        errors = y_pred_exp - y_exp  # Prediction error

        fig, ax1 = plt.subplots(figsize=(10, 10))
        ax1.scatter(y_exp, y_pred_exp, c='black', s=50, alpha=0.7,label="Predicted vs Exp")
        lims = [
            min(np.min(y_exp), np.min(y_pred_exp)),
            max(np.max(y_exp), np.max(y_pred_exp))
        ]
        ax1.plot(lims, lims, 'k--', linewidth=1.5, label="y = x (perfect prediction)")
        ax1.set_xlabel("Experimental EoHf (Ha)", fontsize=12)
        ax1.set_ylabel("Predicted EoHf (Ha)", fontsize=12)
        ax1.set_title(f"Test Set (Large Model)\nR² = {r2_test:.4f}", fontsize=13)
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.grid(alpha=0.3)

        # Second y-axis for errors (points only)
        ax2 = ax1.twinx()
        ax2.scatter(y_exp, errors, c='red', s=40, alpha=0.8, label="Prediction Error")
        ax2.set_ylabel("Prediction Error (Ha)", color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')

        # Annotate main scatter points
        for i, n in enumerate(names):
            ax1.text(y_exp[i], y_pred_exp[i], n, fontsize=8, ha='right', va='bottom', alpha=0.7)

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

        plt.tight_layout()
        plt.show()


    return y_pred_exp, r2_test



# === Run everything ===
atom_labels, bond_labels = get_dynamic_labels(fitting_dict, test_dict)
coeffs, contributions, y_pred, r2_fit = fit_atomic_bond_contributions(fitting_dict, atom_labels, bond_labels)
y_pred_test, r2_test = predict_test_set(test_dict, atom_labels, bond_labels, coeffs)