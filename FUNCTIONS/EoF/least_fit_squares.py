import matplotlib.pyplot as plt
from scipy.stats import linregress
from rdkit import Chem
from collections import Counter
import numpy as np

# "NAME": "SMILES", EXPERIMENTAL ENERGY (Kcal/mol -> Ha), MACE ENERGIES (eV -> Ha), PAPER ENERGY (Ha)
fitting_dict = {
    "nitromethane": {
        "SMILES": "C[N+](=O)[O-]",
        "Experiment_Energy": -19.3 / 627.5095,
        "MACE energy": -6671.50648846 / 27.211386245988, # SMALL
        # "MACE energy": -6671.53561703 / 27.211386245988, # MEDIUM
        # "MACE energy": -6671.53681134 / 27.211386245988, # LARGE
        # "Paper energy": 245.1064954,
    },
    "azido11dinitroethane": {
        "SMILES": "CC(N=[N+]=[N-])([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 60.4 / 627.5095,
        "MACE energy": -17764.70543565 / 27.211386245988, # SMALL
        # "MACE energy": -17764.68747379 / 27.211386245988, # MEDIUM
        # "MACE energy": -17764.66301768 / 27.211386245988, # LARGE
        # "Paper energy": -652.6296314,
        
    },
    "hexanitroethane": {
        "SMILES": "C(C([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-])([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 42.8 / 627.5095,
        "MACE energy": -35581.35362901 / 27.211386245988, # SMALL
        # "MACE energy": -35581.38938505 / 27.211386245988, # MEDIUM
        # "MACE energy": -35581.42930880 / 27.211386245988, # LARGE
        "Paper energy": -1307.1561375,

    },
    "nitroglycerin": {
        "SMILES": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",
        "Experiment_Energy": -66.71 / 627.5095,
        "MACE energy": -26091.01398203 / 27.211386245988, # SMALL
        # "MACE energy": -26090.96992151 / 27.211386245988, # MEDIUM
        # "MACE energy": -26091.05035830 / 27.211386245988, # LARGE
        "Paper energy": -958.5266091,

    },
    "dinitroso14piperazine": {
        "SMILES": "C1CN(CCN1N=O)N=O",
        "Experiment_Energy": 46.4 / 627.5095,
        "MACE energy": -14337.78974013 / 27.211386245988, # SMALL
        # "MACE energy": -14337.77507288 / 27.211386245988, # MEDIUM
        # "MACE energy": -14337.74581524 / 27.211386245988, # LARGE
        "Paper energy": -526.7255232,

    },
    "dinitro14piperazine": {
        "SMILES": "C1CN(CCN1[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 13.9 / 627.5095,
        "MACE energy": -18432.29259350 / 27.211386245988, # SMALL
        # "MACE energy": -18432.21454611 / 27.211386245988, # MEDIUM
        # "MACE energy": -18432.19973047 / 27.211386245988, # LARGE
        "Paper energy": -677.1533702,
    },

    "nitrobenzene": {
        "SMILES": "C1=CC=C(C=C1)[N+](=O)[O-]",
        "Experiment_Energy": 16.38 / 627.5095,
        "MACE energy": -11892.76308846 / 27.211386245988, # SMALL
        # "MACE energy": -11892.77009503 / 27.211386245988, # MEDIUM
        # "MACE energy": -11892.77026396 / 27.211386245988, # LARGE
        "Paper energy": -436.9026044,
    },

    "nitro2phenol": {
        "SMILES": "C1=CC=C(C(=C1)[N+](=O)[O-])O",
        "Experiment_Energy": -31.62 / 627.5095,
        "MACE energy": -13941.32814988 / 27.211386245988, # SMALL
        # "MACE energy": -13941.32303331 / 27.211386245988, # MEDIUM
        # "MACE energy": -13941.31257310 / 27.211386245988, # LARGE
        "Paper energy": -512.1616809,
    },

    "nitro3phenol": {
        "SMILES": "C1=CC(=CC(=C1)O)[N+](=O)[O-]",
        "Experiment_Energy": -26.12 / 627.5095,
        "MACE energy": -13941.14837621 / 27.211386245988, # SMALL
        # "MACE energy": -13941.16405913 / 27.211386245988, # MEDIUM
        # "MACE energy": -13941.16408853 / 27.211386245988, # LARGE
        "Paper energy": -512.1535723,
    },

    # "nitro4phenol": {
    #     "SMILES": "C1=CC(=CC=C1[N+](=O)[O-])O",
    #     "Experiment_Energy": 27.41 / 627.5095,
    #     "MACE energy": -13941.21155085 / 27.211386245988, # SMALL
    #     # "MACE energy": -13941.21393941 / 27.211386245988, # MEDIUM
    #     # "MACE energy": -13941.20866497 / 27.211386245988, # LARGE
    #     "Paper energy": -512.1567977,
    # },

    "mnitroaniline": {
        "SMILES": "C1=CC(=CC(=C1)[N+](=O)[O-])N",
        "Experiment_Energy": 14.9 / 627.5095,
        "MACE energy": -13400.30744689 / 27.211386245988, # SMALL
        # "MACE energy": -13400.31264451 / 27.211386245988, # MEDIUM
        # "MACE energy": -13400.31309819 / 27.211386245988, # LARGE
        "Paper energy": -492.2836904,
    },
    
    "pnitroaniline": {
        "SMILES": "C1=CC(=CC=C1N)[N+](=O)[O-]",
        "Experiment_Energy": 13.2 / 627.5095,
        "MACE energy": -13400.36662051 / 27.211386245988, # SMALL
        # "MACE energy": -13400.38312772 / 27.211386245988, # MEDIUM
        # "MACE energy": -13400.37316552 / 27.211386245988, # LARGE
        "Paper energy": -492.2876859,
    },

    "azidobenzene": {
        "SMILES": "C1=CC=C(C=C1)N=[N+]=[N-]",
        "Experiment_Energy": 93.0 / 627.5095,
        "MACE energy": -10778.75954697 / 27.211386245988, # SMALL
        # "MACE energy": -10778.77510060 / 27.211386245988, # MEDIUM
        # "MACE energy": -10778.76284046 / 27.211386245988, # LARGE
        "Paper energy": -395.9704254,
    },

    "azido1nitro4benzene": {
        "SMILES": "C1=CC(=CC=C1N=[N+]=[N-])[N+](=O)[O-]",
        "Experiment_Energy": 93.1 / 627.5095,
        "MACE energy": -16347.35684216 / 27.211386245988, # SMALL
        # "MACE energy": -16347.38932583 / 27.211386245988, # MEDIUM
        # "MACE energy": -16347.38105695 / 27.211386245988, # LARGE
        "Paper energy": -600.5459965,
    },
    "methyl1nitro4benzene": {
        "SMILES": "CC1=CC=C(C=C1)[N+](=O)[O-]",
        "Experiment_Energy": 7.38 / 627.5095,
        "MACE energy": -12963.36704287 / 27.211386245988, # SMALL
        # "MACE energy": -12963.36932102 / 27.211386245988, # MEDIUM
        # "MACE energy": -12963.36605582 / 27.211386245988, # LARGE
        "Paper energy": -476.2339492,
    },

    "methyl1dinitro24benzene": {
        "SMILES": "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 7.93 / 627.5095,
        "MACE energy": -18531.76437310 / 27.211386245988, # SMALL
        # "MACE energy": -18531.80614829 / 27.211386245988, # MEDIUM
        # "MACE energy": -18531.79536728 / 27.211386245988, # LARGE
        "Paper energy": -680.7997820,
    },

    "azidomethylbenzene": {
        "SMILES": "C1=CC=C(C=C1)CN=[N+]=[N-]",
        "Experiment_Energy": 99.5 / 627.5095,
        "MACE energy": -11849.12752812 / 27.211386245988, # SMALL
        # "MACE energy": -11849.11935131 / 27.211386245988, # MEDIUM
        # "MACE energy": -11849.09884166 / 27.211386245988, # LARGE
        "Paper energy": -435.2904566,
    },

    "azido3ethyl3pentane": {
        "SMILES": "CCC(CC)(CC)N=[N+]=[N-]",
        "Experiment_Energy": 40.6 / 627.5095,
        "MACE energy": -11980.37260140 / 27.211386245988, # SMALL
        # "MACE energy": -11980.35881750 / 27.211386245988, # MEDIUM
        # "MACE energy": -11980.33411541 / 27.211386245988, # LARGE
        "Paper energy": -440.1288955,
    },

    "tnt": {
        "SMILES": "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 5.75 / 627.5095,
        "MACE energy": -24100.00763976 / 27.211386245988, # SMALL
        # "MACE energy": -24100.03032071 / 27.211386245988, # MEDIUM
        # "MACE energy": -24100.00418762 / 27.211386245988, # LARGE
        "Paper energy": -885.3582522,
    },
    
    "dinitro22adamantane": {
        "SMILES": "C1C2CC3CC1CC(C2)C3([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": -36.88 / 627.5095,
        "MACE energy": -21776.71822500 / 27.211386245988, # SMALL
        # "MACE energy": -21776.75066348 / 27.211386245988, # MEDIUM
        # "MACE energy": -21776.67820248 / 27.211386245988, # LARGE  
        "Paper energy": -799.9797733,
    },

    "azido1adamantane": {
        "SMILES": "C1C2CC3CC1CC(C2)(C3)N=[N+]=[N-]",
        "Experiment_Energy": 51.6 / 627.5095,
        "MACE energy": -15094.40825930 / 27.211386245988, # SMALL
        # "MACE energy": -15094.34323451 / 27.211386245988, # MEDIUM
        # "MACE energy": -15094.35279210 / 27.211386245988, # LARGE 
        "Paper energy": -554.4870509,
    },

    "hns": {
        "SMILES": "C1=C(C=C(C(=C1[N+](=O)[O-])/C=C/C2=C(C=C(C=C2[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 56.98 / 627.5095,
        "MACE energy": -48133.88855310 / 27.211386245988, # SMALL
        # "MACE energy": -48133.87041827 / 27.211386245988, # MEDIUM
        # "MACE energy": -48133.88260023 / 27.211386245988, # LARGE
        "Paper energy": -1768.2654374,
    },
}


test_dict = {
        "methylnitrite": {
        "SMILES": "CON=O",
        "Experiment_Energy": -15.64 / 627.5095,
        "MACE energy": -6671.32812732 / 27.211386245988, # SMALL
        # "MACE energy": -6671.45085405 / 27.211386245988, # MEDIUM
        # "MACE energy": -6671.41629503 / 27.211386245988, # LARGE
        "Paper energy": -245.1002638,
        },

        "methylnitrate": {
        "SMILES": "CO[N+](=O)[O-]",
        "Experiment_Energy": -29.2 / 627.5095,
        "MACE energy": -8718.63615021 / 27.211386245988, # SMALL
        # "MACE energy": -8718.63486707 / 27.211386245988, # MEDIUM
        # "MACE energy": -8718.67110644 / 27.211386245988, # LARGE
        "Paper energy": -320.3135279,
        },

        "dinitromethane": {
        "SMILES": "C([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": -14.1 / 627.5095,
        "MACE energy": -12239.60258108 / 27.211386245988, # SMALL
        # "MACE energy": -12239.61913241 / 27.211386245988, # MEDIUM
        # "MACE energy": -12239.62267937 / 27.211386245988, # LARGE
        "Paper energy": -449.6606863,
        },

        "ethylnitrite":{
        "SMILES": "CCON=O",
        "Experiment_Energy": -25.9 / 627.5095, 
        "MACE energy": -9789.20514177 / 27.211386245988, # SMALL
        # "MACE energy": -9789.23086211 / 27.211386245988, # MEDIUM
        # "MACE energy": -9789.26453612 / 27.211386245988, # LARGE
        "Paper energy": -284.4316497,
        },

        "methyl2nitro2propane": {
        "SMILES": "CC(C)(C)[N+](=O)[O-]",
        "Experiment_Energy": -42.79 / 627.5095,
        "MACE energy": -9883.32776301 / 27.211386245988, # SMALL
        # "MACE energy": -9883.29252853 / 27.211386245988, # MEDIUM
        # "MACE energy": -9883.32101723 / 27.211386245988, # LARGE
        "Paper energy": -363.0946801,
        },

        "nbutylnitrite": {
        "SMILES": "CCCCON=O",
        "Experiment_Energy": -34.8 / 627.5095,
        "MACE energy": -9882.85196516 / 27.211386245988, # SMALL
        # "MACE energy": -9882.95566221 / 27.211386245988, # MEDIUM
        # "MACE energy": -9882.94801119 / 27.211386245988, # LARGE
        "Paper energy": -363.0848109
        },

        "nitrosobenzene":{
        "SMILES": "C1=CC=C(C=C1)N=O",
        "Experiment_Energy": 48.1 / 627.5095,
        "MACE energy": -9844.75581342 / 27.211386245988, # SMALL
        # "MACE energy": -9844.77533086 / 27.211386245988, # MEDIUM
        # "MACE energy": -9844.80180183 / 27.211386245988, # LARGE
        "Paper energy": -566.2201576
        },

        "nitromethylbenzene":{
        "SMILES": "C1=CC=C(C=C1)C[N+](=O)[O-]",
        "Experiment_Energy": 7.34 / 627.5095,
        "MACE energy": -12963.24083556 / 27.211386245988, # SMALL
        # "MACE energy": -12963.27156823 / 27.211386245988, # MEDIUM
        # "MACE energy": -12963.26666729 / 27.211386245988, # LARGE
        # "Paper energy": -476.2290336,
        },

        "dinitromethylbenzene": {
        "SMILES": "C1=CC=C(C=C1)C([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 8.3 / 627.5095,
        "MACE energy": -18531.39293741 / 27.211386245988, # SMALL
        # "MACE energy": -18531.43304492 / 27.211386245988, # MEDIUM
        # "MACE energy": -18531.42580645 / 27.211386245988, # LARGE
        "Paper energy": -680.7851161
        },
}

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


# === Example feature labels ===
atom_labels = ["C", "H", "N", "O", "C_arom", "N_arom", "O_arom"]
bond_labels = [
    "C - C SINGLE", "C - C DOUBLE", "C - N SINGLE", "C - O SINGLE",
    "C - N DOUBLE", "N - O SINGLE", "N - O DOUBLE"
]


# === Build design matrix and perform least squares fit ===
def fit_atomic_bond_contributions(fitting_dict, atom_labels, bond_labels):
    feature_vectors, _ = build_feature_vectors(fitting_dict, atom_labels, bond_labels)

    X = []
    y = []

    for name, vec in feature_vectors.items():
        X.append(vec)
        y.append(fitting_dict[name]["Experiment_Energy"])

    X = np.array(X)
    y = np.array(y)

    # Add bias term (intercept)
    X_aug = np.column_stack((np.ones(X.shape[0]), X))

    # Solve least squares: w = (X^T X)^(-1) X^T y
    coeffs, residuals, rank, s = np.linalg.lstsq(X_aug, y, rcond=None)

    intercept = coeffs[0]
    contributions = coeffs[1:]

    # Predicted values
    y_pred = X_aug @ coeffs

    # Correlation statistics
    slope, intercept_r, r_value, p_value, std_err = linregress(y, y_pred)

    print(f"R² = {r_value**2:.4f}")
    print(f"Intercept (bias): {intercept:.6f} Ha")

    # Print contributions
    print("\n=== Atomic/Bond Contributions (Ha) ===")
    for label, val in zip(atom_labels + bond_labels, contributions):
        print(f"{label:20s}: {val: .6f}")

    # Print per-molecule results for fitting set
    print("\n=== Fitting Set Predictions (Ha) ===")
    for name, exp, pred in zip(feature_vectors.keys(), y, y_pred):
        print(f"{name:25s} | Exp: {exp: .6f} | Pred: {pred: .6f} | Δ = {pred - exp: .6f}")

    # Plot correlation with labels
    plt.figure(figsize=(7,7))
    plt.scatter(y, y_pred, color="royalblue", s=50)

    for i, name in enumerate(feature_vectors.keys()):
        plt.text(y[i], y_pred[i], name, fontsize=8, ha='right', va='bottom', rotation=30)

    plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=1)
    plt.xlabel("Experimental Energy (Ha)")
    plt.ylabel("Fitted Energy (Ha)")
    plt.title(f"Least Squares Fit: R² = {r_value**2:.3f}")
    plt.tight_layout()
    plt.show()

    return coeffs, contributions, y_pred


# === Run the fit ===
coeffs, contributions, y_pred = fit_atomic_bond_contributions(fitting_dict, atom_labels, bond_labels)

# === Apply fitted model to test set ===
def predict_test_set(test_dict, atom_labels, bond_labels, coeffs):
    feature_vectors, _ = build_feature_vectors(test_dict, atom_labels, bond_labels)

    X_test = []
    y_test = []

    for name, vec in feature_vectors.items():
        X_test.append(vec)
        y_test.append(test_dict[name]["Experiment_Energy"])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Add bias term
    X_test_aug = np.column_stack((np.ones(X_test.shape[0]), X_test))

    # Predicted energies using fitted coefficients
    y_pred_test = X_test_aug @ coeffs

    # Correlation statistics
    slope, intercept_r, r_value, p_value, std_err = linregress(y_test, y_pred_test)

    print(f"\n=== Test Set Performance ===")
    print(f"R² = {r_value**2:.4f}")
    print(f"Slope = {slope:.4f}")
    print(f"Intercept = {intercept_r:.6f} Ha")

    # Plot correlation with labels
    plt.figure(figsize=(7,7))
    plt.scatter(y_test, y_pred_test, color="darkorange", s=50)

    for i, name in enumerate(feature_vectors.keys()):
        plt.text(y_test[i], y_pred_test[i], name, fontsize=8, ha='right', va='bottom', rotation=30)

    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=1)
    plt.xlabel("Experimental Energy (Ha)")
    plt.ylabel("Predicted Energy (Ha)")
    plt.title(f"Test Set Prediction: R² = {r_value**2:.3f}")
    plt.tight_layout()
    plt.show()

    # Print out per-molecule results
    print("\n=== Test Set Predictions (Ha) ===")
    for name, exp, pred in zip(feature_vectors.keys(), y_test, y_pred_test):
        print(f"{name:25s} | Exp: {exp: .6f} | Pred: {pred: .6f} | Δ = {pred - exp: .6f}")

    return y_pred_test, r_value**2



# === Run prediction on test set ===
y_pred_test, r2_test = predict_test_set(test_dict, atom_labels, bond_labels, coeffs)
