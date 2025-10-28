import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
from collections import Counter
import numpy as np

# "NAME": "SMILES", EXPERIMENTAL ENERGY (Kcal/mol -> Ha), MACE ENERGIES (eV -> Ha), PAPER ENERGY (Ha)
fitting_dict = { # 20 mols in test set
    "nitromethane": {
        "SMILES": "C[N+](=O)[O-]",
        "Experiment_Energy": -19.3 / 627.5095,
        # "MACE energy": -6671.50648846 / 27.211386245988, # SMALL
        # "MACE energy": -6671.53561703 / 27.211386245988, # MEDIUM
        "MACE energy": -6671.53681134 / 27.211386245988, # LARGE
        "Paper energy": 245.1064954,
    },
    "azido11dinitroethane": {
        "SMILES": "CC(N=[N+]=[N-])([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 60.4 / 627.5095,
        # "MACE energy":  -17764.70543565 / 27.211386245988, # SMALL
        # "MACE energy": -17764.68747360 / 27.211386245988, # MEDIUM
        "MACE energy": -17764.66301768 / 27.211386245988, # LARGE
        "Paper energy": -652.6296314,
        
    },

    "nitroglycerin": {
        "SMILES": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",
        "Experiment_Energy": -66.71 / 627.5095,
        # "MACE energy": -26091.01398203 / 27.211386245988, # SMALL
        # "MACE energy": -26090.96992151 / 27.211386245988, # MEDIUM
        "MACE energy": -26091.05035830 / 27.211386245988, # LARGE
        "Paper energy": -958.5266091,

    },
    "dinitroso14piperazine": {
        "SMILES": "C1CN(CCN1N=O)N=O",
        "Experiment_Energy": 46.4 / 627.5095,
        # "MACE energy":  -14337.78974013 / 27.211386245988, # SMALL
        # "MACE energy":  -14337.77507288 / 27.211386245988, # MEDIUM
        "MACE energy": -14337.74581524 / 27.211386245988, # LARGE
        "Paper energy": -526.7255232,

    },
    "dinitro14piperazine": {
        "SMILES": "C1CN(CCN1[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 13.9 / 627.5095,
        # "MACE energy": -18432.29259350/ 27.211386245988, # SMALL
        # "MACE energy": -18432.21454611 / 27.211386245988, # MEDIUM
        "MACE energy": -18432.19973047 / 27.211386245988, # LARGE
        "Paper energy": -677.1533702,
    },

    "nitrobenzene": {
        "SMILES": "C1=CC=C(C=C1)[N+](=O)[O-]",
        "Experiment_Energy": 16.38 / 627.5095,
        # "MACE energy": -11892.76308846 / 27.211386245988, # SMALL
        # "MACE energy": -11892.77009503 / 27.211386245988, # MEDIUM
        "MACE energy": -11892.77026396 / 27.211386245988, # LARGE
        "Paper energy": -436.9026044,
    },

    "nitro2phenol": {
        "SMILES": "C1=CC=C(C(=C1)[N+](=O)[O-])O",
        "Experiment_Energy": -31.62 / 627.5095,
        # "MACE energy": -13941.32814988 / 27.211386245988, # SMALL
        # "MACE energy":  -13941.32303331 / 27.211386245988, # MEDIUM
        "MACE energy": -13941.31257310 / 27.211386245988, # LARGE
        "Paper energy": -512.1616809,
    },

    "nitro3phenol": {
        "SMILES": "C1=CC(=CC(=C1)O)[N+](=O)[O-]",
        "Experiment_Energy": -26.12 / 627.5095,
        # "MACE energy": -13941.14837621 / 27.211386245988, # SMALL
        # "MACE energy":  -13941.16405913 / 27.211386245988, # MEDIUM
        "MACE energy": -13941.16408853 / 27.211386245988, # LARGE
        "Paper energy": -512.1535723,
    },

    "mnitroaniline": {
        "SMILES": "C1=CC(=CC(=C1)[N+](=O)[O-])N",
        "Experiment_Energy": 14.9 / 627.5095,
        # "MACE energy": -13400.30744689 / 27.211386245988, # SMALL
        # "MACE energy":  -13400.31264451 / 27.211386245988, # MEDIUM
        "MACE energy": -13400.31309819 / 27.211386245988, # LARGE
        "Paper energy": -492.2836904,
    },
    
    "pnitroaniline": {
        "SMILES": "C1=CC(=CC=C1N)[N+](=O)[O-]",
        "Experiment_Energy": 13.2 / 627.5095,
        # "MACE energy":  -13400.36662051 / 27.211386245988, # SMALL
        # "MACE energy": -13400.38312772 / 27.211386245988, # MEDIUM
        "MACE energy": -13400.37316552 / 27.211386245988, # LARGE
        "Paper energy": -492.2876859,
    },

    "azidobenzene": {
        "SMILES": "C1=CC=C(C=C1)N=[N+]=[N-]",
        "Experiment_Energy": 93.0 / 627.5095,
        # "MACE energy": -10778.75954697 / 27.211386245988, # SMALL
        # "MACE energy": -10778.77510060 / 27.211386245988, # MEDIUM
        "MACE energy": -10778.76284064 / 27.211386245988, # LARGE
        "Paper energy": -395.9704254,
    },

    "azido1nitro4benzene": {
        "SMILES": "C1=CC(=CC=C1N=[N+]=[N-])[N+](=O)[O-]",
        "Experiment_Energy": 93.1 / 627.5095,
        # "MACE energy": -16347.35684216 / 27.211386245988, # SMALL
        # "MACE energy": -16347.38932583 / 27.211386245988, # MEDIUM
        "MACE energy": -16347.38105695 / 27.211386245988, # LARGE
        "Paper energy": -600.5459965,
    },
    "methyl1nitro4benzene": {
        "SMILES": "CC1=CC=C(C=C1)[N+](=O)[O-]",
        "Experiment_Energy": 7.38 / 627.5095,
        # "MACE energy": -12963.36704287 / 27.211386245988, # SMALL
        # "MACE energy": -12963.36932102 / 27.211386245988, # MEDIUM
        "MACE energy": -12963.36605582 / 27.211386245988, # LARGE
        "Paper energy": -476.2339492,
    },

    "methyl1dinitro24benzene": {
        "SMILES": "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 7.93 / 627.5095,
        # "MACE energy": -18531.76437310 / 27.211386245988, # SMALL
        # "MACE energy": -18531.80614829 / 27.211386245988, # MEDIUM
        "MACE energy": -18531.79536728 / 27.211386245988, # LARGE
        "Paper energy": -680.7997820,
    },

    "azidomethylbenzene": {
        "SMILES": "C1=CC=C(C=C1)CN=[N+]=[N-]",
        "Experiment_Energy": 99.5 / 627.5095,
        # "MACE energy": -11849.12752812 / 27.211386245988, # SMALL
        # "MACE energy": -11849.11935131 / 27.211386245988, # MEDIUM
        "MACE energy": -11849.09884166 / 27.211386245988, # LARGE
        "Paper energy": -435.2904566,
    },

    "azido3ethyl3pentane": {
        "SMILES": "CCC(CC)(CC)N=[N+]=[N-]",
        "Experiment_Energy": 40.6 / 627.5095,
        # "MACE energy": -11980.37260140 / 27.211386245988, # SMALL
        # "MACE energy": -11980.35881750 / 27.211386245988, # MEDIUM
        "MACE energy": -11980.33411541 / 27.211386245988, # LARGE
        "Paper energy": -440.1288955,
    },

    "tnt": {
        "SMILES": "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 5.75 / 627.5095,
        # "MACE energy": -24100.00763976 / 27.211386245988, # SMALL
        # "MACE energy": -24100.03032071 / 27.211386245988, # MEDIUM
        "MACE energy": -24100.00418762 / 27.211386245988, # LARGE
        "Paper energy": -885.3582522,
    },
    
    "dinitro22adamantane": {
        "SMILES": "C1C2CC3CC1CC(C2)C3([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": -36.88 / 627.5095,
        # "MACE energy": -21776.71822500 / 27.211386245988, # SMALL
        # "MACE energy": -21776.75066348 / 27.211386245988, # MEDIUM
        "MACE energy": -21776.67820248 / 27.211386245988, # LARGE  
        "Paper energy": -799.9797733,
    },

    "azido1adamantane": {
        "SMILES": "C1C2CC3CC1CC(C2)(C3)N=[N+]=[N-]",
        "Experiment_Energy": 51.6 / 627.5095,
        # "MACE energy":  -15094.40825930 / 27.211386245988, # SMALL
        # "MACE energy": -15094.34323451 / 27.211386245988, # MEDIUM
        "MACE energy": -15094.35279210 / 27.211386245988, # LARGE 
        "Paper energy": -554.4870509,
    },

    "hns": {
        "SMILES": "C1=C(C=C(C(=C1[N+](=O)[O-])/C=C/C2=C(C=C(C=C2[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 56.98 / 627.5095,
        # "MACE energy": -48133.88855310 / 27.211386245988, # SMALL
        # "MACE energy": -48133.87041803 / 27.211386245988, # MEDIUM
        "MACE energy": -48134.01261656 / 27.211386245988, # LARGE
        "Paper energy": -1768.2654374,
    },

}


test_dict = { # 9 in test set
        "methylnitrite": {
        "SMILES": "CON=O",
        "Experiment_Energy": -15.64 / 627.5095,
        # "MACE energy": -6671.32812732 / 27.211386245988, # SMALL
        # "MACE energy": -6671.45085405 / 27.211386245988, # MEDIUM
        "MACE energy": -6671.41629503 / 27.211386245988, # LARGE
        "Paper energy": -245.1002638,
        },

        "methylnitrate": {
        "SMILES": "CO[N+](=O)[O-]",
        "Experiment_Energy": -29.2 / 627.5095,
        # "MACE energy": -8718.63615021 / 27.211386245988, # SMALL
        # "MACE energy": -8718.63486707 / 27.211386245988, # MEDIUM
        "MACE energy": -8718.67110644 / 27.211386245988, # LARGE
        "Paper energy": -320.3135279,
        },

        "dinitromethane": {
        "SMILES": "C([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": -14.1 / 627.5095,
        # "MACE energy":  -12239.60258108 / 27.211386245988, # SMALL
        # "MACE energy": -12239.61913241 / 27.211386245988, # MEDIUM
        "MACE energy":  -12239.62267937 / 27.211386245988, # LARGE
        "Paper energy": -449.6606863,
        },

        "ethylnitrite":{
        "SMILES": "CCON=O",
        "Experiment_Energy": -25.9 / 627.5095, 
        # "MACE energy":  -7741.94224597 / 27.211386245988, # SMALL
        # "MACE energy": -7741.97920809 / 27.211386245988, # MEDIUM
        "MACE energy": -7742.00436052 / 27.211386245988, # LARGE
        "Paper energy": -284.4316497,
        },

        "methyl2nitro2propane": {
        "SMILES": "CC(C)(C)[N+](=O)[O-]",
        "Experiment_Energy": -42.79 / 627.5095,
        # "MACE energy": -9883.32776301 / 27.211386245988, # SMALL
        # "MACE energy": -9883.29252853 / 27.211386245988, # MEDIUM
        "MACE energy": -9883.32101723 / 27.211386245988, # LARGE
        "Paper energy": -363.0946801,
        },

        "nbutylnitrite": {
        "SMILES": "CCCCON=O",
        "Experiment_Energy": -34.8 / 627.5095,
        # "MACE energy": -9882.85196516 / 27.211386245988, # SMALL
        # "MACE energy": -9882.95566221 / 27.211386245988, # MEDIUM
        "MACE energy": -9882.94801119 / 27.211386245988, # LARGE
        "Paper energy": -363.0848109
        },

        "nitrosobenzene":{
        "SMILES": "C1=CC=C(C=C1)N=O",
        "Experiment_Energy": 48.1 / 627.5095,
        # "MACE energy": -9844.75581342 / 27.211386245988, # SMALL
        # "MACE energy": -9844.77533086 / 27.211386245988, # MEDIUM
        "MACE energy": -9844.80180183 / 27.211386245988, # LARGE
        "Paper energy": -566.2201576
        },

        "nitromethylbenzene":{
        "SMILES": "C1=CC=C(C=C1)C[N+](=O)[O-]",
        "Experiment_Energy": 7.34 / 627.5095,
        # "MACE energy": -12963.24083556 / 27.211386245988, # SMALL
        # "MACE energy": -12963.27156823 / 27.211386245988, # MEDIUM
        "MACE energy": -12963.26666729 / 27.211386245988, # LARGE
        "Paper energy": -476.2290336,
        },

        "dinitromethylbenzene": {
        "SMILES": "C1=CC=C(C=C1)C([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 8.3 / 627.5095,
        # "MACE energy": -18531.39293741 / 27.211386245988, # SMALL
        # "MACE energy": -18531.4330449 / 27.211386245988, # MEDIUM
        "MACE energy": -18531.42580645 / 27.211386245988, # LARGE
        "Paper energy": -680.7851161
        },
       
}

HARTREE_TO_KCAL = 627.5095

def get_dynamic_labels(*smiles_dicts):
    """
    Generate dynamic atom and bond labels from one or more SMILES dictionaries.

    This function parses all SMILES strings provided in the input dictionaries,
    adds explicit hydrogens, and collects unique labels for atoms and bonds.
    Aromatic atoms are distinguished with the suffix "_arom".

    Args:
        *smiles_dicts: One or more dictionaries containing molecule data.
            Each dictionary should map molecule names to a sub-dictionary
            containing a "SMILES" key.

    Returns:
        tuple[list[str], list[str]]: Two sorted lists:
            - atom_labels: unique atom labels including aromatic variants.
            - bond_labels: unique bond labels including bond order.
    """
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

def build_feature_vectors(SMILES_dict, atom_labels, bond_labels):
    """
    Build molecular feature vectors based on atom and bond counts.

    For each molecule in the input dictionary, this function constructs a feature
    vector consisting of:
      - counts of all atom types defined in `atom_labels`
      - counts of all bond types defined in `bond_labels`

    Explicit hydrogens are added before counting to ensure consistency.
    Aromatic atoms are labelled with "_arom". The resulting feature vector
    concatenates atom and bond count features.

    Args:
        SMILES_dict (dict): Dictionary mapping molecule names to data including
            a "SMILES" key.
        atom_labels (list[str]): List of atom labels to include in feature vectors.
        bond_labels (list[str]): List of bond labels to include in feature vectors.

    Returns:
        tuple[dict[str, list[int]], dict[str, dict[str, int]]]:
            - feature_vectors: mapping of molecule names to numerical feature vectors.
            - atom_counts_dict: mapping of molecule names to atom count dictionaries
              (simple counts by element symbol).
    """    
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


HARTREE_TO_KCAL = 627.5095


def fit_atomic_bond_contributions(fitting_dict, atom_labels, bond_labels):
    """
    Fit atomic and bond contributions to experimental formation enthalpies
    using least squares regression.

    The function constructs feature vectors for all molecules in the fitting set
    and fits a linear model of the form:

        E_exp ≈ intercept + Σ_i (n_i * w_i)

    where `n_i` are atom and bond counts, and `w_i` are their fitted contributions.

    Model performance (R² and MAE) is computed, and both experimental and predicted
    enthalpies are plotted with a secondary error axis.

    Args:
        fitting_dict (dict): Dictionary of molecules with experimental energies
            and SMILES strings.
        atom_labels (list[str]): List of atom labels.
        bond_labels (list[str]): List of bond labels.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            - coeffs: full regression coefficients (including intercept).
            - contributions: atom and bond contribution coefficients (Ha).
            - y_pred: predicted energies (Ha).
            - r2: coefficient of determination for the fitting set.
    """
    feature_vectors, _ = build_feature_vectors(fitting_dict, atom_labels, bond_labels)

    X = np.array(list(feature_vectors.values()))
    y = np.array([fitting_dict[name]["Experiment_Energy"] for name in feature_vectors.keys()])

    X_aug = np.column_stack((np.ones(X.shape[0]), X))
    coeffs, residuals, rank, s = np.linalg.lstsq(X_aug, y, rcond=None)

    intercept = coeffs[0]
    contributions = coeffs[1:]
    y_pred = X_aug @ coeffs

    # --- Correct R² and MAE computation ---
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = mean_absolute_error(y, y_pred)

    # --- Convert values to kcal/mol ---
    y_kcal = y * HARTREE_TO_KCAL
    y_pred_kcal = y_pred * HARTREE_TO_KCAL
    mae_kcal = mae * HARTREE_TO_KCAL
    errors_kcal = (y_pred - y) * HARTREE_TO_KCAL

    print(f"R² = {r2:.4f}")
    print(f"MAE = {mae_kcal:.3f} kcal/mol")
    print(f"Intercept (bias): {intercept:.6f} Ha")

    print("\n=== Atomic/Bond Contributions (Ha) ===")
    for label, val in zip(atom_labels + bond_labels, contributions):
        print(f"{label:20s}: {val: .6f}")

    print("\n=== Fitting Set Predictions ===")
    for name, exp, pred in zip(feature_vectors.keys(), y_kcal, y_pred_kcal):
        print(f"{name:25s} | Exp: {exp: .2f} kcal/mol | Pred: {pred: .2f} kcal/mol | Δ = {pred - exp: .2f}")

    # --- Plot with secondary y-axis (scatter) for error ---
    fig, ax1 = plt.subplots(figsize=(7,7))
    ax1.scatter(y_kcal, y_pred_kcal, color="black", s=50,)# label="Predicted vs Exp")
    # for i, name in enumerate(feature_vectors.keys()):
    #     ax1.text(y_kcal[i], y_pred_kcal[i], name, fontsize=8, ha='right', va='bottom', rotation=30)
    ax1.plot([min(y_kcal), max(y_kcal)], [min(y_kcal), max(y_kcal)], 'k--', lw=1)
    ax1.set_xlabel("Experimental Enthalpy of Formation / Kcal mol-1")
    ax1.set_ylabel("Calculated Enthalpy of Formation / Kcal mol-1")

    # Secondary axis for error (scatter)
    ax2 = ax1.twinx()
    ax2.scatter(y_kcal, errors_kcal, color='red', s=40, alpha=0.7,)# label='Error (Pred − Exp)')
    ax2.set_ylabel("Prediction Error (kcal/mol)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower right")

    plt.title(f"Least Squares Fit: R² = {r2:.3f}, MAE = {mae_kcal:.2f} kcal/mol")
    plt.tight_layout()
    plt.show()

    return coeffs, contributions, y_pred, r2


# === Predict test set ===
def predict_test_set(test_dict, atom_labels, bond_labels, coeffs):
    """
    Predict formation enthalpies for a test set using fitted atomic and bond coefficients.

    This function builds molecular feature vectors for the test set and applies the
    linear model obtained from the fitting set. Model performance metrics (R² and MAE)
    are reported, and predicted vs. experimental energies are plotted with a
    secondary y-axis for prediction error.

    Args:
        test_dict (dict): Dictionary of test molecules with experimental energies
            and SMILES strings.
        atom_labels (list[str]): List of atom labels used for feature construction.
        bond_labels (list[str]): List of bond labels used for feature construction.
        coeffs (np.ndarray): Regression coefficients obtained from the fitting step.

    Returns:
        tuple[np.ndarray, float]:
            - y_pred_test: predicted test set energies (Ha).
            - r2_test: coefficient of determination for the test set.
    """
    feature_vectors, _ = build_feature_vectors(test_dict, atom_labels, bond_labels)

    X_test = np.array(list(feature_vectors.values()))
    y_test = np.array([test_dict[name]["Experiment_Energy"] for name in feature_vectors.keys()])

    X_test_aug = np.column_stack((np.ones(X_test.shape[0]), X_test))
    y_pred_test = X_test_aug @ coeffs

    # --- Correct R² and MAE computation ---
    ss_res_test = np.sum((y_test - y_pred_test) ** 2)
    ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_test = 1 - ss_res_test / ss_tot_test
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # --- Convert values to kcal/mol ---
    y_test_kcal = y_test * HARTREE_TO_KCAL
    y_pred_test_kcal = y_pred_test * HARTREE_TO_KCAL
    mae_test_kcal = mae_test * HARTREE_TO_KCAL
    errors_test_kcal = (y_pred_test - y_test) * HARTREE_TO_KCAL

    print(f"\n=== Test Set Performance ===")
    print(f"R² = {r2_test:.4f}")
    print(f"MAE = {mae_test_kcal:.3f} kcal/mol")

    # --- Plot with secondary y-axis (scatter) for error ---
    fig, ax1 = plt.subplots(figsize=(7,7))
    ax1.scatter(y_test_kcal, y_pred_test_kcal, color="black", s=50,)# label="Predicted vs Exp")
    # for i, name in enumerate(feature_vectors.keys()):
    #     ax1.text(y_test_kcal[i], y_pred_test_kcal[i], name, fontsize=8, ha='right', va='bottom', rotation=30)
    ax1.plot([min(y_test_kcal), max(y_test_kcal)], [min(y_test_kcal), max(y_test_kcal)], 'k--', lw=1)
    ax1.set_xlabel("Experimental Enthalpy of Formation / Kcal mol-1")
    ax1.set_ylabel("Calculated Enthalpy of Formation / Kcal mol-1")

    # Secondary axis for error (scatter)
    ax2 = ax1.twinx()
    ax2.scatter(y_test_kcal, errors_test_kcal, color='red', s=40, alpha=0.7, )#label='Error (Pred − Exp)')
    ax2.set_ylabel("Prediction Error / Kcal mol-1", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower right")

    plt.title(f"Test Set Prediction: R² = {r2_test:.3f}, MAE = {mae_test_kcal:.2f} kcal/mol")
    plt.tight_layout()
    plt.show()

    print("\n=== Test Set Predictions ===")
    for name, exp, pred in zip(feature_vectors.keys(), y_test_kcal, y_pred_test_kcal):
        print(f"{name:25s} | Exp: {exp: .2f} kcal/mol | Pred: {pred: .2f} kcal/mol | Δ = {pred - exp: .2f}")

    return y_pred_test, r2_test
# === Run everything ===
atom_labels, bond_labels = get_dynamic_labels(fitting_dict, test_dict)
coeffs, contributions, y_pred, r2_fit = fit_atomic_bond_contributions(fitting_dict, atom_labels, bond_labels)
y_pred_test, r2_test = predict_test_set(test_dict, atom_labels, bond_labels, coeffs)
