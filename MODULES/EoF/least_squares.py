import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from rdkit import Chem
from collections import Counter
import numpy as np
from sklearn.metrics import r2_score

# "NAME": "SMILES", EXPERIMENTAL ENERGY (Kcal/mol -> Ha), MACE ENERGIES (eV -> Ha)
# 64 mols in fitting set, 10 mols in test set
fitting_dict = { 

# === From Byrd 2006 ===
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
        "MACE energy": -24508.03652046 / 27.211386245988, # SMALL
        # "MACE energy": -24507.86659948 / 27.211386245988, # MEDIUM
        # "MACE energy": -24507.97131620 / 27.211386245988, # LARGE
    },

    "nbutylnitrite" : {
        "SMILES": "CCCCON=O",
        "Experiment_Energy": -34.8 / 627.5095,
        "MACE energy": -9882.84097443 / 27.211386245988, # SMALL
        # "MACE energy": -9882.96109645 / 27.211386245988, # MEDIUM
        # "MACE energy": -9882.90924521 / 27.211386245988, # LARGE
    },

    "propylnitrite" : {
        "SMILES": "CCCO[N+](=O)[O-]",
        "Experiment_Energy": -28.4 / 627.5095,
        "MACE energy": -10859.71196604 / 27.211386245988, # SMALL
        # "MACE energy": -10859.71606568 / 27.211386245988, # MEDIUM
        # "MACE energy": -10859.74379917 / 27.211386245988, # LARGE
    },

    "1nitropiperidine" : {
        "SMILES": "O=NN1CCCCC1",
        "Experiment_Energy": -10.6 / 627.5095,
        "MACE energy": -10380.24299052 / 27.211386245988, # SMALL
        # "MACE energy": -10380.23132950 / 27.211386245988, # MEDIUM
        # "MACE energy": -10380.22238569 / 27.211386245988, # LARGE
    },

# === From ACT (https://atct.anl.gov/Thermochemical%20Data/version%201.220/index.php) === 
# "NAME": "SMILES", EXPERIMENTAL ENERGY (KJ/mol -> Ha), MACE ENERGIES (eV -> Ha)
# === CORES ===

    "methane" : {
        "SMILES": "C",
        "Experiment_Energy": -74.513 / 2625.5,
        "MACE energy": -1103.06002443 / 27.211386245988, # SMALL
        # "MACE energy": -1103.06005380 / 27.211386245988, # MEDIUM
        # "MACE energy": -1103.06079782 / 27.211386245988, # LARGE
    },

    "ethane" : {
        "SMILES": "CC",
        "Experiment_Energy": -84.02 / 2625.5,
        "MACE energy": -2173.45577369 / 27.211386245988, # SMALL
        # "MACE energy": -2173.45470694 / 27.211386245988, # MEDIUM
        # "MACE energy": -2173.45465600 / 27.211386245988, # LARGE
    },

    "ethene" : {
        "SMILES": "C=C",
        "Experiment_Energy": 52.39 / 2625.5,
        "MACE energy": -2139.83448764 / 27.211386245988, # SMALL
        # "MACE energy": -2139.83183912 / 27.211386245988, # MEDIUM
        # "MACE energy": -2139.83243001 / 27.211386245988, # LARGE
    },

    "ethyne" : {
        "SMILES": "C#C",
        "Experiment_Energy": 228.32 / 2625.5,
        "MACE energy": -2105.76769785 / 27.211386245988, # SMALL
        # "MACE energy": -2105.76138332 / 27.211386245988, # MEDIUM
        # "MACE energy": -2105.76408550 / 27.211386245988, # LARGE
    },

    "propane" : {
        "SMILES": "CCC",
        "Experiment_Energy": -82.72 / 2625.5,
        "MACE energy": -3243.93501751 / 27.211386245988, # SMALL
        # "MACE energy": -3243.93397001 / 27.211386245988, # MEDIUM
        # "MACE energy": -3243.93441708 / 27.211386245988, # LARGE
    },

    "propene" : {
        "SMILES": "C=CC",
        "Experiment_Energy": 20.06 / 2625.5,
        "MACE energy": -3210.43714643 / 27.211386245988, # SMALL
        # "MACE energy": -3210.43593922 / 27.211386245988, # MEDIUM
        # "MACE energy": -3210.43612911 / 27.211386245988, # LARGE
    },


    "propyne" : {
        "SMILES": "C#CC",
        "Experiment_Energy": 185.56 / 2625.5,
        "MACE energy": -3176.50524243 / 27.211386245988, # SMALL
        # "MACE energy": -3176.50071510 / 27.211386245988, # MEDIUM
        # "MACE energy": -3176.50284239 / 27.211386245988, # LARGE
    },

    "benzene" : {
        "SMILES": "c1ccccc1",
        "Experiment_Energy": 83.18 / 2625.5,
        "MACE energy":  -6324.12886792 / 27.211386245988, # SMALL
        # "MACE energy": -6324.12788914 / 27.211386245988, # MEDIUM
        # "MACE energy": -6324.12479285 / 27.211386245988, # LARGE
    },

    "toluene" : {
        "SMILES": "Cc1ccccc1",
        "Experiment_Energy": 50.07 / 2625.5,
        "MACE energy": -7394.70111274 / 27.211386245988, # SMALL
        # "MACE energy": -7394.69555773 / 27.211386245988, # MEDIUM
        # "MACE energy": -7394.69361738 / 27.211386245988, # LARGE
    },

    "phenol" : {
        "SMILES": "Oc1ccccc1",
        "Experiment_Energy": -92.94 / 2625.5,
        "MACE energy": -8372.53575089 / 27.211386245988, # SMALL
        # "MACE energy": -8372.53100602 / 27.211386245988, # MEDIUM
        # "MACE energy": -8372.52977062 / 27.211386245988, # LARGE
    },

    "napthalene" : {
        "SMILES": "c1ccc2ccccc2c1",
        "Experiment_Energy": 147.68 / 2625.5,
        "MACE energy":  -10507.91615580 / 27.211386245988, # SMALL
        # "MACE energy": -10507.90858134 / 27.211386245988, # MEDIUM
        # "MACE energy": -10507.91303568 / 27.211386245988, # LARGE
    },

# === NITROSO ===
    # "nitrosyl_nitrate" : {
    #     "SMILES": "",
    #     "Experiment_Energy": 40.4 / 2625.5,
    #     "MACE energy": -11168.18263717  / 27.211386245988, # SMALL
    #     # "MACE energy": -11168.39718086 / 27.211386245988, # MEDIUM
    #     # "MACE energy": -11168.28427295 / 27.211386245988, # LARGE
    # },

    # "nitrosodioxaziridine" : {
    #     "SMILES": "",
    #     "Experiment_Energy": 407.7 / 2625.5,
    #     "MACE energy": -9120.86859056 / 27.211386245988, # SMALL
    #     # "MACE energy": -9121.33886911 / 27.211386245988, # MEDIUM
    #     # "MACE energy": -9121.28663616 / 27.211386245988, # LARGE
    # },

    "nitrosomethane" : {
        "SMILES": "CN=O",
        "Experiment_Energy": 71.06 / 2625.5,
        "MACE energy":  -4623.47874725 / 27.211386245988, # SMALL
        # "MACE energy":  -4623.45853896 / 27.211386245988, # MEDIUM
        # "MACE energy":  -4623.47040148 / 27.211386245988, # LARGE
    },

    "fulmic_acid" : {
        "SMILES": "[O-][N+]#C",
        "Experiment_Energy": 169.29 / 2625.5,
        "MACE energy":  -4590.16135572 / 27.211386245988, # SMALL
        # "MACE energy":  -4590.16494085 / 27.211386245988, # MEDIUM
        # "MACE energy":  -4590.19185992 / 27.211386245988, # LARGE
    },

# === BENZENES ===
    "benzaldehyde" : {
        "SMILES": "c1c(C=O)cccc1",
        "Experiment_Energy": -36.96 / 2625.5,
        "MACE energy":  -9409.99598941  / 27.211386245988, # SMALL
        # "MACE energy":  -9410.00303625 / 27.211386245988, # MEDIUM
        # "MACE energy":  -9410.00293138 / 27.211386245988, # LARGE
    },

    "phenylethene" : {
        "SMILES": "c1ccccc1C=C",
        "Experiment_Energy": 148.56 / 2625.5,
        "MACE energy":  -8431.72927314 / 27.211386245988, # SMALL
        # "MACE energy":  -8431.73373910 / 27.211386245988, # MEDIUM
        # "MACE energy":  -8431.72612394 / 27.211386245988, # LARGE
    },

    "phenylacetylene" : {
        "SMILES": "c1ccccc1C#C",
        "Experiment_Energy": 317.64 / 2625.5,
        "MACE energy":  -8397.75866843 / 27.211386245988, # SMALL
        # "MACE energy":  -8397.74567852 / 27.211386245988, # MEDIUM
        # "MACE energy":  -8397.74266377 / 27.211386245988, # LARGE
    },

    "acetophenone" : {
        "SMILES": "c1ccccc1C(=O)C",
        "Experiment_Energy": -83.90 / 2625.5,
        "MACE energy":  -10480.75947927 / 27.211386245988, # SMALL
        # "MACE energy":  -10480.75667769 / 27.211386245988, # MEDIUM
        # "MACE energy":  -10480.75705451 / 27.211386245988, # LARGE
    },

    "ethylbenzene" : {
        "SMILES": "c1ccccc1CC",
        "Experiment_Energy": 29.96 / 2625.5,
        "MACE energy":   -8465.17166019 / 27.211386245988, # SMALL
        # "MACE energy":  -8465.16904287 / 27.211386245988, # MEDIUM
        # "MACE energy":  -8465.16968358 / 27.211386245988, # LARGE
    },

    "anisole" : {
        "SMILES": "c1ccccc1OC",
        "Experiment_Energy": -70.82 / 2625.5,
        "MACE energy":  -9442.52625948 / 27.211386245988, # SMALL
        # "MACE energy":  -9442.52288818 / 27.211386245988, # MEDIUM
        # "MACE energy":  -9442.53046991 / 27.211386245988, # LARGE
    },

    "aniline" : {
        "SMILES": "c1ccccc1N",
        "Experiment_Energy": 86.82 / 2625.5,
        "MACE energy":  -7831.64376514  / 27.211386245988, # SMALL
        # "MACE energy":  -7831.64843428 / 27.211386245988, # MEDIUM
        # "MACE energy":  -7831.64650153 / 27.211386245988, # LARGE
    },

    "biphenyl" : {
        "SMILES": "c1ccccc1c2ccccc2",
        "Experiment_Energy": 178.8 / 2625.5,
        "MACE energy":  -12615.97284178 / 27.211386245988, # SMALL
        # "MACE energy":  -12615.99169595 / 27.211386245988, # MEDIUM
        # "MACE energy":  -12615.98846663 / 27.211386245988, # LARGE
    },

    "benzoic_acid" : {
        "SMILES": "c1ccccc1C(=O)O",
        "Experiment_Energy": -294.13 / 2625.5,
        "MACE energy":  -11459.30215369 / 27.211386245988, # SMALL
        # "MACE energy":  -11459.30269589 / 27.211386245988, # MEDIUM
        # "MACE energy":  -11459.29852098 / 27.211386245988, # LARGE
    },

    "cyclohexene" : {
        "SMILES": "C1CCC=CC1",
        "Experiment_Energy": -4.28 / 2625.5,
        "MACE energy":  -6389.35632365 / 27.211386245988, # SMALL
        # "MACE energy":  -6389.34740150 / 27.211386245988, # MEDIUM
        # "MACE energy":  -6389.34522231 / 27.211386245988, # LARGE
    },

    "1_2diphenylethane" : {
        "SMILES": "c1ccccc1CCc2ccccc2",
        "Experiment_Energy": 51.07 / 2625.5,
        "MACE energy":  -14756.90541182 / 27.211386245988, # SMALL
        # "MACE energy":  -14756.89448410 / 27.211386245988, # MEDIUM
        # "MACE energy":  -14756.89728679 / 27.211386245988, # LARGE
    },

    "benzophenone" : {
        "SMILES": "c1ccccc1C(=O)c2ccccc2",
        "Experiment_Energy": 51.5 / 2625.5,
        "MACE energy":  -15701.95591703  / 27.211386245988, # SMALL
        # "MACE energy":  -15701.93555013 / 27.211386245988, # MEDIUM
        # "MACE energy":  -15701.93448858 / 27.211386245988, # LARGE
    },

    "2_3dimethylbenzaldehyde" : {
        "SMILES": "Cc1cc(C)ccc1C=O",
        "Experiment_Energy": -93.7 / 2625.5,
        "MACE energy":  -11551.150049  / 27.211386245988, # SMALL
        # "MACE energy":  -11551.135440  / 27.211386245988, # MEDIUM
        # "MACE energy":  -11551.121786 / 27.211386245988, # LARGE
    },

    # === NITRITES ===

     "dioxohydrazine" : {
        "SMILES": "O=NN=O",
        "Experiment_Energy": 171.18 / 2625.5,
        "MACE energy":    -7074.17521614 / 27.211386245988, # SMALL
        # "MACE energy":  -7073.742701 / 27.211386245988, # MEDIUM
        # "MACE energy": -7073.73317278  / 27.211386245988, # LARGE
    },    


     "dinitrogen_pentoxide" : {
        "SMILES": "[O-][N+](=O)O[N+]([O-])=O",
        "Experiment_Energy": 14.79 / 2625.5,
        "MACE energy":   -13215.44906723  / 27.211386245988, # SMALL
        # "MACE energy":   -13215.290185 / 27.211386245988, # MEDIUM
        # "MACE energy":  -13215.35835993 / 27.211386245988, # LARGE
    },  

     "hydrazine" : {
        "SMILES": "NN",
        "Experiment_Energy": 97.64 / 2625.5,
        "MACE energy":   -3046.19594729  / 27.211386245988, # SMALL
        # "MACE energy":   -3046.203110 / 27.211386245988, # MEDIUM
        # "MACE energy":  -3046.21888261 / 27.211386245988, # LARGE
    },     

     "peroxynitrous_acid" : {
        "SMILES": "O=N(=O)OO",
        "Experiment_Energy": -11.54 / 2625.5,
        "MACE energy":   -9694.243137   / 27.211386245988, # SMALL
        # "MACE energy":   -9694.155442 / 27.211386245988, # MEDIUM
        # "MACE energy":  -9694.118822 / 27.211386245988, # LARGE
    },    

    "methylnitrate" : {
        "SMILES": "CO[N+](=O)[O-]",
        "Experiment_Energy": -29.2 / 627.5095,
        "MACE energy": -8718.63615026 / 27.211386245988, # SMALL
        # "MACE energy": -8718.63486708 / 27.211386245988, # MEDIUM
        # "MACE energy": -8718.67110647 / 27.211386245988, # LARGE
    },
    
}


test_dict = {

    "1_3dimethyl2nitrobenzene" : {
        "SMILES": "CC1=C(C(=CC=C1)C)[N+](=O)[O-]",
        "Experiment_Energy": 2.1 / 627.5095,
        "MACE energy": -14033.77625063 / 27.211386245988, # SMALL
        # "MACE energy": -14033.63278412 / 27.211386245988, # MEDIUM
        # "MACE energy": -14033.61713095 / 27.211386245988, # LARGE
    },

    "dinitromethane" : {
        "SMILES": "C([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": -14.1 / 627.5095,
        "MACE energy": -12239.60258122 / 27.211386245988, # SMALL
        # "MACE energy": -12239.61913223 / 27.211386245988, # MEDIUM
        # "MACE energy": -12239.62267937 / 27.211386245988, # LARGE
    },

    "dinitromethylbenzene" : {
        "SMILES": "C1=CC=C(C=C1)C([N+](=O)[O-])[N+](=O)[O-]",
        "Experiment_Energy": 8.3 / 627.5095,
        "MACE energy": -18531.39293788 / 27.211386245988, # SMALL
        # "MACE energy": -18531.43304428 / 27.211386245988, # MEDIUM
        # "MACE energy": -18531.42580448 / 27.211386245988, # LARGE
    },

    "ethylnitrate" : {
        "SMILES": "CCO[N+](=O)[O-]",
        "Experiment_Energy": -25.9 / 627.5095,
        "MACE energy": -9789.23531244 / 27.211386245988, # SMALL
        # "MACE energy": -9789.23505542 / 27.211386245988, # MEDIUM
        # "MACE energy": -9789.26673278 / 27.211386245988, # LARGE
    },

    "ethylnitrite" : {
        "SMILES": "CCON=O",
        "Experiment_Energy": -37.0 / 627.5095,
        "MACE energy": -7741.90209714 / 27.211386245988, # SMALL
        # "MACE energy": -7742.01150896 / 27.211386245988, # MEDIUM
        # "MACE energy": -7742.01470880 / 27.211386245988, # LARGE
    },

    "2methyl2nitropropane" : {
        "SMILES": "CC(C)(C)[N+](=O)[O-]",
        "Experiment_Energy": -42.32 / 627.5095,
        "MACE energy": -9883.32776292 / 27.211386245988, # SMALL
        # "MACE energy": -9883.29251470 / 27.211386245988, # MEDIUM
        # "MACE energy": -9883.32101707 / 27.211386245988, # LARGE
    },


    "methylnitrobenzene" : {
        "SMILES": "c1ccccc1C([N+](=O)[O-])",
        "Experiment_Energy": 7.38 / 627.5095,
        "MACE energy": -12963.24084391 / 27.211386245988, # SMALL
        # "MACE energy": -12963.27153977 / 27.211386245988, # MEDIUM
        # "MACE energy": -12963.26666778 / 27.211386245988, # LARGE
    },


    "nitrosobenzene" : {
        "SMILES": "C1=CC=C(C=C1)N=O",
        "Experiment_Energy": 48.1 / 627.5095,
        "MACE energy": -9844.75581343 / 27.211386245988, # SMALL
        # "MACE energy": -9844.77533103 / 27.211386245988, # MEDIUM
        # "MACE energy": -9844.80180181 / 27.211386245988, # LARGE
    },

    "methylnitrite" : {
        "SMILES": "CON=O",
        "Experiment_Energy": -15.64 / 627.5095,
        "MACE energy": -6671.32812737 / 27.211386245988, # SMALL
        # "MACE energy": -6671.45085405 / 27.211386245988, # MEDIUM
        # "MACE energy": -6671.41629495 / 27.211386245988, # LARGE
    },

    "tertbutylnitrite" : {
        "SMILES": "CC(C)(C)ON=O",
        "Experiment_Energy": -41.0/ 627.5095,
        "MACE energy": -9882.97171893 / 27.211386245988, # SMALL
        # "MACE energy": -9883.12009522 / 27.211386245988, # MEDIUM
        # "MACE energy": -9883.07147083 / 27.211386245988, # LARGE
    },

# === From ATT (https://atct.anl.gov/Thermochemical%20Data/version%201.220/index.php) ===


}

# === Dynamic atom and bond label generation ===
def get_dynamic_labels(*smiles_dicts):
    """
    Determine all unique atom and bond labels present in one or more SMILES dictionaries.

    Each molecule is parsed and hydrogens are added. Atom and bond information
    (including charge and aromaticity) is used to generate sets of unique labels.
    These labels are then used to construct consistent feature vectors across datasets.

    Note:
        This function traverses all atoms and bonds in every molecule in order to
        identify all possible labels. Atom and bond counts are therefore recalculated
        later in `build_feature_vectors` and `fit_atomic_bond_contributions`.

    Args:
        *smiles_dicts: One or more dictionaries where each key is a molecule name and
            each value is a dictionary containing at least a "SMILES" key.

    Returns:
        tuple[list[str], list[str]]: Sorted lists of unique atom and bond labels.
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
                charge = atom.GetFormalCharge()
                # Include charge in atom label if nonzero
                label = f"{symbol}({charge:+d})" if charge != 0 else symbol

                # Preserve aromatic information
                if atom.GetIsAromatic() and symbol in ["C", "N", "O"]:
                    label += "_arom"

                atom_labels.add(label)

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                a1_label = f"{a1.GetSymbol()}({a1.GetFormalCharge():+d})" if a1.GetFormalCharge() != 0 else a1.GetSymbol()
                a2_label = f"{a2.GetSymbol()}({a2.GetFormalCharge():+d})" if a2.GetFormalCharge() != 0 else a2.GetSymbol()
                atoms = sorted([a1_label, a2_label])
                bond_type = str(bond.GetBondType())
                bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
                bond_labels.add(bond_label)

    return sorted(atom_labels), sorted(bond_labels)

def add_charge_resonance(mol):
    """
    Adjust formal charges to represent nitro-like resonance structures.

    This function searches for nitrogen atoms bonded to two oxygens,
    one through a double bond and one through a single bond, and applies
    +1 charge to nitrogen and −1 to the singly bonded oxygen. (allows for more degrees of freedom)

    Args:
        mol (Chem.Mol): RDKit molecule with explicit hydrogens.

    Returns:
        Chem.Mol: Modified molecule with adjusted formal charges.
    """
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N":
            # Detect nitro-like patterns: N bonded to 2 oxygens, one double, one single
            oxygens = [nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "O"]
            if len(oxygens) == 2:
                bond_types = [mol.GetBondBetweenAtoms(atom.GetIdx(), o.GetIdx()).GetBondType() for o in oxygens]
                if set(bond_types) == {Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE}:
                    atom.SetFormalCharge(+1)
                    for o in oxygens:
                        if mol.GetBondBetweenAtoms(atom.GetIdx(), o.GetIdx()).GetBondType() == Chem.rdchem.BondType.SINGLE:
                            o.SetFormalCharge(-1)
    return mol


# === Feature vector builder ===
def build_feature_vectors(SMILES_dict, atom_labels, bond_labels):
    """
    Build atom and bond-based feature vectors for a set of molecules.

    Each molecule is parsed, hydrogens are added, and atom and bond occurrences
    are counted using the provided label lists. The result is a consistent vector
    representation for each molecule.

    Note:
        This function recomputes atom and bond counts for all molecules,
        even though a similar traversal occurs in `get_dynamic_labels`.

    Args:
        SMILES_dict (dict): Dictionary mapping molecule names to data dictionaries
            containing at least a "SMILES" key.
        atom_labels (list[str]): List of all unique atom labels.
        bond_labels (list[str]): List of all unique bond labels.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - feature_vectors: A dictionary mapping molecule names to
              concatenated atom and bond count vectors.
            - atom_counts_dict: A dictionary mapping molecule names to
              simple atomic composition counts.
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
            charge = atom.GetFormalCharge()
            label = f"{symbol}({charge:+d})" if charge != 0 else symbol
            if atom.GetIsAromatic() and symbol in ["C", "N", "O"]:
                label += "_arom"
            atom_counter[label] += 1

        bond_counter = Counter()
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            a1_label = f"{a1.GetSymbol()}({a1.GetFormalCharge():+d})" if a1.GetFormalCharge() != 0 else a1.GetSymbol()
            a2_label = f"{a2.GetSymbol()}({a2.GetFormalCharge():+d})" if a2.GetFormalCharge() != 0 else a2.GetSymbol()
            atoms = sorted([a1_label, a2_label])
            bond_type = str(bond.GetBondType())
            bond_label = f"{atoms[0]} - {atoms[1]} {bond_type}"
            bond_counter[bond_label] += 1

        simple_atom_counter = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
        atom_counts_dict[name] = dict(simple_atom_counter)

        atom_vector = [atom_counter.get(a, 0) for a in atom_labels]
        bond_vector = [bond_counter.get(b, 0) for b in bond_labels]
        feature_vectors[name] = atom_vector + bond_vector

    print(f"\nTotal molecules in test set: {len(feature_vectors)}")

    return feature_vectors, atom_counts_dict


def fit_atomic_bond_contributions(fitting_dict, atom_labels, bond_labels, show_plot=True):
    """
    Fit a linear model for atom and bond contributions to energy corrections.

    The model fits the difference between MACE and experimental ΔHf values
    using counts of atom and bond types as features. Optionally plots predicted
    versus experimental ΔHf values for the fitting set.

    Note:
        Atom and bond counts are recalculated here even if they were previously
        computed in `get_dynamic_labels` or `build_feature_vectors`. This ensures
        up-to-date charge and resonance adjustments are included in the fit.

    Args:
        fitting_dict (dict): Dictionary mapping molecule names to data entries,
            each containing "SMILES", "MACE energy", and "Experiment_Energy".
        atom_labels (list[str]): List of unique atom labels.
        bond_labels (list[str]): List of unique bond labels.
        show_plot (bool, optional): If True, plot predicted vs experimental ΔHf.
            Defaults to True.

    Returns:
        None
    """

    names, X, y_corr, y_exp, y_mace = [], [], [], [], []

    for name, data in fitting_dict.items():
        try:
            mol = Chem.MolFromSmiles(data["SMILES"])
            if mol is None:
                print(f"Could not parse SMILES for {name}. Skipping.")
                continue

            mol = Chem.AddHs(mol)
            mol = add_charge_resonance(mol) 

            # === Count atoms ===
            atom_counts = np.zeros(len(atom_labels))
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                charge = atom.GetFormalCharge()
                label = f"{symbol}({charge:+d})" if charge != 0 else symbol
                if atom.GetIsAromatic() and symbol in ["C", "N", "O"]:
                    label += "_arom"
                if label in atom_labels:
                    atom_counts[atom_labels.index(label)] += 1

            # === Count bonds ===
            bond_counts = np.zeros(len(bond_labels))
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                a1_label = f"{a1.GetSymbol()}({a1.GetFormalCharge():+d})" if a1.GetFormalCharge() != 0 else a1.GetSymbol()
                a2_label = f"{a2.GetSymbol()}({a2.GetFormalCharge():+d})" if a2.GetFormalCharge() != 0 else a2.GetSymbol()
                atoms = sorted([a1_label, a2_label])
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

    print(f"\nTotal molecules in fitting set: {len(X)}")

    X = np.array(X)
    # print(f"{X}")
    y_corr = np.array(y_corr)
    y_exp = np.array(y_exp)
    y_mace = np.array(y_mace)

    if X.shape[0] == 0: # X[0] is the number of rows (number of molecules)
        raise ValueError("No valid data found in fitting_dict.")

    model = LinearRegression(fit_intercept=True) # define the model, and specify to fit the intercept (enables an extra degree of freedom)
    model.fit(X, y_corr) # Solves len(test_dict) equations with len(atom_labels) + len(bond_labels) variables
    coeffs = model.coef_ # extract contributions from each feature
    intercept = model.intercept_ # extract global correction

    y_pred_corr = model.predict(X) # apply contributions to feature vectors, to calculate predicted eofs
    y_pred_exp = y_mace - y_pred_corr # calculate difference between mace and experimental eofs
    r2_fit = r2_score(y_exp, y_pred_exp)

    print("\n===== Linear Fit Summary =====")
    print(f"R² (fit) = {r2_fit:.4f}")
    print(f"Intercept = {intercept:.6f} Ha")
    print("\nCoefficients:")
    for i, label in enumerate(atom_labels + bond_labels):
        print(f"{label:>12s}: {coeffs[i]: .6f}")

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
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.grid(alpha=0.3)

        ax1.set_xlabel("Experimental EoF (Ha)", fontsize=12)
        ax1.set_ylabel("Predicted EoF (Ha)", fontsize=12)
        ax1.set_title(f"Fitting Set (SMALL Model)\nR² = {r2_fit:.4f}", fontsize=13)

        # Secondary axis for prediction errors
        ax2 = ax1.twinx()
        ax2.scatter(y_exp, errors, c='red', s=40, alpha=0.8, label="Prediction Error")
        ax2.set_ylabel("Prediction Error (Ha)", color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')

        # Annotate molecule names
        for i, n in enumerate(names):
            ax1.text(y_exp[i], y_pred_exp[i], n, fontsize=8, ha='right', va='bottom', alpha=0.7)

        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.tight_layout()
        plt.show()



    contributions = {n: c for n, c in zip(names, y_pred_corr)}
    return coeffs, contributions, y_pred_exp, r2_fit


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
            mol = add_charge_resonance(mol) 

            atom_counts = np.zeros(len(atom_labels))
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                charge = atom.GetFormalCharge()
                label = f"{symbol}({charge:+d})" if charge != 0 else symbol
                if atom.GetIsAromatic() and symbol in ["C", "N", "O"]:
                    label += "_arom"
                if label in atom_labels:
                    atom_counts[atom_labels.index(label)] += 1

            bond_counts = np.zeros(len(bond_labels))
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                a1_label = f"{a1.GetSymbol()}({a1.GetFormalCharge():+d})" if a1.GetFormalCharge() != 0 else a1.GetSymbol()
                a2_label = f"{a2.GetSymbol()}({a2.GetFormalCharge():+d})" if a2.GetFormalCharge() != 0 else a2.GetSymbol()
                atoms = sorted([a1_label, a2_label])
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

    y_corr_pred = X_test @ coeffs + intercept # multipy each feature vector by the test set coefficient value
    y_pred_exp = y_mace - y_corr_pred
    r2_test = r2_score(y_exp, y_pred_exp)

    print("\n===== Test Set Performance =====")
    print(f"R² (test) = {r2_test:.4f}")
    print(f"{'Molecule':25s}  {'Pred (Ha)':>12s}  {'Exp (Ha)':>12s}  {'Δ (Ha)':>12s}  {'% Error':>10s}")
    print("-" * 80)

    # --- Sort results by absolute Δ (largest difference first) ---
    results = []
    for n, y_p, y_e in zip(names, y_pred_exp, y_exp):
        delta = y_p - y_e
        percent_error = (delta / abs(y_e)) * 100 if y_e != 0 else float('nan')
        results.append((n, y_p, y_e, delta, percent_error))

    results.sort(key=lambda x: abs(x[3]), reverse=True)

    for n, y_p, y_e, delta, percent_error in results:
        print(f"{n:25s}  {y_p: 12.6f}  {y_e: 12.6f}  {delta: 12.6f}  {percent_error: 10.2f}")



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
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.grid(alpha=0.3)

        ax1.set_xlabel("Experimental EoF (Ha)", fontsize=12)
        ax1.set_ylabel("Predicted EoF (Ha)", fontsize=12)
        ax1.set_title(f"Test Set (SMALL Model)\nR² = {r2_test:.4f}", fontsize=13)

        # Secondary y-axis for errors
        ax2 = ax1.twinx()
        ax2.scatter(y_exp, errors, c='red', s=40, alpha=0.8, label="Prediction Error")
        ax2.set_ylabel("Prediction Error (Ha)", color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')

        # Annotate molecule names
        for i, n in enumerate(names):
            ax1.text(y_exp[i], y_pred_exp[i], n, fontsize=8, ha='right', va='bottom', alpha=0.7)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.tight_layout()
        plt.show()


    return y_pred_exp, r2_test


# === Run everything ===
atom_labels, bond_labels = get_dynamic_labels(fitting_dict, test_dict)
coeffs, contributions, y_pred, r2_fit = fit_atomic_bond_contributions(fitting_dict, atom_labels, bond_labels)
y_pred_test, r2_test = predict_test_set(test_dict, atom_labels, bond_labels, coeffs)
