import pandas as pd

from MODULES.generic_properties import calc_genprop


def test_invalid_smiles_produces_missing_features_without_crashing():
    features = calc_genprop(pd.DataFrame({"SMILES": ["not-a-smiles"]}))

    assert len(features) == 1
    assert pd.isna(features.loc[0, "mol_wt"])
