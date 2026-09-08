import pytest

from MODULES.oxygen_balance import atom_counts_from_smiles, oxygen_balance


def test_tnt_oxygen_balance_from_formula():
    result = oxygen_balance(C=7, H=5, O=6, mol_weight=227.135)
    assert result == pytest.approx(-73.965, abs=0.001)


def test_atom_counts_include_implicit_hydrogen():
    counts = atom_counts_from_smiles("C")
    assert counts == {"C": 1, "H": 4}


def test_zero_molecular_weight_returns_none():
    assert oxygen_balance(C=0, H=0, O=0, mol_weight=0) is None
