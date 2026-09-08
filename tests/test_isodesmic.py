import numpy as np
import pytest

from MODULES.isodesmic import (
    build_combo_lookup,
    filter_and_average,
    find_isodesmics,
)


def test_combo_lookup_preserves_duplicate_balance_vectors():
    combo_vectors = np.array([[1], [1], [2]], dtype=np.int16)
    lookup = build_combo_lookup(combo_vectors)

    assert lookup[(1,)] == [0, 1]

    solutions = find_isodesmics(
        combo_vecs=combo_vectors,
        combo_names=[["a"], ["b"], ["reactant"]],
        target_vec=np.array([1], dtype=np.int16),
        target_smiles="target",
        combo_lookup=lookup,
        max_product_r=1,
    )

    matched_products = [
        solution["products"]
        for solution in solutions
        if solution["reactants"] == ["reactant"]
    ]
    assert matched_products == [["target", "a"], ["target", "b"]]


def test_weighting_schemes_are_explicit_and_distinct():
    solutions = [
        {"Delta_E_Ha": 0.0, "Hf_target": 0.1},
        {"Delta_E_Ha": 0.004, "Hf_target": 0.3},
    ]

    uniform_mean, _, _ = filter_and_average(solutions, weighting="uniform")
    inverse_mean, _, _ = filter_and_average(solutions, weighting="inverse")
    boltzmann_mean, _, _ = filter_and_average(solutions, weighting="boltzmann")

    assert uniform_mean == pytest.approx(0.2)
    assert inverse_mean < boltzmann_mean < uniform_mean
