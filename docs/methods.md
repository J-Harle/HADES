# Methods and equations

This document describes the behaviour implemented in HADES 0.1.0. A publication
should cite the original method sources and report every non-default parameter.

## Molecular generation

The generator samples a bundled molecular core and performs between one and four
substitutions using the bundled functional-group and heteroatom definitions.
RDKit sanitisation and canonical SMILES remove invalid structures and duplicates.
The synthetic-accessibility score is calculated using RDKit's contributed
SA_Score implementation. Generation is stochastic; a release-quality experiment
must record the random seed and the exact core/substituent files.

## Initial geometry and MACE optimisation

For each SMILES, HADES adds explicit hydrogen atoms, generates ten ETKDGv3
conformers with RDKit seed 42, and performs a short MMFF94 optimisation
(25 iterations). UFF is used when MMFF94 parameters are unavailable. The
lowest force-field-energy conformer is converted to ASE and optimised with the
MACE-OFF23 small model using BFGS and an `fmax` threshold of
0.001 eV Å⁻¹.

ASE finite-displacement vibrations provide the Hessian, frequencies, and
vibrational energies. Thermochemical quantities are calculated at 298.15 K and
100 kPa using `IdealGasThermo`, currently with nonlinear geometry, symmetry
number 1, and spin 0. These assumptions must be checked for each chemical class.

## Oxygen balance

For a compound with elemental composition C\\(_a\\)H\\(_b\\)N\\(_c\\)O\\(_d\\)
and molecular mass \\(M\\), HADES calculates

\\[
\\mathrm{OB}(\\%) = \\frac{-1600}{M}\\left(2a + \\frac{b}{2} - d\\right).
\\]

Implicit hydrogen atoms are made explicit before the composition is counted.
The current implementation is intended for CHNO molecules.

## Gas-phase enthalpy of formation

HADES represents target and reference molecules by atom counts and
element-pair/bond-order counts, then enumerates balanced isodesmic reactions.
For each reaction,

\\[
\\Delta E_{\\mathrm{rxn}} = \\sum E_{\\mathrm{products}}
                            - \\sum E_{\\mathrm{reactants}},
\\]

and the target formation enthalpy is

\\[
\\Delta H_{f,\\mathrm{target}} =
\\sum \\Delta H_{f,\\mathrm{reactants}} -
\\sum \\Delta H_{f,\\mathrm{other\\ products}} -
\\Delta E_{\\mathrm{rxn}}.
\\]

By default, reactions satisfy \\(|\\Delta E_{\\mathrm{rxn}}| < 0.005\\) Ha and
are inverse-energy weighted:

\\[
w_i = \\frac{1}{|\\Delta E_i| + 10^{-6}}.
\\]

`--weighting boltzmann` instead applies

\\[
w_i = \\exp\\left[-\\frac{|\\Delta E_i|}{k_\\mathrm{B}T}\\right],
\\]

and `--weighting uniform` assigns equal weights. The cutoff, weighting,
temperature, and maximum reactant/product counts are CLI parameters and must be
reported in derived work.

## Density and detonation properties

The base molecular-density feature is molecular mass divided by RDKit's
grid-based volume from the optimised XYZ geometry. A bundled
`GradientBoostingRegressor` maps that value and nine RDKit descriptors to a
predicted crystal density \\(\\rho\\).

HADES estimates detonation products and then evaluates the Kamlet--Jacobs
relations

\\[
\\Phi = N\\sqrt{\\bar{M}}\\sqrt{Q},
\\]

\\[
D = 1.01\\sqrt{\\Phi}\\left(1 + 1.30\\rho\\right),
\\qquad
P = 1.558\\Phi\\rho^2,
\\]

where \\(N\\) is the amount of gaseous products per gram, \\(\\bar{M}\\) is their
average molar mass, and \\(Q\\) is the detonation energy. The implemented output
units are km s⁻¹ for \\(D\\) and GPa for \\(P\\).

## Impact sensitivity

HADES reads positive vibrational frequencies, constructs a broadened density of
states, applies Bose--Einstein scaling, performs two self-convolutions, and
integrates the projected up-pumped spectrum. The current ω cutoff stage selects
the highest frequency between 100 and 200 cm⁻¹, with a nearest-to-200 cm⁻¹
fallback.

For the integrated metric \\(y\\), impact sensitivity is obtained from

\\[
y = a/H_{50} + b,
\\qquad
H_{50} = \\frac{a}{y-b}.
\\]

The CLI defaults are \\(a=0.065785\\) and \\(b=0.011077\\). Non-finite or negative
predictions are recorded as missing. These calibration constants and the cutoff
definition must be validated and versioned with the benchmark dataset used in a
publication.
