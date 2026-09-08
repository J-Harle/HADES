<p align="center">
  <img width="460" alt="HADES logo" src="https://github.com/user-attachments/assets/324c25e5-314d-4eaa-90b7-ac1ea672e600">
</p>

# HADES

**High-throughput Analysis for the Design of Energetic Systems**

[![tests](https://github.com/J-Harle/HADES/actions/workflows/tests.yml/badge.svg)](https://github.com/J-Harle/HADES/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

HADES is a research workflow for generating and screening candidate organic
energetic molecules. It connects molecular generation, MACE/ASE geometry and
vibrational calculations, gas-phase enthalpy estimation, molecular descriptors,
density prediction, oxygen balance, Kamlet--Jacobs detonation properties, and an
up-pumping impact-sensitivity model through one command-line interface.

> **Status:** HADES is research software under active development. Predictions
> are model estimates, not safety classifications, and must be validated for the
> chemical domain and intended use.

## Statement of need

Energetic-material screening commonly requires researchers to move data between
separate cheminformatics, atomistic simulation, thermochemistry, and empirical
property-prediction tools. That makes large studies difficult to reproduce and
encourages one-off scripts with inconsistent molecule identifiers and units.
HADES provides a transparent, file-backed pipeline for CHNO molecular candidates
so that each stage can be run independently, inspected, repeated, or replaced.
It is intended for computational chemists studying molecular energetic materials.

HADES builds on RDKit, ASE, MACE, NumPy, SciPy, pandas, and scikit-learn; it does
not replace those packages. Its contribution is the domain-specific workflow and
the integration of energetic-material property models.

## Capabilities

| Stage | CLI option | Input required | Main output |
|---|---|---|---|
| Molecule generation | `--generate N` | Bundled cores/substituents | Candidate SMILES CSV |
| Optimisation | `--optimise-generated` | `CID`, `SMILES` | Optimised XYZ and thermochemistry |
| Vibrations | `--vibration` | Optimised XYZ | Frequencies and eigenvectors |
| Impact sensitivity | `--impact-sensitivity` | Vibrational outputs | Predicted H50 |
| Oxygen balance | `--oxygen-balance` | `SMILES` | Oxygen balance (%) |
| Enthalpy of formation | `--enthalpy-of-formation` | Thermochemistry outputs | Gas-phase Hf |
| Generic descriptors | `--generic-properties` | `SMILES` | RDKit descriptors |
| Detonation properties | `--detonation-properties` | Hf and optimised XYZ | Density, D, and P |

## Installation

Clone the repository and create the tested Conda environment:

```bash
git clone https://github.com/J-Harle/HADES.git
cd HADES
conda env create -f hades.yml
conda activate hades
hades --help
```

For stages that do not use MACE, a lighter editable installation is available:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install the MACE extra for optimisation, vibration, and MACE thermochemistry:

```bash
python -m pip install -e ".[mace]"
```

The MACE-OFF23 small model is downloaded on first use into the user cache. Set
`HADES_CACHE_DIR` to choose another cache directory, which is useful on HPC
systems.

## Quick start

The smallest workflow calculates properties directly from SMILES:

```bash
cp examples/minimal.csv molecules.csv
hades --input molecules.csv --oxygen-balance --generic-properties
```

A computational workflow can then add optimisation, thermochemistry, and
detonation estimates:

```bash
hades \\
  --input molecules.csv \\
  --optimise-generated \\
  --enthalpy-of-formation \\
  --detonation-properties \\
  --cpus 8
```

Impact-sensitivity intermediates are removed after a successful calculation.
Pass `--keep-intermediates` to retain them. The calibrated coefficients can be
set explicitly using `--h50-a` and `--h50-b`.

## Input and output conventions

Input CSV files require:

- `CID`: a unique, filesystem-safe molecule identifier; and
- `SMILES`: an RDKit-readable molecular structure.

HADES also recognises `FILENAME`, `molecule`, or `MOLECULE` in older
datasets. Property stages append columns to the input CSV. Expensive structure
outputs are stored under `OPTIMISED_STRUCTURES/<name>/<CID>/`.

## Scientific documentation

- [Methods and equations](docs/methods.md)
- [Model and reference-data provenance](docs/model-and-data.md)
- [Reproducibility and release checklist](docs/publication-checklist.md)
- [Draft software paper](paper/paper.md)

## Testing

```bash
python -m pip install -e ".[test]"
python -m compileall -q hades.py MODULES
pytest
```

Tests run on Python 3.10 and 3.12 in GitHub Actions. The current suite covers
core equations, CSV schema preservation, reaction enumeration, and packaging
smoke checks. Scientific benchmark datasets and end-to-end MACE regression tests
remain required before journal submission.

## Citation, support, and licence

Citation metadata are provided in [CITATION.cff](CITATION.cff). Before a paper
submission, create a tagged GitHub release and archive that exact release with
Zenodo or another DOI-issuing repository.

Please use [GitHub Issues](https://github.com/J-Harle/HADES/issues) for bugs,
questions, or feature requests and see [CONTRIBUTING.md](CONTRIBUTING.md) before
proposing code changes.

HADES is released under the [MIT License](LICENSE).

## AI-use disclosure

The logo was generated with ChatGPT Images 2.0. Docstrings were initially
assisted by ChatGPT 5.6 Sol. The publication-readiness refactor, test scaffolding,
and documentation received assistance from OpenAI Codex (GPT-5). Human authors
remain responsible for reviewing, validating, and editing all assisted material;
the formal disclosure for a software-paper submission is included in
`paper/paper.md`.
