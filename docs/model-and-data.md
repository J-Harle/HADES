# Model and reference-data provenance

HADES includes two scientific assets whose exact versions affect predictions.
They are packaged with the software so that a tagged release is self-contained.

## Density model

Path: `MODULES/TOOLS/DENSITY/gbt_density_model.pkl`

- SHA-256:
  `ff6112da5ee765b3a15fd94167dcc4165827834db0e4ef7bcc9a0129095053ec`
- Serialisation library: joblib
- scikit-learn training version: 1.6.1
- Estimator: `GradientBoostingRegressor`
- Features: `MolecularDensity`, `NumHAcceptors`, `NumHDonors`, `Phi`,
  `NumAromaticRings`, `TPSA`, `MolLogP`, `MaxAbsPartialCharge`,
  `MinAbsPartialCharge`, and `NumAliphaticRings`
- Recorded parameters: 1000 estimators, learning rate 0.03, maximum depth 4,
  subsample 0.8, squared-error loss, random seed 42

The package pins scikit-learn 1.6.1 because loading a pickle with a different
version is not guaranteed to be correct. Pickle/joblib files can execute code
when loaded; only use the model distributed by a trusted HADES release and
verify its checksum.

Before journal submission, add the training-data DOI or repository, licence,
curation rules, train/test splitting strategy, applicability domain, complete
performance table, and a script that reproduces the model from source data.

## Isodesmic reference table

Path: `MODULES/TOOLS/EOF/isodesmic.csv`

- SHA-256:
  `85f192f36e6a363b17fae0e1182b6e82b2c6ec6d5fd49b3be5535e9212dbdda8`
- Rows: 227 reference molecules
- Source labels in the table: NIST (171), ATcT (54), Byrd 2006 (1), and one
  DOI-labelled literature record
- Required values: experimental formation enthalpy, MACE energy, and SMILES

Before journal submission, replace abbreviated source labels with complete,
row-level citations and stable record identifiers, state the temperature and
phase of every experimental value, document all exclusions/corrections, and
provide the script used to calculate the MACE energies.

## External MACE model

Optimisation and vibration stages download `MACE-OFF23_small.model` from the
official ACEsuit `mace-off` repository on first use. The cache directory is
controlled by `HADES_CACHE_DIR`. A release should record the upstream model
commit, licence, download URL, and SHA-256 checksum rather than relying on a
moving `main` branch URL.
