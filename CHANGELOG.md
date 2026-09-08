# Changelog

All notable changes to HADES will be documented in this file. The project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Standards-compliant Python packaging and a `hades` console command.
- Automated tests and continuous integration.
- Citation, licence, contribution, and software-paper documentation.
- Configurable isodesmic reaction limits, energy cutoff, and weighting scheme.
- Atomic MACE model download into a user-writable cache.

### Fixed

- Syntax error in the generic-property stage.
- Inconsistent `CID`/`molecule` handling in detonation calculations.
- Loss of existing CSV columns after impact-sensitivity prediction.
- Loss of distinct isodesmic reactions with identical feature vectors.
- Ignored `--nrows` option in the enthalpy workflow.
- Negative impact-sensitivity predictions are now recorded as missing values.

## [0.1.0] - Unreleased

First citable development release.
