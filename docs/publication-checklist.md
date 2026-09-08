# Software-paper readiness

This checklist separates repository engineering from scientific validation.

## Added in the publication-readiness pull request

- [x] Valid Python build metadata and console entry point
- [x] OSI-approved licence file
- [x] Citation metadata
- [x] Minimal, cross-platform environment specification
- [x] Automated unit tests and GitHub Actions
- [x] Example input
- [x] Contribution guidance and changelog
- [x] Draft JOSS paper with required section headings
- [x] Documented methods, units, bundled assets, and known assumptions

## Required before submission

- [ ] Decide the author list, affiliations, ORCIDs, funding, and conflicts of
      interest.
- [ ] Human-review and validate every AI-assisted code and text contribution,
      then confirm the disclosure in `paper/paper.md`.
- [ ] Reconcile the publication's isodesmic weighting with the CLI default:
      inverse, Boltzmann, or uniform.
- [ ] Confirm and version the impact calibration constants. Existing project
      notes use more than one intercept, while the current code default is
      0.011077.
- [ ] Decide whether the production ω cutoff is the current 100--200 cm⁻¹
      frequency rule or the nitro-group eigenvector-displacement method; add a
      benchmark test for the selected implementation.
- [ ] Publish the density training data, training script, splits, uncertainty,
      applicability domain, and independent validation results under a
      redistributable licence.
- [ ] Provide complete row-level provenance for the isodesmic reference table.
- [ ] Add small end-to-end regression fixtures for optimisation, vibrations,
      enthalpy, density/detonation, and impact sensitivity.
- [ ] Add benchmark outputs with tolerances and document the hardware/software
      used to produce them.
- [ ] Ask an independent user to install HADES from a clean clone and report
      problems through a public issue.
- [ ] Demonstrate research use/adoption in the paper and link the related
      energetic-materials manuscript or preprint.
- [ ] Create a tagged release, archive that exact tag with Zenodo/figshare, and
      add the DOI to `CITATION.cff` and `paper/paper.md`.

## Release sequence

1. Merge the engineering and scientific-validation pull requests.
2. Run CI from a clean tag candidate.
3. Build and inspect the wheel and source distribution.
4. Create a GitHub release (for example, `v0.1.0`).
5. Archive the release and obtain a DOI.
6. Update citation metadata and the paper with the version and DOI.
7. Submit the exact archived release for review.
