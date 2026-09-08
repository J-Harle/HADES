---
title: "HADES: High-throughput Analysis for the Design of Energetic Systems"
tags:
  - Python
  - computational chemistry
  - energetic materials
  - molecular design
  - thermochemistry
authors:
  - name: Joshua T. Harle
    affiliation: 1
affiliations:
  - name: School of Chemistry, University of Birmingham, Birmingham, United Kingdom
    index: 1
bibliography: paper.bib
---

# Summary

HADES is an open-source Python workflow for generating and screening candidate
organic energetic molecules. It combines cheminformatics, machine-learned
atomistic simulation, isodesmic thermochemistry, empirical detonation models,
and a vibrational up-pumping model behind a single command-line interface.
Intermediate structures and tabular results remain inspectable so researchers
can reproduce individual stages and replace a model without rewriting the
entire workflow.

# Statement of need

Computational energetic-material design requires several quantities that are
normally produced by separate packages and research scripts: valid substituted
structures, optimised geometries, vibrational spectra, gas-phase formation
enthalpies, density, oxygen balance, detonation performance, and sensitivity.
Moving large molecular sets between these tools creates opportunities for
identifier, unit, and provenance errors. HADES is intended for computational
chemists who need a repeatable screening workflow for CHNO molecular energetic
materials. It standardises the stage interfaces while retaining CSV and XYZ
outputs that can be inspected with ordinary scientific tools.

# State of the field

HADES uses RDKit for cheminformatics [@rdkit], the Atomic Simulation Environment
for atomistic workflows and thermochemistry [@ase], and MACE-OFF for
machine-learned molecular energies and forces [@maceoff]. scikit-learn provides
the density regressor [@sklearn]. These mature projects provide general
algorithms but do not supply an integrated energetic-material workflow.
HADES contributes the domain-specific orchestration, reference-reaction search,
property models, input/output conventions, and provenance needed to combine
them. It therefore complements rather than duplicates the underlying packages.

# Software design

HADES uses a stage-oriented, file-backed architecture. A small top-level command
constructs an explicit execution plan and launches independent stages. CSV files
act as durable checkpoints for molecular properties, while per-molecule
directories contain geometries, Hessians, and vibrational data. This design was
chosen over an opaque monolithic process because high-throughput calculations
may run for many hours on HPC systems and need to resume after partial failure.
The trade-off is that stable schemas and atomic file updates become part of the
public interface; automated tests therefore check identifier compatibility and
preservation of existing data.

The implemented property layers include the standard oxygen-balance expression,
isodesmic formation-enthalpy estimation, a gradient-boosted density model, and
Kamlet--Jacobs detonation relations [@kamletjacobs]. Method-defining parameters,
including reaction-energy filtering, reaction weighting, and impact-calibration
coefficients, are exposed through the command line.

# Research impact

HADES has been used in the authors' ongoing in-silico study of 100,000 candidate
energetic molecules and to connect property prediction with functional-group
screening. Releasing the workflow, reference inputs, and validation procedures
allows those calculations to be reproduced and provides a foundation for
comparison with alternative thermochemistry, density, and sensitivity models.
Before submission, the associated preprint, archived benchmark data, and
software release DOI will be linked here.

# AI usage disclosure

ChatGPT Images 2.0 was used to generate the project logo. ChatGPT 5.6 Sol was
used to assist with initial docstrings. OpenAI Codex (GPT-5) assisted with
packaging refactoring, test scaffolding, documentation drafting, and preparation
of this software-paper draft. The human authors must review, edit, and validate
all AI-assisted outputs and confirm before submission that they made the core
scientific and software-design decisions.

# Acknowledgements

<!-- Add funding awards, computational facilities, supervisors, collaborators,
and any non-author contributions before submission. -->

# References
