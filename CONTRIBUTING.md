# Contributing to HADES

Bug reports, documentation improvements, tests, and method extensions are
welcome. Please open an issue before beginning a large scientific or API change.

## Development setup

```bash
git clone https://github.com/J-Harle/HADES.git
cd HADES
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,dev]"
```

Install `.[mace]` as well when working on optimisation, vibrational analysis,
or MACE thermochemistry.

## Checks

```bash
python -m compileall -q hades.py MODULES
pytest
ruff check .
python -m build
```

New scientific behaviour should include:

1. a unit test for the governing equation or transformation;
2. a regression test based on a small, redistributable example;
3. the reference, assumptions, units, and applicability limits in
   `docs/methods.md`; and
4. an entry in `CHANGELOG.md`.

Do not commit generated optimisation directories, intermediate CSV files, model
caches, or data that cannot be redistributed.
