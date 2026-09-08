import csv

from MODULES.uppumping import merge_predictions


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_merge_predictions_preserves_input_columns_and_blanks_negative_values(tmp_path):
    input_csv = tmp_path / "molecules.csv"
    predictions_csv = tmp_path / "predictions.csv"

    _write_csv(
        input_csv,
        ["CID", "SMILES", "Core"],
        [
            {"CID": "one", "SMILES": "C", "Core": "alkane"},
            {"CID": "two", "SMILES": "N", "Core": "amine"},
        ],
    )
    _write_csv(
        predictions_csv,
        ["molecule", "predicted_H50"],
        [
            {"molecule": "one", "predicted_H50": "12.5"},
            {"molecule": "two", "predicted_H50": "-1.0"},
        ],
    )

    merge_predictions(input_csv, predictions_csv)

    with input_csv.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows[0]["Core"] == "alkane"
    assert rows[0]["predicted_H50 /J"] == "12.5"
    assert rows[1]["Core"] == "amine"
    assert rows[1]["predicted_H50 /J"] == ""
