import csv

from MODULES.det_v_p import read_csv


def test_read_csv_accepts_public_cid_schema(tmp_path):
    csv_path = tmp_path / "molecules.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["CID", "SMILES", "Hf /kJmol-1"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "CID": "TNT",
                "SMILES": "Cc1cc(cc(c1)[N+](=O)[O-])[N+](=O)[O-]",
                "Hf /kJmol-1": "12.5",
            }
        )

    assert read_csv(csv_path) == [
        {
            "cid": "TNT",
            "smiles": "Cc1cc(cc(c1)[N+](=O)[O-])[N+](=O)[O-]",
            "eof": 12.5,
        }
    ]
