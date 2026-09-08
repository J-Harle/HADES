import argparse
import subprocess
import sys
import os
import csv
import math
import tempfile


ID_COLUMN_CANDIDATES = ("CID", "FILENAME", "molecule", "MOLECULE")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UPPUMPING impact sensitivity workflow"
    )

    parser.add_argument(
        "--input", "-i",
        default="hades_out.csv",
        help="Input CSV file. Default: hades_out.csv"
    )

    parser.add_argument(
        "--dir", "-dir",
        dest="base_dir",
        default="OPTIMISED_STRUCTURES/HADES",
        help="Directory containing optimised molecule folders. Default: OPTIMISED_STRUCTURES/HADES"
    ) 

    parser.add_argument(
        "--h50-a",
        type=float,
        default=0.065785,
        help="Gradient of the calibrated H50 relationship. Default: 0.065785",
    )

    parser.add_argument(
        "--h50-b",
        type=float,
        default=0.011077,
        help="Intercept of the calibrated H50 relationship. Default: 0.011077",
    )

    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep the raw and prediction CSV files produced by the workflow",
    )

    return parser.parse_args()


def run_script(command, script_name, cwd=None):
    # print("\nRunning command:")
    # print(" ".join(command))

    result = subprocess.run(command, cwd=cwd)

    if result.returncode != 0:
        print(f"[ERROR] {script_name} failed. Stopping execution.")
        sys.exit(result.returncode)


def _find_id_column(fieldnames):
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in (fieldnames or []):
            return candidate

    raise KeyError(
        "No molecule identifier column found. Expected one of "
        f"{ID_COLUMN_CANDIDATES}; available columns: {fieldnames}"
    )


def merge_predictions(input_csv, predictions_csv):
    """Append impact predictions without discarding existing input columns."""
    with open(input_csv, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        input_rows = list(reader)
        input_fields = list(reader.fieldnames or [])

    with open(predictions_csv, "r", encoding="utf-8", newline="") as prediction_file:
        reader = csv.DictReader(prediction_file)
        prediction_rows = list(reader)
        prediction_id_column = _find_id_column(reader.fieldnames)

    input_id_column = _find_id_column(input_fields)
    prediction_map = {
        str(row.get(prediction_id_column, "")).strip(): row.get("predicted_H50", "")
        for row in prediction_rows
    }

    output_column = "predicted_H50 /J"
    if output_column not in input_fields:
        input_fields.append(output_column)

    for row in input_rows:
        molecule_id = str(row.get(input_id_column, "")).strip()
        prediction = prediction_map.get(molecule_id, "")

        try:
            numeric_prediction = float(prediction)
        except (TypeError, ValueError):
            numeric_prediction = math.nan

        row[output_column] = (
            numeric_prediction
            if math.isfinite(numeric_prediction) and numeric_prediction >= 0
            else ""
        )

    input_directory = os.path.dirname(os.path.abspath(input_csv))
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=input_directory,
        prefix=".hades-impact-",
        suffix=".csv",
        delete=False,
    ) as temporary_file:
        writer = csv.DictWriter(temporary_file, fieldnames=input_fields)
        writer.writeheader()
        writer.writerows(input_rows)
        temporary_path = temporary_file.name

    os.replace(temporary_path, input_csv)


def main():
    args = parse_args()

    arg_modules_dir = os.path.dirname(os.path.abspath(__file__))

    unknowns_dir = os.path.join(
        arg_modules_dir,
        "UPPUMPING",
#        "UNKNOWNS"
    )

    def resolve_project_path(path):
        if os.path.isabs(path):
            return path
        return os.path.abspath(path)

    input_csv = resolve_project_path(args.input)
    base_dir = resolve_project_path(args.base_dir)

    input_stem = os.path.splitext(os.path.basename(input_csv))[0]

    working_dir = os.path.dirname(input_csv)

    raw_csv = os.path.join(
        working_dir,
        f"{input_stem}_raw.csv"
    )

    predictions_csv = os.path.join(
        working_dir,
        f"{input_stem}_impact_predictions.csv"
    )

    # print("\nUPPUMPING settings")
    # print("-" * 40)
    # print(f"Input CSV: {input_csv}")
    # print(f"Optimised structure directory: {base_dir}")
    # print(f"Raw UPPUMPING CSV: {raw_csv}")
    # print("-" * 40)

    # ---------------------------------------------------------
    # 1. Read optimised files and create *_raw.csv
    # ---------------------------------------------------------

    read_opt_files_script = os.path.join(
        unknowns_dir,
        "read_opt_files.py"
    )

    run_script(
        [
            sys.executable,
            read_opt_files_script,
            "-i", input_csv,
            "-dir", base_dir,
            "-o", raw_csv
        ],
        "read_opt_files.py",
        cwd=unknowns_dir
    )

    # ---------------------------------------------------------
    # 2. Process the raw CSV through the UPPUMPING workflow
    # ---------------------------------------------------------

    processing_scripts = [
        "omax.py",
        "be_scale.py",
        "convolutions.py",
        "integrate.py",
    ]

    for script in processing_scripts:
        script_path = os.path.join(unknowns_dir, script)

        run_script(
            [
                sys.executable,
                script_path,
                "-i", raw_csv
            ],
            script,
            cwd=unknowns_dir
        )

    # ---------------------------------------------------------
    # 3. Predict/write impact sensitivity back to original CSV
    # ---------------------------------------------------------

    plotting_script = os.path.join(
        unknowns_dir,
        "unknown_plotting.py"
    )

    run_script(
        [
            sys.executable,
            plotting_script,
            "-i", raw_csv,
            "-o", predictions_csv,
            "--a", str(args.h50_a),
            "--b", str(args.h50_b),
        ],
        "unknown_plotting.py",
        cwd=unknowns_dir
    )

    merge_predictions(input_csv, predictions_csv)

    if not args.keep_intermediates:
        for intermediate_path in (raw_csv, predictions_csv):
            if os.path.isfile(intermediate_path):
                os.remove(intermediate_path)


if __name__ == "__main__":
    main()
