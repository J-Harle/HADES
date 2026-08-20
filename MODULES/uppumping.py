import argparse
import subprocess
import sys
import os

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

    return parser.parse_args()


def run_script(command, script_name, cwd=None):
    # print("\nRunning command:")
    # print(" ".join(command))

    result = subprocess.run(command, cwd=cwd)

    if result.returncode != 0:
        print(f"[ERROR] {script_name} failed. Stopping execution.")
        sys.exit(result.returncode)


def main():
    args = parse_args()

    arg_modules_dir = os.path.dirname(os.path.abspath(__file__))

    project_dir = os.path.dirname(arg_modules_dir)

    unknowns_dir = os.path.join(
        arg_modules_dir,
        "UPPUMPING",
#        "UNKNOWNS"
    )

    def resolve_project_path(path):
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(project_dir, path))

    input_csv = resolve_project_path(args.input)
    base_dir = resolve_project_path(args.base_dir)

    input_stem = os.path.splitext(os.path.basename(input_csv))[0]

    raw_csv = os.path.join(
        unknowns_dir,
        f"{input_stem}_raw.csv"
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
            "-o", input_csv
        ],
        "unknown_plotting.py",
        cwd=unknowns_dir
    )


if __name__ == "__main__":
    main()
