""" Main for HADES - High-throughput Analysis for the Design of Energetic Systems """

import argparse
import subprocess
import os
import sys

header = r"""
  _    _              _____    ______    _____ 
 | |  | |     /\     |  __ \  |  ____|  / ____|
 | |__| |    /  \    | |  | | | |__    | (___  
 |  __  |   / /\ \   | |  | | |  __|    \___ \ 
 | |  | |_ / ____ \ _| |__| |_| |____ _ ____) |
 |_|  |_(_)_/    \_(_)_____/(_)______(_)_____/ 
                                      
 High-throughput Analysis for the Design of Energetic Systems
"""

print(header)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HADES: High-throughput Analysis for the Design of Energetic Systems"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=False,
        default=None,
        help="Input CSV file path. If not provided, molecule generation will be triggered."
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=False,
        default="hades_out.csv",
        help="Output CSV file path. Default: hades_out.csv"
    )

    parser.add_argument(
        "--generate", "-g",
        nargs="?",
        const=1000,
        default=None,
        type=int,
        metavar="N",
        help="Generate N molecules. If -g is used without N, defaults to 1000."
    )

    parser.add_argument(
        "--optimise-generated", "-opt",
        action="store_true",
        help="Optimise and create xyz files for generated/input molecules"
    )

    parser.add_argument(
        "--vibration", "-v",
        action="store_true",
        help="Calculate vibration frequencies"
    )

    parser.add_argument(
        "--impact-sensitivity", "-s",
        action="store_true",
        help="Calculate impact sensitivity"
    )

    parser.add_argument(
        "--oxygen-balance", "-ob",
        action="store_true",
        help="Calculate oxygen balance"
    )

    parser.add_argument(
        "--enthalpy-of-formation", "-eof",
        action="store_true",
        help="Calculate enthalpy of formation"
    )

    parser.add_argument(
        "--generic-properties", "-gprop",
        action="store_true",
        help="Calculate generic property information"
    )

    parser.add_argument(
        "--detonation-properties", "-det",
        action="store_true",
        help="Calculate detonation property information"
    )

    parser.add_argument(
        "--cpus", "-c",
        type=int,
        default=-1,
        help="Number of CPUs to use for parallel steps. Use -1 for all available cores. Default: -1"
    )

    parser.add_argument(
        "--outdir", "-dir",
        type=str,
        default="HADES",
        help=(
            "Subdirectory inside OPTIMISED_STRUCTURES where optimised structures "
            "are written/read. Default: HADES"
        )
    )

    return parser.parse_args()


def print_run_plan(steps, args, active_input):
    print("\nHADES run plan")
    print("-" * 40)

    print(f"Input file: {active_input}")
    print(f"Output file: {args.output}")
    print(f"Optimised structure directory: {args.outdir}")

    if args.cpus == -1:
        print("Running on: all available CPU cores")
    else:
        print(f"Running on: {args.cpus} CPU cores")

    print("\nSteps to run:")

    if not steps:
        print("  No steps selected.")
    else:
        for i, (description, _) in enumerate(steps, start=1):
            print(f"  {i}. {description}")

    print("-" * 40)


def main():
    args = parse_args()

    project_dir = os.getcwd()

    def resolve_path(path):
        """
        Resolve normal file paths relative to the HADES project directory.
        Used for CSV input/output files.
        """
        if path is None:
            return None

        if os.path.isabs(path):
            return path

        return os.path.abspath(os.path.join(project_dir, path))

    def resolve_outdir(outdir):
        """
        Force all structure directories to live inside:

            <project_dir>/OPTIMISED_STRUCTURES/<outdir>

        Examples
        --------
        -dir test
        -> <project_dir>/OPTIMISED_STRUCTURES/test

        -dir STORM/SMALL_MODEL
        -> <project_dir>/OPTIMISED_STRUCTURES/STORM/SMALL_MODEL

        -dir OPTIMISED_STRUCTURES/test
        -> <project_dir>/OPTIMISED_STRUCTURES/test
        """

        if outdir is None or outdir.strip() == "":
            outdir = "HADES"

        # Normalise slashes and remove trailing slashes
        outdir = os.path.normpath(outdir)

        # Prevent absolute paths from escaping OPTIMISED_STRUCTURES
        # Example: /tmp/test becomes tmp/test inside OPTIMISED_STRUCTURES
        outdir = outdir.lstrip(os.sep)

        parts = outdir.split(os.sep)

        # If user writes -dir OPTIMISED_STRUCTURES/test,
        # strip the duplicated OPTIMISED_STRUCTURES part.
        if parts[0] == "OPTIMISED_STRUCTURES":
            parts = parts[1:]

        # Prevent accidental parent-directory escape using ..
        if any(part == ".." for part in parts):
            raise ValueError(
                "[ERROR] Do not use '..' in -dir. "
                "The structure directory must stay inside OPTIMISED_STRUCTURES."
            )

        clean_outdir = os.path.join(*parts) if parts else "HADES"

        return os.path.abspath(
            os.path.join(project_dir, "OPTIMISED_STRUCTURES", clean_outdir)
        )

    # Auto-enable generation if no input file provided
    if args.input is None and args.generate is None:
        print("No input file provided; assuming molecule generation is required.")
        args.generate = 1000

    # Resolve paths relative to the HADES directory
    args.output = resolve_path(args.output)
    args.outdir = resolve_outdir(args.outdir)

    if args.input is not None:
        args.input = resolve_path(args.input)

    # Decide which CSV downstream scripts should read
    if args.generate is not None:
        active_input = args.output
    else:
        active_input = args.input

    if active_input is None:
        active_input = args.output

    print(f"Input: {active_input}")
    print(f"Output: {args.output}")
    print(f"Structure directory: {args.outdir}")

    steps = []

    if args.generate is not None:
        steps.append((
            f"Generating {args.generate} molecules",
            [
                sys.executable,
                "-m",
                "hades.substitution",
                "-o",
                args.output,
                "-n",
                str(args.generate),
            ]
        ))

    if args.optimise_generated:
        steps.append((
            "Optimising molecules",
            [
                sys.executable,
                "-m",
                "hades.create_object",
                "-i",
                active_input,
                "--outdir",
                args.outdir,
                "-c",
                str(args.cpus),
            ]
        ))

    if args.vibration:
        steps.append((
            "Calculating vibrations",
            [
                sys.executable,
                "-m",
                "hades.vibration",
                "-i",
                active_input,
                "-dir",
                args.outdir,
                "-c",
                str(args.cpus),
            ]
        ))

    if args.impact_sensitivity:
        steps.append((
            "Predicting impact sensitivity",
            [
                sys.executable,
                "-m",
                "hades.uppumping",
                "-i",
                active_input,
                "-dir",
                args.outdir,
            ]
        ))

    if args.oxygen_balance:
        steps.append((
            "Calculating oxygen balance",
            [
                sys.executable,
                "-m",
                "hades.oxygen_balance",
                "-i",
                active_input,
            ]
        ))

    if args.enthalpy_of_formation:
        command = [
            sys.executable,
            "-m",
            "hades.isodesmic",
            "-i",
            active_input,
            "-dir",
            args.outdir,
        ]

        if args.generate is not None:
            command.extend([
                "-n",
                str(args.generate),
            ])

        steps.append((
            "Predicting enthalpy of formation",
            command
        ))

    if args.generic_properties:
        steps.append((
            "Predicting generic properties",
            [
                sys.executable,
                "-m",
                "hades.generic_properties",
                "-i",
                active_input,
            ]
        ))

    if args.detonation_properties:
        steps.append((
            "Predicting detonation properties",
            [
                sys.executable,
                "-m",
                "hades.det_v_p",
                "-i",
                active_input,
                "-dir",
                args.outdir,
            ]
        ))

    total_steps = len(steps)

    print_run_plan(steps, args, active_input)

    for i, (description, command) in enumerate(steps, start=1):
        print(f"\n[{i}/{total_steps}] {description}...")
        subprocess.run(command, check=True, cwd=project_dir)
        print(header)


if __name__ == "__main__":
    main()
