""" Main for HADES - High-throughput Analysis for the Design of Energetic Systems """

import argparse
import subprocess
import os

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

parser = argparse.ArgumentParser(
    description="HADES: High-throughput Analysis for the Design of Energetic Systems"
)

# Input and output files
parser.add_argument(
    "--input", "-i", type=str, required=False,
    help="Input file path (if not provided, molecule generation will be triggered)"
)

parser.add_argument(
    "--output", "-o", type=str, required=False,
    help="Output file path (default: hades_out.csv)"
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
    "--optimise-generated", "-opt", action="store_true",
    help="Optimise and create xyz files for generated molecules"
)

parser.add_argument(
    "--vibration", "-v", action="store_true",
    help="Calculate vibration frequencies"
)

parser.add_argument(
    "--impact-sensitivity", "-s", action="store_true",
    help="Calculate impact sensitivity"
)

parser.add_argument(
    "--oxygen-balance", "-ob", action="store_true",
    help="Calculate oxygen balance"
)

parser.add_argument(
    "--enthalpy-of-formation", "-eof", action="store_true",
    help="Calculate enthalpy of formation"
)

parser.add_argument(
    "--generic-properties", "-gprop", action="store_true",
    help="Calculate generic property information"
)

parser.add_argument(
    "--detonation-properties", "-det", action="store_true",
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
    default="OPTIMISED_STRUCTURES/HADES",
    help="Directory where optimised structures should be written. Default: OPTIMISED_STRUCTURES/HADES"
)

args = parser.parse_args()

# Default output file
if args.output is None:
    args.output = "hades_out.csv"

# Auto-enable generation if no input file provided
if args.input is None and args.generate is None:
    print("No input file provided; assuming molecule generation is required.")
    args.generate = 1000


def main():
    print(f"Input: {args.input if args.input else 'None (auto-generation mode)'}")
    print(f"Output: {args.output}")

    steps = []

    if args.generate is not None:
        steps.append((
            f"Generating {args.generate} molecules",
            [
                "python",
                f"{parent_dir}/MODULES/substitution.py",
                "-i", args.input or "none",
                "-o", args.output,
                "-n", str(args.generate)
            ]
        ))

    if args.optimise_generated:
        steps.append((
            "Optimising generated molecules",
            [
                "python",
                f"{parent_dir}/MODULES/create_object.py",
                "-i", args.output,
                "--outdir", args.outdir,
                "-c", str(args.cpus)
            ]
        ))

    if args.vibration:
        steps.append((
            "Calculating vibrations",
            [
                "python",
                f"{parent_dir}/MODULES/vibration.py",
                "-i", args.output
            ]
        ))

    if args.impact_sensitivity:
        steps.append((
            "Predicting impact sensitivity",
            [
                "python",
                f"{parent_dir}/MODULES/uppumping.py",
                "-i", args.output
            ]
        ))

    if args.oxygen_balance:
        steps.append((
            "Calculating oxygen balance",
            [
                "python",
                f"{parent_dir}/MODULES/oxygen_balance.py",
                "-i", args.output
            ]
        ))

    if args.enthalpy_of_formation:
        command = [
            "python",
            f"{parent_dir}/MODULES/isodesmic.py",
            "-i", args.output,
            "-dir", args.outdir,
        ]

        if args.generate is not None:
            command.extend([
                "-n", str(args.generate)
            ])

        steps.append((
            "Predicting enthalpy of formation",
            command
        ))

    if args.generic_properties:
        steps.append((
            "Predicting generic properties",
            [
                "python",
                f"{parent_dir}/MODULES/generic_properties.py",
                "-i", args.output
            ]
        ))

    if args.detonation_properties:
        steps.append((
            "Predicting detonation properties",
            [
                "python",
                f"{parent_dir}/MODULES/det_v_p.py",
                "-i", args.output,
                "-dir", args.outdir,
            ]
        ))

    total_steps = len(steps)

    def print_run_plan(steps, args):
        print("\nHADES run plan")
        print("-" * 40)

        if args.input:
            print(f"Input file: {args.input}")
        else:
            print("Input file: None")

        print(f"Output file: {args.output}")

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

    print_run_plan(steps, args)

    for i, (description, command) in enumerate(steps, start=1):
        print(f"\n[{i}/{total_steps}] {description}...")
        subprocess.run(command, check=True)
        print(header)


if __name__ == "__main__":
    parent_dir = os.getcwd()
    main()
