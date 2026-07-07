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
    "--generate", "-g", action="store_true",
    help="Generate new molecules"
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

args = parser.parse_args()

# Default output file
if args.output is None:
    args.output = "hades_out.csv"

# Auto-enable generation if no input file provided
if args.input is None:
    print("No input file provided; assuming molecule generation is required.")
    args.generate = True


def main():
    print(f"Input: {args.input if args.input else 'None (auto-generation mode)'}")
    print(f"Output: {args.output}")

    steps = []

    if args.generate:
        steps.append((
            "Generating molecules",
            [
                "python",
                f"{parent_dir}/MODULES/substitution.py",
                "-i", args.input or "none",
                "-o", args.output
            ]
        ))

    if args.optimise_generated:
        steps.append((
            "Optimising generated molecules",
            [
                "python",
                f"{parent_dir}/MODULES/create_object.py",
                "-i", args.output
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
            "Calculating impact sensitivity",
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
        steps.append((
            "Calculating enthalpy of formation",
            [
                "python",
                f"{parent_dir}/MODULES/enthalpy_of_formation.py",
                "-i", args.output
            ]
        ))

    if args.generic_properties:
        steps.append((
            "Calculating generic properties",
            [
                "python",
                f"{parent_dir}/MODULES/generic_properties.py",
                "-i", args.output
            ]
        ))

    total_steps = len(steps)

    for i, (description, command) in enumerate(steps, start=1):
        print(f"\n[{i}/{total_steps}] {description}...")
        subprocess.run(command, check=True)
        print(header)


if __name__ == "__main__":
    parent_dir = os.getcwd()
    main()
