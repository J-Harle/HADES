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
parser.add_argument("--input", "-i", type=str, required=False,
                    help="Input file path (if not provided, molecule generation will be triggered)")
parser.add_argument("--output", "-o", type=str, required=False,
                    help="Output file path (default: hades_out.csv)")

# Optional workflow stages — all default to 'n'
parser.add_argument("--generate", "-g", type=str, default="n",
                    help="Generate new molecules (y/n)")

parser.add_argument("--optimise-generated", "-opt", type=str, default="n",
                    help="Optimise and create xyz files for generated molecules (y/n)")

parser.add_argument("--vibration", "-v", type=str, default="n",
                    help="Calculate vibration frequencies (y/n)")

parser.add_argument("--impact-sensitivity", "-s", type=str, default="n",
                    help="Calculate impact sensitivity (y/n)")

parser.add_argument("--oxygen-balance", "-ob", type=str, default="n",
                    help="Calculate oxygen balance (y/n)")

parser.add_argument("--enthalpy-of-formation", "-eof", type=str, default="n",
                    help="Calculate enthalpy of formation (y/n)")

args = parser.parse_args()

# Default output file if not provided
if args.output is None:
    args.output = "hades_out.csv"

# Auto-enable generation if no input file provided
if args.input is None:
    print("No input file provided; assuming molecule generation is required.")
    args.generate = "y"

def main():
    print(f"Input: {args.input if args.input else 'None (auto-generation mode)'}")
    print(f"Output: {args.output}")

    # Create a list of (step_name, flag, command) for enabled modules
    steps = []

    if args.generate.lower() == "y":
        steps.append(("Generating molecules", ["python", f"{parent_dir}/MODULES/substitution.py",
                                               "-i", args.input or "none",
                                               "-o", args.output]))

    if args.optimise_generated.lower() == "y":
        steps.append(("Optimising generated molecules", ["python", f"{parent_dir}/MODULES/create_object.py", "-i", args.output]))

    if args.vibration.lower() == "y":
        steps.append(("Calculating vibrations", ["python", "MODULES/vibrations.py", "-i", args.output]))

    if args.impact_sensitivity.lower() == "y":
        steps.append(("Calculating impact sensitivity", ["python", "MODULES/uppumping.py", "-i", args.output]))

    if args.oxygen_balance.lower() == "y":
        steps.append(("Calculating oxygen balance", ["python", "MODULES/oxygen_balance.py", "-i", args.output]))

    if args.enthalpy_of_formation.lower() == "y":
        steps.append(("Calculating enthalpy of formation", ["python", "MODULES/enthalpy_of_formation.py", "-i", args.output]))

    total_steps = len(steps)

    for i, (description, command) in enumerate(steps, start=1):
        print(f"\n[{i}/{total_steps}] {description}...")
        subprocess.run(command)
        print(header)


if __name__ == "__main__":
    parent_dir = os.getcwd()

    main()
