""" Main for HADES - High-throughput Analysis for Discovery of Energetic Systems """

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
                                      
 High-throughput Analysis for Discovery of Energetic Systems
""" 
print(header)

parser = argparse.ArgumentParser(
    description="HADES: High-throughput Analysis for Discovery of Energetic Systems"
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

    if args.generate.lower() == "y":
        subprocess.run(["python", f"{parent_dir}/MODULES/substitution.py",
                        "-i", args.input or "none",
                        "-o", args.output])
        print(header)


    if args.optimise_generated.lower() == "y":
        print("\n[2/6] Running optimisation module...")
        subprocess.run(["python", f"{parent_dir}/create_object.py", "-i", args.output])
        print(header)


    if args.vibration.lower() == "y":
        print("\n[3/6] Running vibrations module...")
        subprocess.run(["python", "MODULES/vibrations.py", "-i", args.output])
        print(header)

    if args.impact_sensitivity.lower() == "y":
        print("\n[4/6] Running impact sensitivity module...")
        subprocess.run(["python", "MODULES/uppumping.py", "-i", args.output])
        print(header)

    if args.oxygen_balance.lower() == "y":
        print("\n[5/6] Running oxygen balance module...")
        subprocess.run(["python", "MODULES/oxygen_balance.py", "-i", args.output])
        print(header)

    if args.enthalpy_of_formation.lower() == "y":
        print("\n[6/6] Running enthalpy of formation module...")
        subprocess.run(["python", "MODULES/enthalpy_of_formation.py", "-i", args.output])
        print(header)

if __name__ == "__main__":
    parent_dir = os.getcwd()

    main()
