import subprocess
import sys
import os

scripts = [
    "read_opt_files.py",
    "eigenvectors.py",
    "be_scale.py",
    "omax.py",
    "convolutions.py",
    "integrate.py",
    "plotting.py",
]

base_dir = os.path.join(os.path.dirname(__file__), "UPPUMPING")

for script in scripts:
    script_path = os.path.join(base_dir, script)

    result = subprocess.run([sys.executable, script_path])

    if result.returncode != 0:
        print(f"[ERROR] {script} failed. Stopping execution.")
        sys.exit(result.returncode)
