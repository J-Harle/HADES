import subprocess
import sys

scripts = [
    "read_opt_files.py",
    "be_scale.py",
    "omax.py",
    "convolutions.py",
    "integrate.py",
    "plotting.py",
]

for script in scripts:
    # print(f"[INFO] Running {script}...")
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"[ERROR] {script} failed. Stopping execution.")
        sys.exit(result.returncode)

# print("[INFO] All scripts completed successfully.")