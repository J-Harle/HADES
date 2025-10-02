import os
import subprocess

def read_xyz():
    xyz_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "OPTIMISED_STRUCTURES"))
    os.chdir(xyz_dir)

    for root, dirs, files in os.walk(xyz_dir):
        xyz_files = [f for f in files if f.endswith(".xyz")]

    # print(subprocess.getoutput("pwd"))
