import os 
import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from collections import Counter
from tqdm import tqdm

def read_csv(csv_path):
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eof_val = row["Hf /kJmol-1"].strip()

            if not eof_val:
                print(f"[WARNING] Missing EOF for CID {row['CID']}")
                continue

            mol = {
                "smiles": row["SMILES"],
                "eof": float(eof_val),
                "cid": row["CID"]
            }
            data.append(mol)

    return data


def calc_density(xyz_path, data):
    a = -0.028604
    b =  1.662404
    c = -0.484438

    # --- SMILES → MW ---
    smiles = data.get("smiles")
    mol_smiles = Chem.MolFromSmiles(smiles)
    if mol_smiles is None:
        print(f"[ERROR] Invalid SMILES for {data.get('cid')}")
        return None

    mol_smiles = Chem.AddHs(mol_smiles)
    mw = Descriptors.MolWt(mol_smiles)

    # --- XYZ → Volume ---
    mol_xyz = Chem.MolFromXYZFile(xyz_path)
    if mol_xyz is None:
        print(f"[ERROR] Failed to read {xyz_path}")
        return None

    try:
        Chem.SanitizeMol(mol_xyz)
    except Exception as e:
        print(f"[ERROR] Failed to sanitise {xyz_path}: {e}")
        return None

    volume = AllChem.ComputeMolVolume(mol_xyz)

    # --- Density ---
    density = mw / volume
    density = (a * density**2) + (b * density) + c

    data["density"] = density
    data["mw"] = mw

    return data

def calc_gas_products(data):
    smiles = data.get("smiles")
    if smiles is None:
        print(f"[ERROR] No SMILES for {data.get('cid')}")
        return data
    
    mw = data.get("mw")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[ERROR] Invalid SMILES for {data.get('cid')}")
        return data

    mol = Chem.AddHs(mol)
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())

    C = counts.get("C", 0)
    H = counts.get("H", 0)
    N = counts.get("N", 0)
    O = counts.get("O", 0)

    """
    Original KJ paper gives the product formation as:
    Ca Hb Nc Od --> c/2 N + b/2 H20 + (d/2 - b/4) CO2 + (a - d/2 + b/4) C
    """

    O_required_for_H2O = H / 2

    if O >= O_required_for_H2O:
        H2O = H / 2
        O_remaining = O - H2O
        CO2 = O_remaining / 2
        C_solid = C - CO2
        H2 = 0
    else:
        H2O = O
        CO2 = 0
        C_solid = C
        H_excess = H - (2 * H2O)
        H2 = H_excess / 2

    products = {
        "N2": N / 2,
        "H2O": H2O,
        "CO2": CO2,
        "C": C_solid,
        "H2": H2
    }

    gas_prods = ["CO2", "N2", "H2O", "H2"]
    gas_prods_masses = {"CO2": 44, "N2": 28, "H2O": 18, "H2": 2}
    gas_eofs = {"CO2": -393.477, "N2": 0, "H2O": -241.808, "H2": 0}

    total_gas = sum(products.get(species, 0) for species in gas_prods)
    if total_gas == 0:
        print(f"[ERROR] No gas products for {data.get('cid')}")
        return data

    eod = sum(products.get(species, 0) * gas_eofs[species] for species in gas_prods)

    gas_mols_per_g = total_gas / mw

    average_mass = (
        sum(products.get(species, 0) * gas_prods_masses[species] for species in gas_prods)
        / total_gas
    )

    """
    q = Chemical energy of detonation reaction (kcal/g) 
    Defined in KJ paper as:
        Q = -[dHf (detonation products) - dFf (explosive)] / formula weight
          = ([eof (kj/mol) - enthalpy of formation (kj/mol)] / 4.184) / mw
    """
    q = ((-(eod - data.get("eof")) / 4.184) / mw) * 1000

    data["Q"] = q
    data["M"] = average_mass
    data["N"] = gas_mols_per_g
    data["eod"] = eod
    data["products"] = products
    data["mw"] = mw

    return data


def calc_phi(data):
    """
    Phi = N * sqrt(M) * sqrt(Q)
    Where:
        - N = Moles of gaseous product per gram of explosive
        - M = Average molecular mass of gaseous product (g/mol)
        - Q = Chemical energy of detonation reaction (kcal/g)
    """
    N = data.get("N")
    M = data.get("M")
    Q = data.get("Q")

    phi = N * np.sqrt(M) * np.sqrt(Q)

    data["phi"] = phi

    return data

def det_v(data):
    """
    D (km/s) = 1.01 * sqrt(phi) * (1 + (1.3 * Rho))
    Where:
        - Rho = Initial loading density
    """
    
    phi = data.get("phi")
    rho = data.get("density")

    d = 1.01 * np.sqrt(phi) * (1 + (1.3 * rho))

    data["d"] = d

    return data

def det_p(data):
    """
    P (Gpa) = 1.558 * phi * Rho ** 2
    P (kbar) = 15.58 * phi * Rho ** 2
    Where:
        - Rho = Initial loading density
    """

    phi = data.get("phi")
    rho = data.get("density")

    p = 1.558 * phi * (rho ** 2)

    data["p"] = p

    return data
    

def write_to_csv(csv_path, results):
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames if reader.fieldnames else []

    new_fields = [
        "density / gcm-3",
        "det_velocity / kms-1",
        "det_pressure / Gpa",
        # "N",
        # "M",
        "Q",
        # "phi",
        # "eod",
        # "CO2",
        # "N2",
        # "H2O",
        # "C",
        # "mw",
    ]
    for field in new_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    results_map = {mol["cid"]: mol for mol in results}

    for row in rows:
        cid = row["CID"]
        if cid in results_map:
            mol = results_map[cid]

            row["density / gcm-3"] = mol.get("density")
            row["det_velocity / kms-1"] = mol.get("d")
            row["det_pressure / Gpa"] = mol.get("p")

            ###########
            # row["N"] = mol.get("N")
            # row["M"] = mol.get("M")
            row["Q"] = mol.get("Q")
            # row["phi"] = mol.get("phi")
            # row["eod"] = mol.get("eod")
            
            # products = mol.get("products", {})
            # row["CO2"] = products.get("CO2")
            # row["N2"] = products.get("N2")
            # row["H2O"] = products.get("H2O")
            # row["C"] = products.get("C")
            # row["mw"] = products.get("mw")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(script_dir, "..", "large_data.csv")
    xyz_dir = os.path.join(script_dir, "..", "OPTIMISED_STRUCTURES", "LARGE_DATASET")

    # csv_path = os.path.join(script_dir, "..", "30_bench.csv")
    # xyz_dir = os.path.join(script_dir, "..", "OPTIMISED_STRUCTURES", "30_MOL")

    # csv_path = os.path.join(script_dir, "..", "bak_30_bench.csv")
    # xyz_dir = os.path.join(script_dir, "..", "OPTIMISED_STRUCTURES", "DET_V_P_TEST")

    data = read_csv(csv_path)

    results = []

    for mol in tqdm(data, desc="Processing molecules"):
        cid = mol["cid"]
        xyz_path = os.path.join(xyz_dir, f"{cid}", f"{cid}.xyz")

        mol = calc_density(xyz_path, mol)
        if mol is None:
            continue

        mol = calc_gas_products(mol)
        mol = calc_phi(mol)
        mol = det_v(mol)
        mol = det_p(mol)

        results.append(mol)
        
    write_to_csv(csv_path, results)
