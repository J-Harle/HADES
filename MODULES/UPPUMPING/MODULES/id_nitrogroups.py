import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import sys
import json

csv.field_size_limit(sys.maxsize)  # handle large JSON arrays

def read_csv(csv_name):
    csv_path = os.path.join(script_dir, csv_name)
    data = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({
                    "molecule": row["molecule"],
                    "coordinates": json.loads(row["coordinates"]),
                    "eigenvectors": json.loads(row["eigenvectors"]),
                    "frequencies": json.loads(row["frequencies"]),
                    "all_freqs": json.loads(row["all_freqs"]),
                })
            except Exception as e:
                print(f"[WARN] Skipping row due to parsing error: {e}")

    return data
  

def id_nitro_groups(mol):
    coords = mol.get("coordinates")
    confirmed_nitros = []

    for i, atom1 in enumerate(coords):

        if atom1[0] != "N":
            continue

        coords1 = np.array(atom1[1:])
        bonded_indices = []

        for j, atom2 in enumerate(coords):

            if i == j:
                continue

            coords2 = np.array(atom2[1:])
            distance = np.linalg.norm(coords1 - coords2)

            if 1.1 < distance < 1.7:
                bonded_indices.append(j)

        oxygens = [j for j in bonded_indices if coords[j][0] == "O"]
        others  = [j for j in bonded_indices if coords[j][0] != "O"]

        if len(oxygens) == 2 and len(others) == 1:
            confirmed_nitros.append({
                "N": i,
                "O": oxygens,
                "attached": others[0]
            })

        # elif len(oxygens) == 3 and len(others) == 0:
        #     confirmed_nitros.append({
        #         "N": i,
        #         "O": oxygens
        #     })

        elif len(oxygens) == 3 and len(others) == 0:

            bridging_O = None
            terminal_O = []

            for o_idx in oxygens:

                o_coords = np.array(coords[o_idx][1:])
                bonded_to_carbon = False

                for k, atom3 in enumerate(coords):

                    if k == o_idx:
                        continue

                    if atom3[0] != "C":
                        continue

                    c_coords = np.array(atom3[1:])
                    dist_oc = np.linalg.norm(o_coords - c_coords)

                    if 1.2 < dist_oc < 1.7:   # O–C bond window
                        bonded_to_carbon = True
                        break

                if bonded_to_carbon:
                    bridging_O = o_idx
                else:
                    terminal_O.append(o_idx)

            # Only accept if exactly one bridging O found
            if bridging_O is not None and len(terminal_O) == 2:

                confirmed_nitros.append({
                    "N": i,
                    "O": terminal_O,
                    "attached": bridging_O
                })

    mol["nitro_groups"] = confirmed_nitros

    # if mol.get("molecule") == "CH-PETN":
    #     print("\n[DEBUG] Nitro groups for CH-PETN:")
    #     for g in confirmed_nitros:
    #         print("  ", g)

    return mol


def unmass_weight_evecs(mol):
    mass_dict = {"H": 1, "C": 12, "N": 14, "O": 16}

    evecs = mol.get("eigenvectors")
    mol_modes = []

    for mode in evecs:
        new_mode = []
        for atom in mode:
            atom_index = atom[0]
            atom_id = atom[1]
            factor = np.sqrt(mass_dict[atom_id])
            new_xyz = (np.array(atom[2]) * factor).tolist()
            new_mode.append([atom_index, atom_id, new_xyz])

        mol_modes.append(new_mode)

    mol["unmass_modes"] = mol_modes
    return mol



def concat_freqs_evecs(mol):

    freqs = mol.get("all_freqs")
    modes = mol.get("unmass_modes")

    if len(freqs) != len(modes):
        raise ValueError(
            f"Frequency/eigenvector length mismatch in {mol.get('molecule')}"
        )

    mol["freq_evec_pairs"] = list(zip(freqs, modes))
    # if mol.get("molecule") == "CH-PETN":
    #     print("\n[DEBUG] CH-PETN freq/mode count:")
    #     print("  # freqs:", len(freqs))
    #     print("  # modes:", len(modes))
    return mol


def evecs_to_cartesian(all_data, scale=1.0):
    for mol in all_data:
        # if mol.get("molecule") == "CH-PETN":
        #     print("\n[DEBUG] Building cartesian displacements for CH-PETN")

        coords = mol.get("coordinates")
        freq_evec_pairs = mol.get("freq_evec_pairs")

        if coords is None or freq_evec_pairs is None:
            continue

        cartesian_modes = []

        for freq, mode in freq_evec_pairs:

            mode_cartesian = []

            for atom in mode:

                atom_index = atom[0] - 1   # convert 1-based index to 0-based
                atom_symbol = atom[1]
                displacement = np.array(atom[2]) * scale

                base_xyz = np.array(coords[atom_index][1:])
                new_xyz = (base_xyz + displacement).tolist()

                mode_cartesian.append((atom_symbol, new_xyz))

            cartesian_modes.append((freq, mode_cartesian))
            # if mol.get("molecule") == "CH-PETN":
            #     print("  # cartesian modes:", len(cartesian_modes))
                        
            # break

        # print(cartesian_modes)
        mol["cartesian_displacements"] = cartesian_modes

    return all_data


def extract_no2_from_displacements(mol):

    nitro_groups = mol.get("nitro_groups")
    cartesian_modes = mol.get("cartesian_displacements")

    # if mol.get("molecule") == "CH-PETN":
    #     print("\n[DEBUG] Extracting NO2 displacements for CH-PETN")
    #     print("  nitro_groups:", len(nitro_groups))
    #     print("  cartesian_modes:", len(cartesian_modes))

    if nitro_groups is None or cartesian_modes is None:
        return mol

    no2_modes = []

    for freq, mode in cartesian_modes:
        # if mol.get("molecule") == "CH-PETN":
        #     print(f"  Processing frequency: {freq:.2f}")

        mode_no2 = []

        for group in nitro_groups:

            N_idx = group["N"]
            O_indices = group["O"]

            N_atom = mode[N_idx]
            O_atoms = [mode[i] for i in O_indices]

            mode_no2.append({
                "N": N_atom,
                "O": O_atoms
            })

        no2_modes.append((freq, mode_no2))
        # if mol.get("molecule") == "CH-PETN":
        #     print("    NO2 groups found in this mode:", len(mode_no2))

    mol["no2_displacements"] = no2_modes
    return mol


def calc_dihedral_derivatives(mol):

    name = mol.get("molecule", "Unknown")
    no2_modes = mol.get("no2_displacements")
    nitro_groups = mol.get("nitro_groups")
    coords = mol.get("coordinates")

    # if name == "CH-PETN":
    #     print("\n[DEBUG] Entering calc_dihedral_derivatives for CH-PETN")
    #     print("  # no2_modes:", len(no2_modes) if no2_modes else 0)
    #     print("  # nitro_groups:", len(nitro_groups) if nitro_groups else 0)

    if not no2_modes or not nitro_groups:
        return mol

    print(f"\nMolecule: {name}")
    print("=" * (10 + len(name)))

    for (freq, mode_no2) in no2_modes:

        print(f"  Frequency: {freq:.2f} cm⁻¹")
        print("  " + "-" * 30)

        for group_idx, group in enumerate(mode_no2):

            N_index = nitro_groups[group_idx]["N"]
            O_indices = nitro_groups[group_idx]["O"]
            anchor_index = nitro_groups[group_idx].get("attached")

            rj = np.array(group["N"][1])
            rk = np.array(group["O"][0][1])
            rl = np.array(group["O"][1][1])

            if anchor_index is not None:
                ri = np.array(coords[anchor_index][1:])
            else:
                continue

            rij = ri - rj
            rkj = rl = rj
            rkl = rk - rl

            rijxrkj = np.cross(rij, rkj)
            rkjxrkl = np.cross(rkj, rkl)

            n_rkj = np.linalg.norm(rkj)
            n_rijxrkj = np.linalg.norm(rijxrkj)
            n_rkjxrkl = np.linalg.norm(rkjxrkl)

            rijrkj_rkj2 = np.dot(rij, rkj) / n_rkj**2
            rklrkj_rkj2 = np.dot(rkl, rkj) / n_rkj**2

            d_ri =  n_rkj / n_rijxrkj**2 * rijxrkj
            d_rl = -n_rkj / n_rkjxrkl**2 * rkjxrkl
            d_rj = (rijrkj_rkj2 - 1) * d_ri - rklrkj_rkj2 * d_rl
            d_rk = (rklrkj_rkj2 - 1) * d_rl - rijrkj_rkj2 * d_ri            


            # print(f"    NO2 Group {group_idx + 1}")
            # print(f"      Anchor (atom {anchor_index+1:>3}) : {ri}")
            # print(f"      N      (atom {N_index+1:>3}) : {rj}")
            # print(f"      O1     (atom {O_indices[0]+1:>3}) : {rk}")
            # print(f"      O2     (atom {O_indices[1]+1:>3}) : {rl}")
            # print("")

        # break   # remove this if you want all modes printed

    return mol


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))

    data = read_csv("raw.csv")
    for mol in data:
        id_nitro_groups(mol)
        unmass_weight_evecs(mol)
        concat_freqs_evecs(mol)

    evecs_to_cartesian(data)
    for mol in data:
        extract_no2_from_displacements(mol)

    # print("\n[DEBUG] Molecules in dataset:")
    # for mol in data:
    #     print("  ", mol["molecule"])

        # print(f"Molecule: {mol['molecule']}")
        nitros = mol.get("nitro_groups", [])
        # print(f"  # Nitros: {len(nitros)}")

        calc_dihedral_derivatives(mol)
