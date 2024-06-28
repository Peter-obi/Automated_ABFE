import MDAnalysis as mda
import numpy as np
import argparse

def identify_anchors(structure_file, ligand_name, l1_name):
    """
    Identify anchor atoms in a protein-ligand complex.

    Args:
        structure_file (str): Path to the structure file (e.g., PDB).
        ligand_name (str): Residue name of the ligand.
        l1_name (str): Name of the L1 atom.

    Returns:
        tuple: (l1, p1, p2, p3, l2, l3) atom objects, or None if identification fails.
    """
    u = mda.Universe(structure_file)

    l1 = identify_l1(u, ligand_name, l1_name)
    if l1 is None:
        return None

    p1 = identify_p1(u, l1)
    p2 = identify_p2(u, l1, p1)
    if p2 is None:
        return None

    p3 = identify_p3(u, p1, p2)
    if p3 is None:
        return None

    l2, l3 = identify_l2_l3(u, ligand_name, l1, p1)

    return l1, p1, p2, p3, l2, l3

def identify_l1(universe, ligand_name, l1_name):
    """Identify the L1 anchor atom."""
    l1_atom = universe.select_atoms(f'resname {ligand_name} and name {l1_name}')
    if not l1_atom:
        print(f"Error: The atom with resname {ligand_name} and name {l1_name} was not found.")
        return None
    return l1_atom[0]

def identify_p1(universe, l1):
    """Identify the P1 anchor atom."""
    l1_position = l1.position
    all_within_8a = universe.select_atoms(f'protein and (name CA or name C or name N) and point 8 {l1_position[0]} {l1_position[1]} {l1_position[2]}')
    all_within_7a = universe.select_atoms(f'protein and (name CA or name C or name N) and point 7 {l1_position[0]} {l1_position[1]} {l1_position[2]}')
    shell_7_8a = [atom for atom in all_within_8a if atom not in all_within_7a]
    return min(shell_7_8a, key=lambda atom: np.linalg.norm(atom.position - l1_position))

def identify_p2(universe, l1, p1):
    """Identify the P2 anchor atom."""
    all_within_12a = universe.select_atoms(f'protein and (name CA or name C or name N) and point 12 {p1.position[0]} {p1.position[1]} {p1.position[2]}')
    all_within_8a = universe.select_atoms(f'protein and (name CA or name C or name N) and point 8 {p1.position[0]} {p1.position[1]} {p1.position[2]}')
    shell_8_12a = [atom for atom in all_within_12a if atom not in all_within_8a]
    
    filtered_p2 = filter_atoms_by_angle(l1.position, p1.position, shell_8_12a)
    if not filtered_p2:
        print("Warning: No suitable atoms found for P2 based on the angle criteria.")
        return None

    return min(filtered_p2, key=lambda atom: abs(90 - np.degrees(mda.lib.distances.calc_angles(l1.position, p1.position, atom.position, box=universe.dimensions))))

def identify_p3(universe, p1, p2):
    """Identify the P3 anchor atom."""
    all_within_12a = universe.select_atoms(f'protein and (name CA or name C or name N) and point 12 {p2.position[0]} {p2.position[1]} {p2.position[2]}')
    all_within_8a = universe.select_atoms(f'protein and (name CA or name C or name N) and point 8 {p2.position[0]} {p2.position[1]} {p2.position[2]}')
    shell_8_12a = [atom for atom in all_within_12a if atom not in all_within_8a]

    filtered_p3_angle = filter_atoms_by_angle(p1.position, p2.position, shell_8_12a)
    if not filtered_p3_angle:
        print("Warning: No suitable atoms found for P3 based on the angle criteria.")
        return None

    filtered_p3_distance = [atom for atom in filtered_p3_angle if 8 <= np.linalg.norm(atom.position - p2.position) <= 12]
    if not filtered_p3_distance:
        print("Warning: No atoms found for P3 based on the distance criteria.")
        return None

    return min(filtered_p3_distance, key=lambda atom: abs(90 - np.degrees(mda.lib.distances.calc_angles(p1.position, p2.position, atom.position, box=universe.dimensions))))

def identify_l2_l3(universe, ligand_name, l1, p1):
    """Identify the L2 and L3 anchor atoms."""
    ligand_atoms = universe.select_atoms(f'resname {ligand_name} and not name H*')
    
    # Find L2: atom closest to 90 degree angle with L1 and P1
    l2 = min(ligand_atoms, key=lambda atom: abs(90 - np.degrees(mda.lib.distances.calc_angles(p1.position, l1.position, atom.position))))
    
    # Find L3: atom closest to 90 degree angle with L1 and L2, excluding atoms too close to L2
    l2_neighbors = universe.select_atoms(f'around 1.7 index {l2.index}')
    l3_candidates = [atom for atom in ligand_atoms if atom not in l2_neighbors and atom != l1 and atom != l2]
    l3 = min(l3_candidates, key=lambda atom: abs(90 - np.degrees(mda.lib.distances.calc_angles(l1.position, l2.position, atom.position))))
    
    return l2, l3

def filter_atoms_by_angle(ref1_position, ref2_position, candidates, angle_target=90, tolerance=10):
    """Filter atoms based on angle criteria."""
    angles = [mda.lib.distances.calc_angles(ref1_position, ref2_position, atom.position, box=candidates[0].universe.dimensions) for atom in candidates]
    return [atom for idx, atom in enumerate(candidates) if abs(np.degrees(angles[idx]) - angle_target) <= tolerance]

def get_anchor_help():
    return """
    Guidance for choosing anchor atoms (based on BAT.py article):

    L1: Choose an atom in the ligand that is relatively rigid and central.
        It should be within 7-8 Å of a protein backbone atom.

    L2 and L3 will be automatically determined based on geometric criteria:
    - L2 is chosen to form an angle close to 90° with L1 and P1.
    - L3 is chosen to form an angle close to 90° with L1 and L2, while being sufficiently far from L2.

    P1, P2, and P3 are automatically determined based on distance and angle criteria:
    - P1 is a protein backbone atom (CA, C, or N) within 7-8 Å of L1.
    - P2 is chosen to form an angle close to 90° with L1 and P1, at a distance of 8-12 Å from P1.
    - P3 is chosen to form an angle close to 90° with P1 and P2, at a distance of 8-12 Å from P2.

    The selection of these atoms is crucial for the stability and accuracy of the ABFE calculations.
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify anchor atoms for ABFE calculations.")
    parser.add_argument("structure_file", help="Path to the structure file (e.g., PDB)")
    parser.add_argument("ligand_name", help="Residue name of the ligand")
    parser.add_argument("l1_name", help="Name of the L1 atom")
    parser.add_argument("--help_anchors", action="store_true", help="Show help for choosing anchor atoms")

    args = parser.parse_args()

    if args.help_anchors:
        print(get_anchor_help())
    else:
        anchors = identify_anchors(args.structure_file, args.ligand_name, args.l1_name)
        if anchors:
            l1, p1, p2, p3, l2, l3 = anchors
            print(f"Anchors identified:\nL1: {l1}\nL2: {l2}\nL3: {l3}\nP1: {p1}\nP2: {p2}\nP3: {p3}")
        else:
            print("Failed to identify all anchor atoms.")
