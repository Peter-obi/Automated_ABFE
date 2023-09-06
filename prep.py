import MDAnalysis as mda
from MDAnalysis import AtomGroup
import numpy as np

def identify_anchors(structure_file,output_file):
    # Load the protein-ligand complex
    u = mda.Universe(structure_file)

    # Identify L1
    l1_atom = u.select_atoms('resname LIG and name C10')
    
    if not l1_atom:
        print("Error: The atom with resname LIG and name C10 was not found.")
        return

    l1_position = l1_atom.positions[0]
    # Function to filter atoms based on angle criteria
    def filter_atoms_by_angle(ref1_position, ref2_position, candidates, angle_target=90, tolerance=10):
        # Calculate angles for each candidate atom
        angles = [mda.lib.distances.calc_angles(ref1_position, ref2_position, atom.position, box=u.dimensions) for atom in candidates]
        # Filter out atoms that are within tolerance of the target angle
        return [atom for idx, atom in enumerate(candidates) if abs(np.degrees(angles[idx]) - angle_target) <= tolerance]

    # Define a function to facilitate atom selection based on distance
    def select_atoms_around_point(point, radius):
        return u.select_atoms(f'point {radius} {point[0]} {point[1]} {point[2]}')

    # Identify P1
    x, y, z = l1_atom.atoms[0].position
    all_within_8A = u.select_atoms(f'protein and (name CA or name C or name N) and point {x} {y} {z} 8')
    all_within_7A = u.select_atoms(f'protein and (name CA or name C or name N) and point {x} {y} {z} 7')
    shell_7_8A = [atom for atom in all_within_8A if atom not in all_within_7A]
    p1 = min(shell_7_8A, key=lambda atom: np.linalg.norm(atom.position - l1_position))

    # Identify P2
    x, y, z = p1.position
    all_within_12A = u.select_atoms(f'protein and (name CA or name C or name N) and point {x} {y} {z} 12')
    all_within_8A = u.select_atoms(f'protein and (name CA or name C or name N) and point {x} {y} {z} 8')
    shell_8_12A = [atom for atom in all_within_12A if atom not in all_within_8A]
    
    filtered_p2 = filter_atoms_by_angle(l1_position, p1.position, shell_8_12A)
    if not filtered_p2:  # if no atoms are found that match the angle criterion
        print("Warning: No suitable atoms found for P2 based on the angle criteria.")
        return

    p2 = min(filtered_p2, key=lambda atom: abs(90 - np.degrees(mda.lib.distances.calc_angles(l1_position, p1.position, atom.position, box=u.dimensions))))

    # Identify P3 based on both angle and distance criteria
    x, y, z = p2.position
    all_within_12A = u.select_atoms(f'protein and (name CA or name C or name N) and point {x} {y} {z} 12')
    all_within_8A = u.select_atoms(f'protein and (name CA or name C or name N) and point {x} {y} {z} 8')
    shell_8_12A = [atom for atom in all_within_12A if atom not in all_within_8A]

    # First filter by angle
    filtered_p3_angle = filter_atoms_by_angle(p1.position, p2.position, shell_8_12A)
    if not filtered_p3_angle:  # if no atoms are found that match the angle criterion
        print("Warning: No suitable atoms found for P3 based on the angle criteria.")
        return

    # Then filter by distance (this is necessary as the shell might have atoms slightly outside the desired range)
    filtered_p3_distance = [atom for atom in filtered_p3_angle if 8 <= np.linalg.norm(atom.position - p2.position) <= 12]
    if not filtered_p3_distance:
        print("Warning: No atoms found for P3 based on the distance criteria.")
        return

    # Finally, select the P3 atom with an angle closest to 90° from the filtered list
    p3 = min(filtered_p3_distance, key=lambda atom: abs(90 - np.degrees(mda.lib.distances.calc_angles(p1.position, p2.position, atom.position, box=u.dimensions))))

    # Print distances and angles
    p1_l1_distance = np.linalg.norm(p1.position - l1_position)
    print(f'P1-L1 distance: {p1_l1_distance:.2f} Å')

    p1_p2_distance = np.linalg.norm(p1.position - p2.position)
    print(f'P1-P2 distance: {p1_p2_distance:.2f} Å')

    p2_p3_distance = np.linalg.norm(p2.position - p3.position)
    print(f'P2-P3 distance: {p2_p3_distance:.2f} Å')

    angle_p2_p1_l1 = mda.lib.distances.calc_angles(p2.position, p1.position, l1_position, box=u.dimensions)
    print(f'P2-P1-L1 angle: {np.degrees(angle_p2_p1_l1):.2f} degrees')

    angle_p1_p2_p3 = mda.lib.distances.calc_angles(p1.position, p2.position, p3.position, box=u.dimensions)
    print(f'P1-P2-P3 angle: {np.degrees(angle_p1_p2_p3):.2f} degrees')

    with open(output_file, 'w') as file:
        # Write the details to the file
        file.write("Selected Anchor Points:\n")
        file.write(f"P1 details: {p1}\n")
        file.write(f"P2 details: {p2}\n")
        if p3:
            file.write(f"P3 details: {p3}\n")
        
        # Write distances to the file
        file.write(f"P1-L1 distance: {np.linalg.norm(p1.position - l1_position):.2f} Å\n")
        file.write(f"P1-P2 distance: {np.linalg.norm(p1.position - p2.position):.2f} Å\n")
        if p3:
            file.write(f"P2-P3 distance: {np.linalg.norm(p2.position - p3.position):.2f} Å\n")

        # Write angles to the file
        file.write(f"P2-P1-L1 angle: {np.degrees(mda.lib.distances.calc_angles(p2.position, p1.position, l1_position, box=u.dimensions)):.2f} degrees\n")
        if p3:
            file.write(f"P1-P2-P3 angle: {np.degrees(mda.lib.distances.calc_angles(p1.position, p2.position, p3.position, box=u.dimensions)):.2f} degrees\n")

# Example call
identify_anchors('your_pdb.pdb', 'output.txt')

