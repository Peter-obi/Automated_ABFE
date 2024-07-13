import openmm as mm
import openmm.unit as unit

def add_protein_restraints(system, positions, p1, p2, p3):
    """
    Add distance restraints between protein anchor atoms.

    Args:
        system (openmm.System): The OpenMM system.
        positions (list): List of atom positions.
        p1, p2, p3 (int): Indices of protein anchor atoms.
    """
    force = mm.CustomBondForce("0.5 * k * (r - r0)^2")
    force.addPerBondParameter("k")
    force.addPerBondParameter("r0")

    k_distance = 10.0 * unit.kilocalories_per_mole / unit.angstrom**2
    for i, j in [(p1, p2), (p2, p3), (p1, p3)]:
        distance = unit.norm(positions[i] - positions[j])
        force.addBond(i, j, [k_distance, distance])

    system.addForce(force)

def add_ligand_restraints(system, positions, l1, l2, l3):
    """
    Add distance restraints between ligand anchor atoms.

    Args:
        system (openmm.System): The OpenMM system.
        positions (list): List of atom positions.
        l1, l2, l3 (int): Indices of ligand anchor atoms.
    """
    force = mm.CustomBondForce("0.5 * k * (r - r0)^2")
    force.addPerBondParameter("k")
    force.addPerBondParameter("r0")

    k_distance = 10.0 * unit.kilocalories_per_mole / unit.angstrom**2
    for i, j in [(l1, l2), (l2, l3), (l1, l3)]:
        distance = unit.norm(positions[i] - positions[j])
        force.addBond(i, j, [k_distance, distance])

    system.addForce(force)

def add_tr_restraints(system, positions, p1, l1, dummy_atoms):
    """
    Add translational and rotational restraints.

    Args:
        system (openmm.System): The OpenMM system.
        positions (list): List of atom positions.
        p1, l1 (int): Indices of protein and ligand anchor atoms.
        dummy_atoms (list): Indices of three dummy atoms.
    """
    force = mm.CustomCompoundBondForce(5, "0.5 * k_d * (distance(p1, l1) - r0)^2 + "
                                          "0.5 * k_a * (angle(d1, p1, l1) - a0)^2 + "
                                          "0.5 * k_a * (angle(p1, l1, d2) - a1)^2 + "
                                          "0.5 * k_d * (dihedral(d1, p1, l1, d2) - t0)^2 + "
                                          "0.5 * k_d * (dihedral(p1, l1, d2, d3) - t1)^2")
    force.addPerBondParameter("k_d")
    force.addPerBondParameter("k_a")
    force.addPerBondParameter("r0")
    force.addPerBondParameter("a0")
    force.addPerBondParameter("a1")
    force.addPerBondParameter("t0")
    force.addPerBondParameter("t1")

    k_distance = 10.0 * unit.kilocalories_per_mole / unit.angstrom**2
    k_angle = 10.0 * unit.kilocalories_per_mole / unit.radian**2

    d1, d2, d3 = dummy_atoms
    r0 = unit.norm(positions[p1] - positions[l1])
    a0 = mm.app.internal.angle(positions[d1], positions[p1], positions[l1])
    a1 = mm.app.internal.angle(positions[p1], positions[l1], positions[d2])
    t0 = mm.app.internal.dihedral(positions[d1], positions[p1], positions[l1], positions[d2])
    t1 = mm.app.internal.dihedral(positions[p1], positions[l1], positions[d2], positions[d3])

    force.addBond([p1, l1, d1, d2, d3], [k_distance, k_angle, r0, a0, a1, t0, t1])
    system.addForce(force)

def set_restraint_strengths(system, strength, context=None):
    """
    Set the strength of all restraints in the system.

    Args:
        system (openmm.System): The OpenMM system.
        strength (float): The relative strength of the restraints (0.0 to 1.0).
        context (openmm.Context, optional): The simulation context, if available.
    """
    for force in system.getForces():
        if isinstance(force, mm.CustomBondForce) or isinstance(force, mm.CustomCompoundBondForce):
            for i in range(force.getNumPerBondParameters()):
                if force.getPerBondParameterName(i).startswith('k'):
                    for j in range(force.getNumBonds()):
                        params = list(force.getBondParameters(j))
                        params[i] *= strength
                        force.setBondParameters(j, *params)
            if context:
                force.updateParametersInContext(context)
