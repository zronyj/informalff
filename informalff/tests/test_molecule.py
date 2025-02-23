"""
Unit and regression tests for the molecule module of the informalff package.
"""

# Import this package, (test suite), and other packages as needed
import os
import pytest
import numpy as np

import informalff

@pytest.fixture
def methane_molecule():
    c1 = informalff.Atom("C", 0.00000, 0.00000, 0.00000,-1.063938)
    h1 = informalff.Atom("H", 0.00000, 0.00000, 1.08900, 0.272119)
    h2 = informalff.Atom("H", 1.02672, 0.00000,-0.36300, 0.263905)
    h3 = informalff.Atom("H",-0.51336,-0.88916,-0.36300, 0.263954)
    h4 = informalff.Atom("H",-0.51336, 0.88916,-0.36300, 0.263961)

    atoms = [c1, h1, h2, h3, h4]

    mol = informalff.Molecule("Methane")
    mol.add_atoms(atoms)

    return mol, atoms

@pytest.fixture
def methanol_molecule():
    c1 = informalff.Atom("C", 0.01088,-0.00000,-0.08750,-0.698054)
    o1 = informalff.Atom("O", 0.07641, 0.00000, 1.32886,-0.481487)
    h1 = informalff.Atom("H", 1.00034, 0.00000,-0.54552, 0.337572)
    h2 = informalff.Atom("H",-0.52570,-0.88619,-0.42760, 0.287862)
    h3 = informalff.Atom("H",-0.52570, 0.88619,-0.42760, 0.287884)
    h4 = informalff.Atom("H", 0.99049, 0.00000, 1.60937, 0.266224)

    atoms = [c1, o1, h1, h2, h3, h4]

    mol = informalff.Molecule("Methanol")
    mol.add_atoms(atoms)

    return mol, atoms

def test_molecule_create():

    h1 = informalff.Atom(element="H")
    h2 = informalff.Atom(element="H")
    h3 = informalff.Atom(element="H")
    h4 = informalff.Atom(element="H")
    h1.set_coordinates(0.0, 0.0, 0.0)
    h2.set_coordinates(1.0, 0.0, 0.0)
    h3.set_coordinates(0.5, 0.5, 0.5)
    h4.set_coordinates(1.5, 0.5, 0.5)

    mol1 = informalff.Molecule("H2_a")
    mol1.add_atoms(h1, h2)

    mol2 = informalff.Molecule("H2_b")
    mol2.add_atoms(h3, h4)

    assert mol1.mol_weight - mol2.mol_weight == 0.0

def test_molecule_get_distance_matrix():

    h1 = informalff.Atom(element="H")
    h2 = informalff.Atom(element="H")
    h1.set_coordinates(0.0, 0.0, 0.0)
    h2.set_coordinates(1.0, 0.0, 0.0)

    mol1 = informalff.Molecule("H2")
    mol1.add_atoms(h1, h2)

    mat = mol1.get_distance_matrix()

    assert mat[0][1] + mat[1][0] == 2

def test_molecule_get_bonds(methane_molecule):

    mol1, atoms1 = methane_molecule

    bonds = mol1.get_bonds()

    assert len(bonds) == 4

def test_molecule_get_angles(methane_molecule):

    mol1, atoms1 = methane_molecule

    angles = mol1.get_angles()

    assert len(angles) == 6

def test_molecule_get_dihedrals(methanol_molecule):

    mol1, atoms1 = methanol_molecule

    dihedrals = mol1.get_dihedrals()

    assert len(dihedrals) == 3

def test_molecule_geometric_center():

    h1 = informalff.Atom(element="H")
    h2 = informalff.Atom(element="H")
    h1.set_coordinates(0.0, 0.0, 0.0)
    h2.set_coordinates(1.0, 0.0, 0.0)

    mol1 = informalff.Molecule("H2")
    mol1.add_atoms(h1, h2)

    assert sum(mol1.get_center()) == 0.5

def test_molecule_bond_distance(methane_molecule):

    mol1, atoms1 = methane_molecule

    bond = mol1.get_distance(0, 3)

    assert pytest.approx(bond, 1e-3) == 1.089

def test_molecule_bond_distance2(methane_molecule):

    mol1, atoms1 = methane_molecule

    bonds = [0]*4

    for b in range(1,5):
        bonds[b-1] = mol1.get_distance(0, b)
    
    assert pytest.approx(sum(bonds), 1e-3) == pytest.approx(1.089 * 4, 1e-3)

def test_molecule_angle(methane_molecule):

    mol1, atoms1 = methane_molecule

    angle = mol1.get_angle(1,0,2)

    assert pytest.approx(angle, 1e-2) == 109.47

def test_molecule_dihedral(methane_molecule):

    mol1, atoms1 = methane_molecule

    dihedral = mol1.get_dihedral(2,0,1,3)

    assert pytest.approx(dihedral, 1e-2) == 120

def test_molecule_center_atom(methane_molecule):

    mol1, atoms1 = methane_molecule

    assert mol1.get_center_atom()[1] == atoms1[0].element

def test_molecule_center_of_mass(methane_molecule):

    mol1, atoms1 = methane_molecule

    assert sum(mol1.get_center_of_mass()) == 0.0

def test_molecule_get_molecular_volume(methane_molecule):

    mol1, atoms1 = methane_molecule

    vol = mol1.get_molecular_volume(1000, 10)

    assert pytest.approx(vol, 0.03) == 6.5

def test_molecule_remove_atoms(methane_molecule):
    
    mol1, atoms1 = methane_molecule

    mol1.remove_atoms(2,3)

    assert mol1.get_num_atoms() == 3


def test_molecule_move_molecule(methane_molecule):

    mol1, atoms1 = methane_molecule

    mol1.move_molecule(np.array([0.3, 0.5, 0.7]))

    assert sum(mol1.get_center_of_mass()) == 1.5

def test_molecule_selected_atoms(methane_molecule):

    mol1, atoms1 = methane_molecule

    mol1.atoms[1].flag = True

    mol1.move_selected_atoms(np.array([0, 0, 1.0]))

    mol1.atoms[1].flag = False

    second_atom = np.array(mol1.get_coords()[1][1:])

    x = pytest.approx(second_atom[0], 0.1) == 0.0
    y = pytest.approx(second_atom[1], 0.1) == 0.0
    z = pytest.approx(second_atom[2], 1e-3) == 2.089

    assert x and y and z

def test_molecule_rotate_molecule_over_center(methane_molecule):

    mol1, atoms1 = methane_molecule

    mol1.move_molecule(np.array([5.0, 0, 0]))

    mol1.rotate_molecule_over_center(np.array([-90,0,0]))

    second_atom = np.array(mol1.get_coords()[1][1:])

    x = pytest.approx(second_atom[0], 0.1) == 5.0
    y = pytest.approx(second_atom[1], 1e-3) == 1.089
    z = pytest.approx(second_atom[2], 0.1) == 0.0

    assert x and y and z

def test_molecule_rotate_molecule_over_atom(methane_molecule):

    mol1, atoms1 = methane_molecule

    mol1.rotate_molecule_over_atom(np.array([0,90,0]), 1)

    first_atom = np.array(mol1.get_coords()[0][1:])

    x = pytest.approx(first_atom[0], 1e-3) == -1.089
    y = pytest.approx(first_atom[1], 0.1) == 0.0
    z = pytest.approx(first_atom[2], 1e-3) == 1.089

    assert x and y and z

def test_molecule_rotate_selected_atoms_over_atom(methane_molecule):

    mol1, atoms1 = methane_molecule

    for q in range(2,5):
        mol1.atoms[q].flag = True

    c1 = mol1.get_coords()

    mol1.rotate_selected_atoms_over_atom(np.array([0,0,120]), 0)

    c2 = mol1.get_coords()

    x1 = pytest.approx(c1[2][1], 1e-3) == pytest.approx(c2[3][1], 1e-3)
    x2 = pytest.approx(c1[3][1], 1e-3) == pytest.approx(c2[4][1], 1e-3)
    x3 = pytest.approx(c1[4][1], 1e-3) == pytest.approx(c2[2][1], 1e-3)

    assert x1 and x2 and x3

def test_molecule_rotate_selected_atoms_over_atom2(methane_molecule):

    mol1, atoms1 = methane_molecule

    for q in range(1,5):
        mol1.atoms[q].flag = True
    
    with pytest.raises(ValueError) as e_info:
        mol1.rotate_selected_atoms_over_atom(np.array([0,90,0]), 1)

def test_molecule_rotate_molecule_over_bond(methane_molecule):

    mol1, atoms1 = methane_molecule

    mol1.move_molecule(np.array([1.02672, 0, 0.36300]))

    mol1.rotate_molecule_over_bond(0, 1, 60)

    fifth_atom = np.array(mol1.get_coords()[4][1:])

    x = np.round(fifth_atom[0], 3) == 0.0
    y = np.round(fifth_atom[1], 3) == 0.0
    z = np.round(fifth_atom[2], 3) == 0.0

    assert x and y and z

def test_molecule_rotate_selected_atoms_over_bond(methane_molecule):

    mol1, atoms1 = methane_molecule

    for q in range(2,5):
        mol1.atoms[q].flag = True

    mol1.move_molecule(np.array([1.02672, 0, 0.36300]))

    mol1.rotate_selected_atoms_over_bond(0, 1, 60)

    fifth_atom = np.array(mol1.get_coords()[4][1:])

    x = np.round(fifth_atom[0], 3) == 0.0
    y = np.round(fifth_atom[1], 3) == 0.0
    z = np.round(fifth_atom[2], 3) == 0.0

    assert x and y and z

def test_molecule_rotate_selected_atoms_over_bond2(methane_molecule):

    mol1, atoms1 = methane_molecule

    for q in range(1,5):
        mol1.atoms[q].flag = True

    mol1.move_molecule(np.array([1.02672, 0, 0.36300]))

    with pytest.raises(ValueError) as e_info:
        mol1.rotate_selected_atoms_over_bond(0, 1, 60)


def test_molecule_mol_weight(methane_molecule):

    mol1, atoms1 = methane_molecule

    mol_weight = informalff.PERIODIC_TABLE.loc["C", "AtomicMass"]
    mol_weight += 4 * informalff.PERIODIC_TABLE.loc["H", "AtomicMass"]

    assert pytest.approx(mol1.mol_weight, 1e-3) == pytest.approx(mol_weight, 1e-3)

def test_molecule_save_load_xyz(methane_molecule):

    mol1, atoms1 = methane_molecule

    mol1.save_as_xyz()

    mol2 = informalff.Molecule("Methane2")
    mol2.read_xyz("Methane.xyz")

    atomsX = len(mol1.atoms) == len(mol2.atoms)
    weight = mol1.mol_weight == mol2.mol_weight
    center = sum(mol1.get_center() - mol2.get_center()) == 0.0

    os.remove(f"{mol1.name}.xyz")

    assert atomsX and weight and center

def test_molecule_get_limits():

    h1 = informalff.Atom(element="H")
    h2 = informalff.Atom(element="H")
    h1.set_coordinates(0.0, 0.0, 0.0)
    h2.set_coordinates(1.0, 0.0, 0.0)

    mol1 = informalff.Molecule("H2")
    mol1.add_atoms(h1, h2)

    box = mol1.get_limits()

    assert pytest.approx(box['X'][1] - box['X'][0], 0.1) == 2.27

def test_molecule_max_distance_to_center(methane_molecule):

    mol1, atoms1 = methane_molecule

    atom, distance = mol1.max_distance_to_center()

    assert pytest.approx(distance, 1e-2) == 1.72

def test_molecule_charge_in_field(methane_molecule):

    mol1, atoms1 = methane_molecule

    charge, vector = mol1.charge_in_field(0,0,-0.55)

    assert pytest.approx(charge, 1e-2) == 1.0

def test_molecule_compute_charge_box_grid(methane_molecule):

    mol1, atoms1 = methane_molecule

    charge_grid, charge_field = mol1.compute_charge_box_grid(mesh = 0.1)

    total_charge = (charge_grid * (0.1)**3).sum()

    assert np.round(total_charge, 1) == 2.5

def test_molecule_chiral(methane_molecule):

    mol1, atoms1 = methane_molecule

    here = os.path.dirname(os.path.abspath(__file__))

    mol2 = informalff.Molecule("methanol")
    mol2.read_xyz(os.path.join(here, "mols", "methanol.xyz"))

    mol3 = informalff.Molecule("chloro-fluoro-methanol")
    mol3.read_xyz(os.path.join(here, "mols", "ClFMeOH.xyz"))

    assert not mol1.is_chiral()
    assert not mol2.is_chiral()
    assert mol3.is_chiral()