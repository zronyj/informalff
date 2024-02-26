"""
Unit and regression test for the chemistry module of the informalff package.
"""

# Import package, test suite, and other packages as needed
import os
import sys
import pytest
import informalff
import numpy as np

@pytest.fixture
def methane_molecule():
    c1 = informalff.Atom("C", 0.00000, 0.00000, 0.00000)
    h1 = informalff.Atom("H", 0.00000, 0.00000, 1.08900)
    h2 = informalff.Atom("H", 1.02672, 0.00000,-0.36300)
    h3 = informalff.Atom("H",-0.51336,-0.88916,-0.36300)
    h4 = informalff.Atom("H",-0.51336, 0.88916,-0.36300)

    atoms = [c1, h1, h2, h3, h4]

    mol = informalff.Molecule("Methane")
    mol.add_atoms(atoms)

    return mol, atoms

def test_atom_create():

    he1 = informalff.Atom(element="He")
    he2 = informalff.Atom(element="He")
    he1.set_coordinates(1.0, 0.0, 0.0)
    he2.set_coordinates(0.0, 0.0, 0.0)
    he2.move_atom(-1.0, 0.0, 0.0)

    assert sum(he1.get_coordinates() + he2.get_coordinates()) == 0.0

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

    x = pytest.approx(second_atom[0],1) == 0.0
    y = pytest.approx(second_atom[1],1) == 0.0
    z = pytest.approx(second_atom[2],3) == 2.089

    assert x and y and z

def test_molecule_rotate_molecule_over_center(methane_molecule):

    mol1, atoms1 = methane_molecule

    mol1.move_molecule(np.array([5.0, 0, 0]))

    mol1.rotate_molecule_over_center(np.array([-90,0,0]))

    second_atom = np.array(mol1.get_coords()[1][1:])

    x = pytest.approx(second_atom[0],1) == 5.0
    y = pytest.approx(second_atom[1],3) == 1.089
    z = pytest.approx(second_atom[2],1) == 0.0

    assert x and y and z

def test_molecule_rotate_molecule_over_atom(methane_molecule):

    mol1, atoms1 = methane_molecule

    mol1.rotate_molecule_over_atom(np.array([0,90,0]), 1)

    first_atom = np.array(mol1.get_coords()[0][1:])

    x = pytest.approx(first_atom[0],1) == -1.089
    y = pytest.approx(first_atom[1],3) == 0.0
    z = pytest.approx(first_atom[2],3) == 1.089

    assert x and y and z

def test_molecule_rotate_molecule_over_bond(methane_molecule):

    mol1, atoms1 = methane_molecule

    mol1.move_molecule(np.array([1.02672, 0, 0.36300]))

    mol1.rotate_molecule_over_bond(0, 1, 60)

    fourth_atom = np.array(mol1.get_coords()[3][1:])

    x = pytest.approx(fourth_atom[0],1) == 0.0
    y = pytest.approx(fourth_atom[1],1) == 0.0
    z = pytest.approx(fourth_atom[2],1) == 0.0

    assert x and y and z

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

