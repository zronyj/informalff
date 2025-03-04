"""
Unit and regression tests for the collection module of the informalff package.
"""

# Import this package, (test suite), and other packages as needed
import numpy as np
import pytest
import os

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
def water_box():

    coll = informalff.Collection("water box")

    here = os.path.realpath(__file__)
    here = os.path.dirname(here)
    molecule_path = os.path.join(here, "mols", "water.pdb")

    with open(molecule_path, "r") as f:
        data = f.readlines()

    atoms = []
    current_mol = informalff.Molecule("WAT0")
    for l in data:
        temp = l.split()
        temp = [c for c in temp if c != ""]

        if temp[0] == "HETATM":
            mol_name = f"{temp[3]}{temp[4]}"
            if mol_name != current_mol.name:
                current_mol.add_atoms(*atoms)
                coll.add_molecule(current_mol.name, current_mol)
                current_mol = informalff.Molecule(mol_name)
                atoms = []
            
            atoms.append(informalff.Atom(
                temp[10],
                float(temp[5]),
                float(temp[6]),
                float(temp[7]),
                float(temp[9])
                )
            )
        else:
            current_mol.add_atoms(*atoms)
            coll.add_molecule(current_mol.name, current_mol)
            current_mol = informalff.Molecule(mol_name)
            break
    return coll

def test_collection_get_center():

    h1 = informalff.Atom(element="H")
    h2 = informalff.Atom(element="H")
    h3 = informalff.Atom(element="H")
    h4 = informalff.Atom(element="H")
    h1.set_coordinates(0.0, 0.0, 0.0)
    h2.set_coordinates(1.0, 0.0, 0.0)
    h3.set_coordinates(0.0, 2.0, 0.0)
    h4.set_coordinates(1.0, 2.0, 0.0)

    mol1 = informalff.Molecule("H2_a")
    mol1.add_atoms(h1, h2)

    mol2 = informalff.Molecule("H2_b")
    mol2.add_atoms(h3, h4)

    coll = informalff.Collection("hydrogens")
    coll.add_molecule("H2a", mol1)
    coll.add_molecule("H2b", mol2)

    temp = coll.get_center() - np.array([0.5, 1.0, 0.0])

    assert np.linalg.norm(temp) == 0

def test_collection_detect_collisions(methane_molecule):

    mol1, atoms1 = methane_molecule

    h1 = informalff.Atom(element="H")
    h2 = informalff.Atom(element="H")
    h1.set_coordinates(0.5, 0.5, 1.0)
    h2.set_coordinates(0.5, 1.5, 1.0)

    mol2 = informalff.Molecule("H2")
    mol2.add_atoms(h1, h2)

    coll = informalff.Collection("clash")
    coll.add_molecule("H2", mol2)
    coll.add_molecule("CH4", mol1)

    with pytest.warns(UserWarning, match="found between molecules"):
        assert coll.detect_collisions()

def test_get_total_mass(water_box):

    coll = water_box

    mass_waters = len(coll.molecules) * 18.015

    total_mass = coll.get_total_mass()

    prec = 1e-8

    assert pytest.approx(total_mass, prec) == pytest.approx(mass_waters, prec)

def test_collection_get_limits_edges(water_box):

    coll = water_box

    for mol_id, mol in coll.molecules.items():
        for a in mol.atoms:
            print(a)

    limits = coll.get_limits("edges")

    precision = 0.25

    assert pytest.approx(limits["X"][0], precision) == -10.0
    assert pytest.approx(limits["X"][1], precision) ==  10.0
    assert pytest.approx(limits["Y"][0], precision) == -10.0
    assert pytest.approx(limits["Y"][1], precision) ==  10.0
    assert pytest.approx(limits["Z"][0], precision) == -10.0
    assert pytest.approx(limits["Z"][1], precision) ==  10.0

def test_collection_get_limits_factor(water_box):

    coll = water_box

    for mol_id, mol in coll.molecules.items():
        for a in mol.atoms:
            print(a)

    limits = coll.get_limits("factor", factor=2.5)

    precision = 0.2

    assert pytest.approx(limits["X"][0], precision) == -10.0
    assert pytest.approx(limits["X"][1], precision) ==  10.0
    assert pytest.approx(limits["Y"][0], precision) == -10.0
    assert pytest.approx(limits["Y"][1], precision) ==  10.0
    assert pytest.approx(limits["Z"][0], precision) == -10.0
    assert pytest.approx(limits["Z"][1], precision) ==  10.0

def test_collection_get_limits_scan(water_box):

    coll = water_box

    for mol_id, mol in coll.molecules.items():
        for a in mol.atoms:
            print(a)

    limits = coll.get_limits("scan")

    precision = 0.175

    assert pytest.approx(limits["X"][0], precision) == -10.0
    assert pytest.approx(limits["X"][1], precision) ==  10.0
    assert pytest.approx(limits["Y"][0], precision) == -10.0
    assert pytest.approx(limits["Y"][1], precision) ==  10.0
    assert pytest.approx(limits["Z"][0], precision) == -10.0
    assert pytest.approx(limits["Z"][1], precision) ==  10.0