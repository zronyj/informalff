"""
Unit and regression tests for the chemical bond module of
the informalff package.
"""

# Import this package, (test suite), and other packages as needed
import os
import pytest
import numpy as np
from copy import deepcopy

from informalff import Molecule, Structure
from informalff.chemical_bond import ChemicalBond

here = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def eugenol_molecule():
    mol = Molecule("Eugenol")
    mol.read_xyz(os.path.join(here, "mols", "eugenol.xyz"))
    return mol

@pytest.fixture
def picric_acid_molecule():
    mol = Molecule("Picric Acid")
    mol.read_xyz(os.path.join(here, "mols", "picric.xyz"))
    return mol

@pytest.fixture
def benzoquinone_molecule():
    mol = Molecule("Benzoquinone")
    mol.read_xyz(os.path.join(here, "mols", "benzoquinone.xyz"))
    return mol

@pytest.fixture
def dnbs_molecule():
    mol = Molecule("DNBS")
    mol.read_xyz(os.path.join(here, "mols", "3,5-dinitrobenzenesulfonic.xyz"))
    return mol

@pytest.fixture
def diborane_molecule():
    mol = Molecule("Diborane")
    mol.read_xyz(os.path.join(here, "mols", "diborane.xyz"))
    return mol

@pytest.fixture
def hclo_molecule():
    mol = Molecule("HClO")
    mol.read_xyz(os.path.join(here, "mols", "hclo.xyz"))
    return mol

@pytest.fixture
def hclo2_molecule():
    mol = Molecule("HClO2")
    mol.read_xyz(os.path.join(here, "mols", "hclo2.xyz"))
    return mol

@pytest.fixture
def hclo3_molecule():
    mol = Molecule("HClO3")
    mol.read_xyz(os.path.join(here, "mols", "hclo3.xyz"))
    return mol

@pytest.fixture
def hclo4_molecule():
    mol = Molecule("HClO4")
    mol.read_xyz(os.path.join(here, "mols", "hclo4.xyz"))
    return mol

@pytest.fixture
def water_molecule():
    mol = Molecule("Water")
    mol.read_xyz(os.path.join(here, "mols", "h2o.xyz"))
    return mol

@pytest.fixture
def ammonium_ion():
    mol = Molecule("Ammonium")
    mol.read_xyz(os.path.join(here, "mols", "ammonium.xyz"))
    return mol

@pytest.fixture
def hydronium_ion():
    mol = Molecule("Hydronium")
    mol.read_xyz(os.path.join(here, "mols", "hydronium.xyz"))
    return mol

@pytest.fixture
def sulfate_ion():
    mol = Molecule("Sulfate")
    mol.read_xyz(os.path.join(here, "mols", "sulfate.xyz"))
    return mol

@pytest.fixture
def aminophenolate_ion():
    mol = Molecule("Aminophenolate")
    mol.read_xyz(os.path.join(here, "mols", "4-aminophenolate.xyz"))
    return mol

def test_chemical_bond_regular_molecule(eugenol_molecule):

    expected_bond_list = [
        [0, 1],
        [0, 4],
        [0, 10],
        [1, 5],
        [1, 6],
        [2, 6],
        [2, 7],
        [2, 8],
        [3, 8],
        [3, 11],
        [3, 20],
        [3, 21],
        [4, 8],
        [4, 9],
        [5, 15],
        [6, 16],
        [10, 14],
        [12, 13],
        [12, 20],
        [12, 22],
        [14, 17],
        [14, 18],
        [14, 19],
        [20, 23]
    ]

    expected_bond_orders = [
        1.5,   # [0, 1]
        1.5,   # [0, 4]
        1,     # [0, 10]
        1,     # [1, 5]
        1.5,   # [1, 6]
        1.5,   # [2, 6]
        1,     # [2, 7]
        1.5,   # [2, 8]
        1,     # [3, 8]
        1,     # [3, 11]
        1,     # [3, 20]
        1,     # [3, 21]
        1.5,   # [4, 8]
        1,     # [4, 9]
        1,     # [5, 15]
        1,     # [6, 16]
        1,     # [10, 14]
        1,     # [12, 13]
        2,     # [12, 20]
        1,     # [12, 22]
        1,     # [14, 17]
        1,     # [14, 18]
        1,     # [14, 19]
        1      # [20, 23]
    ]

    cb = ChemicalBond(
        eugenol_molecule.atoms,
        eugenol_molecule.bonds
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_nitro_expansion(picric_acid_molecule):

    expected_bond_list = [
        [0, 1],
        [0, 4],
        [0, 7],
        [1, 9],
        [1, 10],
        [2, 4],
        [2, 12],
        [2, 13],
        [3, 5],
        [3, 8],
        [3, 9],
        [4, 8],
        [5, 16],
        [5, 17],
        [6, 8],
        [9, 11],
        [10, 14],
        [10, 15],
        [11, 18]
    ]

    expected_bond_orders = [
        1.5,  # [0, 1]
        1.5,  # [0, 4]
        1,    # [0, 7]
        1.5,  # [1, 9]
        1,    # [1, 10]
        1,    # [2, 4]
        2,    # [2, 12]
        2,    # [2, 13]
        1,    # [3, 5]
        1.5,  # [3, 8]
        1.5,  # [3, 9]
        1.5,  # [4, 8]
        2,    # [5, 16]
        2,    # [5, 17]
        1,    # [6, 8]
        1,    # [9, 11]
        2,    # [10, 14]
        2,    # [10, 15]
        1     # [11, 18]
    ]

    cb = ChemicalBond(
        picric_acid_molecule.atoms,
        picric_acid_molecule.bonds
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_non_aromatic(benzoquinone_molecule):

    expected_bond_list = [
        [0, 1],
        [0, 4],
        [0, 7],
        [1, 2],
        [1, 3],
        [3, 5],
        [3, 11],
        [4, 8],
        [4, 10],
        [6, 8],
        [8, 11],
        [9, 11]
    ]

    expected_bond_orders = [
        2, # [0, 1]
        1, # [0, 4]
        1, # [0, 7]
        1, # [1, 2]
        1, # [1, 3]
        2, # [3, 5]
        1, # [3, 11]
        1, # [4, 8]
        2, # [4, 10]
        1, # [6, 8]
        2, # [8, 11]
        1  # [9, 11]
    ]

    cb = ChemicalBond(
        benzoquinone_molecule.atoms,
        benzoquinone_molecule.bonds
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_sulfate_expansion(dnbs_molecule):

    expected_bond_list = [
        [0, 1],
        [0, 2],
        [0, 11],
        [1, 5],
        [1, 12],
        [2, 3],
        [2, 13],
        [3, 4],
        [3, 14],
        [4, 5],
        [4, 15],
        [5, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [9, 10],
        [11, 16],
        [11, 17],
        [14, 18],
        [14, 19]
    ]

    expected_bond_orders = [
        1.5,  # [0, 1]
        1.5,  # [0, 2]
        1,    # [0, 11]
        1.5,  # [1, 5]
        1,    # [1, 12]
        1.5,  # [2, 3]
        1,    # [2, 13]
        1.5,  # [3, 4]
        1,    # [3, 14]
        1.5,  # [4, 5]
        1,    # [4, 15]
        1,    # [5, 6]
        2,    # [6, 7]
        2,    # [6, 8]
        1,    # [6, 9]
        1,    # [9, 10]
        2,    # [11, 16]
        2,    # [11, 17]
        2,    # [14, 18]
        2     # [14, 19]
    ]

    cb = ChemicalBond(
        dnbs_molecule.atoms,
        dnbs_molecule.bonds
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_half_bonds(diborane_molecule):

    expected_bond_list = [
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [1, 2],
        [1, 3],
        [1, 6],
        [1, 7]
    ]

    expected_bond_orders = [
        1,   # [0, 4]
        1,   # [0, 5]
        0.5, # [0, 6]
        0.5, # [0, 7]
        1,   # [1, 2]
        1,   # [1, 3]
        0.5, # [1, 6]
        0.5  # [1, 7]
    ]

    cb = ChemicalBond(
        diborane_molecule.atoms,
        diborane_molecule.bonds
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_regular_inorganic(hclo_molecule):

    expected_bond_list = [
        [0, 1],
        [1, 2]
    ]

    expected_bond_orders = [
        1,  # [0, 1]
        1   # [1, 2]
    ]

    cb = ChemicalBond(
        hclo_molecule.atoms,
        hclo_molecule.bonds
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_chlorine_expansion1(hclo2_molecule):

    expected_bond_list = [
        [0, 1],
        [0, 3],
        [1, 2]
    ]

    expected_bond_orders = [
        1,  # [0, 1]
        2,  # [0, 3]
        1   # [1, 2]
    ]

    cb = ChemicalBond(
        hclo2_molecule.atoms,
        hclo2_molecule.bonds
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_chlorine_expansion2(hclo3_molecule):

    expected_bond_list = [
        [0, 1],
        [0, 3],
        [0, 4],
        [1, 2]
    ]

    expected_bond_orders = [
        1,  # [0, 1]
        2,  # [0, 3]
        2,  # [0, 4]
        1   # [1, 2]
    ]

    cb = ChemicalBond(
        hclo3_molecule.atoms,
        hclo3_molecule.bonds
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_chlorine_expansion3(hclo4_molecule):

    expected_bond_list = [
        [0, 1],
        [0, 3],
        [0, 4],
        [0, 5],
        [1, 2]
    ]

    expected_bond_orders = [
        1,  # [0, 1]
        2,  # [0, 3]
        2,  # [0, 4]
        2,  # [0, 5]
        1   # [1, 2]
    ]

    cb = ChemicalBond(
        hclo4_molecule.atoms,
        hclo4_molecule.bonds
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_non_bonded_dimer(water_molecule):

    wm1 = water_molecule
    wm2 = deepcopy(wm1)

    h2_interatomic_vector = wm2[2][1] - wm1[1][1]

    move_vector = h2_interatomic_vector / np.linalg.norm(h2_interatomic_vector)
    move_vector *= (np.linalg.norm(h2_interatomic_vector) + 1.0)

    wm2.move_molecule(move_vector)

    dimer = Structure("water_dimer")
    dimer.add_atoms(wm1.atoms)
    dimer.add_atoms(wm2.atoms)

    dimer.distance_matrix()

    expected_bond_list = [
        [0, 1],
        [0, 2],
        [3, 4],
        [3, 5]
    ]

    expected_bond_orders = [
        1,  # [0, 1]
        1,  # [0, 2]
        1,  # [3, 4]
        1   # [3, 5]
    ]

    # The graph should have detected 5 bonds
    assert len(dimer.bonds) == 5

    # But the chemical bond should only detect 4
    cb = ChemicalBond(
        dimer.atoms,
        dimer.bonds
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_ammonium(ammonium_ion):

    expected_bond_list = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4]
    ]

    expected_bond_orders = [
        1,  # [0, 1]
        1,  # [0, 2]
        1,  # [0, 3]
        1   # [0, 4]
    ]

    cb = ChemicalBond(
        ammonium_ion.atoms,
        ammonium_ion.bonds,
        1
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_hydronium(hydronium_ion):

    expected_bond_list = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]

    expected_bond_orders = [
        1,  # [0, 1]
        1,  # [0, 2]
        1   # [0, 3]
    ]

    cb = ChemicalBond(
        hydronium_ion.atoms,
        hydronium_ion.bonds,
        1
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_sulfate(sulfate_ion):

    expected_bond_list = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4]
    ]

    expected_bond_orders = [
        2,  # [0, 1]
        2,  # [0, 2]
        1,  # [0, 3]
        1   # [0, 4]
    ]

    cb = ChemicalBond(
        sulfate_ion.atoms,
        sulfate_ion.bonds,
        -2
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])

def test_chemical_bond_aminophenolate(aminophenolate_ion):

    expected_bond_list = [
        [0, 1],
        [0, 5],
        [0, 8],
        [1, 2],
        [1, 7],
        [2, 3],
        [2,11],
        [3, 4],
        [3,10],
        [4, 5],
        [4, 6],
        [5, 9],
        [7,12],
        [7,13]
    ]

    expected_bond_orders = [
        1.5,  # [0, 1]
        1.5,  # [0, 5]
        1,  # [0, 8]
        1.5,  # [1, 2]
        1,  # [1, 7]
        1.5,  # [2, 3]
        1,  # [2,11]
        1.5,  # [3, 4]
        1,  # [3,10]
        1.5,  # [4, 5]
        1,  # [4, 6]
        1,  # [5, 9]
        1,  # [7,12]
        1   # [7,13]
    ]

    cb = ChemicalBond(
        aminophenolate_ion.atoms,
        aminophenolate_ion.bonds,
        -1
    )

    bond_list, bond_orders = cb.get_bond_types()

    assert np.allclose(bond_list, expected_bond_list)

    for i in range(len(bond_orders)):
        assert np.isclose(bond_orders[i], expected_bond_orders[i])