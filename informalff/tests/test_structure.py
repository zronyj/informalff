"""
Unit tests for the structure module of the informalff package.
"""

# Import this package, (test suite), and other packages as needed
import os
import copy
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

def test_structure_read_xyz(methane_molecule, methanol_molecule):

    mol1, atoms1 = methane_molecule
    mol2, atoms2 = methanol_molecule

    mol1.save_as_xyz()
    mol2.save_as_xyz()

    coll = informalff.Collection("Collection")
    coll.add_molecule("methane", copy.deepcopy(mol1))
    coll.add_molecule("methanol", copy.deepcopy(mol2))
    coll.molecules["methane"].move_molecule(np.array([5.0, 5.0, 0.0]))
    coll.save_as_xyz()

    struct1 = informalff.Structure("test")
    struct1.read_xyz("Methane.xyz")
    struct2 = informalff.Structure("test")
    struct2.read_xyz("Methanol.xyz")
    struct3 = informalff.Structure("test")
    struct3.read_xyz("Collection.xyz")

    mol1b = struct1.get_sub_structure()
    mol2b = struct2.get_sub_structure()

    collb = struct3.get_sub_structure()

    assert isinstance(mol1b, informalff.Molecule)
    assert isinstance(mol2b, informalff.Molecule)
    assert isinstance(collb, informalff.Collection)

    atoms1 = len(mol1.atoms) == len(mol1b.atoms)
    weight1 = mol1.get_mol_weight() == mol1b.get_mol_weight()
    center1 = sum(mol1.get_center() - mol1b.get_center()) == 0.0

    atoms2 = len(mol2.atoms) == len(mol2b.atoms)
    weight2 = mol2.get_mol_weight() == mol2b.get_mol_weight()
    center2 = sum(mol2.get_center() - mol2b.get_center()) == 0.0

    assert atoms1 and weight1 and center1
    assert atoms2 and weight2 and center2

    os.remove(f"{mol1.name}.xyz")
    os.remove(f"{mol2.name}.xyz")
    os.remove(f"{coll.name}.xyz")

def test_structure_substructure(water_box):

    coll = water_box
    coll.save_as_xyz(f_nam="Collection")

    struct = informalff.Structure("test")
    struct.read_xyz("Collection.xyz")

    sub = struct.get_sub_structure()
    os.remove("Collection.xyz")

    assert isinstance(sub, informalff.Collection)

    assert len(sub.molecules) == len(coll.molecules)