"""
Unit and regression tests for the changer module of the informalff package.
"""

# Import this package, (test suite), and other packages as needed
import pytest
import copy
import os
import numpy as np

import informalff

@pytest.fixture
def eugenol_molecule():
    here = os.path.dirname(os.path.abspath(__file__))
    mol0 = informalff.Molecule("Eugenol")
    mol0.read_xyz(os.path.join(here, "mols", "eugenol.xyz"))
    
    return mol0

def test_stretch(eugenol_molecule):

    mol0 = eugenol_molecule

    mol1 = copy.deepcopy(mol0)

    axis1 = mol0.atoms[3].coordinates
    axis2 = mol0.atoms[8].coordinates

    stretch = informalff.Stretch(
                    mol0,
                    "Eugenol",
                    8
                )
    
    mol2 = stretch.change(mol0, 0.1)

    v1 = axis1 - axis2
    v2 = axis2 - axis1
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    side1 = [3, 11, 12, 13, 20, 21, 22, 23]
    side2 = [i for i in range(24) if i not in side1]

    for i in side1:
        mol1.atoms[i].flag = True
    
    mol1.move_selected_atoms(v1 * 0.1/2)
    mol1.deselect()

    for i in side2:
        mol1.atoms[i].flag = True
    
    mol1.move_selected_atoms(v2 * 0.1/2)
    mol1.deselect()

    compare = []
    for i in range(24):
        compare.append(mol2.atoms[i].coordinates == \
                       mol1.atoms[i].coordinates)

    assert all([ all(c) for c in compare])

def test_bend(eugenol_molecule):

    mol0 = eugenol_molecule

    mol1 = copy.deepcopy(mol0)

    vertex0 = mol0.atoms[20].coordinates
    vertex1 = mol0.atoms[3].coordinates
    vertex2 = mol0.atoms[8].coordinates

    bend = informalff.Bend(
                    mol0,
                    "Eugenol",
                    19
                )
    
    mol2 = bend.change(mol0, 10)

    v1 = vertex0 - vertex1
    v2 = vertex2 - vertex1
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    v3 = np.cross(v1, v2)
    v3 /= np.linalg.norm(v3)
    
    side1 = [12, 13, 20, 22, 23]
    side2 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19]

    for i in side1:
        mol1.atoms[i].flag = True
    
    mol1.rotate_selected_atoms_over_atom_axis(
        v3,
        -10.0/2,
        3
    )
    mol1.deselect()

    for i in side2:
        mol1.atoms[i].flag = True
    
    mol1.rotate_selected_atoms_over_atom_axis(
        v3,
        10.0/2,
        3
    )
    mol1.deselect()

    compare = []
    for i in range(24):
        compare.append(mol2.atoms[i].coordinates == \
                       mol1.atoms[i].coordinates)
    
    assert all([ all(c) for c in compare])