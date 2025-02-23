"""
Unit and regression tests for the graph module of the informalff package.
"""

# Import this package, (test suite), and other packages as needed
import pytest
import copy

import informalff

@pytest.fixture
def water_graph():
    atoms = 3
    bonds = [[0, 1],
             [0, 2]]
    return atoms, bonds

@pytest.fixture
def eugenol_graph():
    atoms = 24
    bonds = [
        [0, 1],
        [0, 3],
        [0, 18],
        [1, 2],
        [3, 4],
        [3, 5],
        [5, 6],
        [5, 7],
        [7, 8],
        [7, 16],
        [8, 9],
        [8, 10],
        [8, 11],
        [11, 12],
        [11, 13],
        [13, 14],
        [13, 15],
        [16, 17],
        [16, 18],
        [18, 19],
        [19, 20],
        [20, 21],
        [20, 22],
        [20, 23]]
    
    return atoms, bonds

@pytest.fixture
def camphor_graph():
    atoms = 27
    bonds = [
        [0, 3],
        [1, 3],
        [2, 3],
        [3, 4],
        [4, 5],
        [4, 12],
        [4, 24],
        [5, 6],
        [5, 7],
        [7, 8],
        [7, 9],
        [7, 10],
        [10, 11],
        [10, 12],
        [10, 21],
        [12, 13],
        [12, 17],
        [13, 14],
        [13, 15],
        [13, 16],
        [17, 18],
        [17, 19],
        [17, 20],
        [21, 22],
        [21, 23],
        [21, 24],
        [24, 25],
        [24, 26]]
    
    return atoms, bonds

@pytest.fixture
def anthracene_graph():
    atoms = 24
    bonds = [
        [0, 1],
        [0, 2],
        [0, 22],
        [2, 3],
        [2, 4],
        [4, 5],
        [4, 6],
        [6, 7],
        [6, 21],
        [7, 8],
        [7, 9],
        [9, 18],
        [9, 10],
        [10, 11],
        [10, 12],
        [12, 13],
        [12, 14],
        [14, 15],
        [14, 16],
        [16, 17],
        [16, 18],
        [18, 19],
        [19, 20],
        [19, 21],
        [21, 22],
        [22, 23]]
    
    return atoms, bonds

def increase_atom_index(atoms, bonds):

    new_bonds = copy.deepcopy(bonds)
    for b in range(len(bonds)):
        for i in [0, 1]:
            new_bonds[b][i] += atoms

    return new_bonds

def test_molecular_graph_create(water_graph):

    atoms, bonds = water_graph

    graph_water = informalff.MolecularGraph(atoms, bonds)
    hs = graph_water.get_neighbors(0)
    assert hs == [1, 2]

def test_molecular_graph_get_branch(eugenol_graph):

    atoms, bonds = eugenol_graph
    graph_eugenol = informalff.MolecularGraph(atoms, bonds)
    branch = graph_eugenol.get_branch(7, 8, 3)

    assert branch == [8, 9, 10, 11, 12, 13, 14, 15]

def test_molecular_graph_shortest_path(eugenol_graph):

    atoms, bonds = eugenol_graph
    graph_eugenol = informalff.MolecularGraph(atoms, bonds)
    path = graph_eugenol.shortest_path(8, 1)

    assert path == [8, 7, 5, 3, 0, 1]

def test_molecular_graph_find_rings(camphor_graph):

    atoms, bonds = camphor_graph
    grap_camphor = informalff.MolecularGraph(atoms, bonds)
    path_camphor, rings_camphor = grap_camphor._find_rings()

    assert rings_camphor == [
                                [4, 10, 12, 21, 24],
                                [4, 5, 7, 10, 21, 24],
                                [4, 5, 7, 10, 12]
                            ]

def test_molecular_graph_get_rings(anthracene_graph):

    atoms, bonds = anthracene_graph

    graph_anthracene = informalff.MolecularGraph(atoms, bonds)
    rings_anthracene = graph_anthracene.get_rings()

    assert rings_anthracene == [
                                    [0, 2, 4, 6, 21, 22],
                                    [6, 7, 9, 18, 19, 21],
                                    [9, 10, 12, 14, 16, 18]
                                ]

def test_molecular_graph_follow_bonds(
        anthracene_graph,
        eugenol_graph):
    
    all_atoms = 0
    all_bonds = []
    atoms_a, bonds_a = anthracene_graph
    atoms_e, bonds_e = eugenol_graph

    all_bonds += increase_atom_index(all_atoms, bonds_a)
    all_atoms += atoms_a

    all_bonds += increase_atom_index(all_atoms, bonds_e)
    all_atoms += atoms_e

    structure_graph = informalff.MolecularGraph(all_atoms, all_bonds)

    path = structure_graph._follow_bonds(0, [])

    assert path == list(range(atoms_a))

def test_molecular_graph_connectivity(
        anthracene_graph,
        eugenol_graph,
        water_graph):

    all_atoms = 0
    all_bonds = []
    atoms_a, bonds_a = anthracene_graph
    atoms_e, bonds_e = eugenol_graph
    atoms_w, bonds_w = water_graph

    all_bonds += increase_atom_index(all_atoms, bonds_a)
    all_atoms += atoms_a

    all_bonds += increase_atom_index(all_atoms, bonds_e)
    all_atoms += atoms_e

    all_bonds += increase_atom_index(all_atoms, bonds_a)
    all_atoms += atoms_a

    for i in range(3):
        all_bonds += increase_atom_index(all_atoms, bonds_w)
        all_atoms += atoms_w
    
    structure_graph = informalff.MolecularGraph(all_atoms, all_bonds)

    connectivity = structure_graph.get_connectivity()

    assert len(connectivity) == 6