"""
Unit and regression tests for the changer module of the informalff package.
"""

# Import this package, (test suite), and other packages as needed
import pytest
import copy
import os
import numpy as np
from copy import deepcopy

import informalff

@pytest.fixture
def azuelene_cell():
    data = {
        'a': 7.7154,
        'b': 5.9019,
        'c': 7.6969,
        'alpha': 90.0,
        'beta': 100.411,
        'gamma': 90.0,
        'volume': 344.712
    }
    return data

@pytest.fixture
def paracetamol_cell():
    data = {
        'a': 11.805,
        'b': 17.164,
        'c': 7.393,
        'alpha': 90.0,
        'beta': 90.0,
        'gamma': 90.0,
        'volume': 1497.98
    }
    return data

def test_crystal_cell_construction(azuelene_cell):

    azl = azuelene_cell

    cell_params = informalff.Cell(
                    azl["a"], azl["b"], azl["c"],
                    azl["alpha"], azl["beta"], azl["gamma"])
    assert np.round(cell_params.volume, 3) == azl["volume"]

    params_as_list = [
        azl["a"], azl["b"], azl["c"],
        azl["alpha"], azl["beta"], azl["gamma"]
    ]
    cell_list = informalff.Cell(params_as_list)
    assert np.round(cell_list.volume, 3) == azl["volume"]

    params_as_vectors = cell_params.vector
    cell_vectors = informalff.Cell(params_as_vectors)
    assert np.round(cell_vectors.volume, 3) == azl["volume"]

    params_as_list_list = params_as_vectors.tolist()
    cell_list_list = informalff.Cell(params_as_list_list)
    assert np.round(cell_list_list.volume, 3) == azl["volume"]

    params_as_list_mat = params_as_vectors.flatten().tolist()
    cell_list_mat = informalff.Cell(params_as_list_mat)
    assert np.round(cell_list_mat.volume, 3) == azl["volume"]

    assert cell_params.params == cell_list.params == cell_vectors.params == \
        cell_list_list.params == cell_list_mat.params
    
def test_crystal_cell_change_params(azuelene_cell):
    
    azl = azuelene_cell

    cell_param = informalff.Cell(
        azl["a"], azl["b"], azl["c"],
        azl["alpha"], azl["beta"], azl["gamma"]
    )
    cell_vec = informalff.Cell(
        azl["a"], azl["b"], azl["c"],
        azl["alpha"], azl["beta"], azl["gamma"]
    )

    assert np.round(cell_param.volume, 3) == azl["volume"]
    assert np.round(cell_vec.volume, 3) == azl["volume"]

    new_params = {
        'a': 8.0,
        'b': 6.0,
        'c': 8.0,
        'alpha': 90.0,
        'beta': 90.0,
        'gamma': 90.0
    }

    new_vector = np.array([
        [8.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 0.0, 8.0]
    ])

    cell_param.params = new_params
    cell_vec.vector = new_vector

    assert cell_vec.params["a"] == new_params["a"]
    assert cell_vec.params["b"] == new_params["b"]
    assert cell_vec.params["c"] == new_params["c"]
    assert cell_vec.params["alpha"] == new_params["alpha"]
    assert cell_vec.params["beta"] == new_params["beta"]
    assert cell_vec.params["gamma"] == new_params["gamma"]
    assert np.round(cell_vec.volume, 3) == 384.0

    assert np.allclose(cell_param.vector, new_vector)
    assert np.round(cell_param.volume, 3) == 384.0

def test_crystal_symmetry_operations():

    so = informalff.SymmetryOperations()
    so.add('1 x,y,z')
    so.add('2 -x,1/2+y,-z')

    sg = so.get_sg_from_symops()

    assert len(so) == 2
    assert sg['identifier'] == 'P2_1'

    so[1] = '2 -x,1/2+y,1/2-z'
    so.add('3 x,-1/2-y,-1/2+z')
    so.add('4 -x,-y,-z')

    sg = so.get_sg_from_symops()

    assert len(so) == 4
    assert sg['identifier'] == 'P2_1/c'

    so[1] = '2 1/2-x,1/2+y,-z'
    so[2] = '3 -x,y,-z'
    so[3] = '4 1/2+x,1/2+y,z'

    sg = so.get_sg_from_symops()

    assert len(so) == 4
    assert sg['identifier'] == 'C2'

def test_crystal_construction_and_symops(paracetamol_cell):

    pctml = paracetamol_cell

    cell_param = informalff.Cell(
        pctml["a"], pctml["b"], pctml["c"],
        pctml["alpha"], pctml["beta"], pctml["gamma"]
    )

    so = informalff.SymmetryOperations()
    so.add('1 x,y,z')
    so.add('2 -x+1/2,-y,z+1/2')
    so.add('3 -x,y+1/2,-z+1/2')
    so.add('4 x+1/2,-y+1/2,-z')
    so.add('5 -x,-y,-z')
    so.add('6 x+1/2,y,-z+1/2')
    so.add('7 x,-y+1/2,z+1/2')
    so.add('8 -x+1/2,y+1/2,z')

    individual_sg = so.get_sg_from_symops()

    here = os.path.dirname(os.path.abspath(__file__))

    cry = informalff.Crystal("paracetamol")
    cry.read_cif(os.path.join(here,
                              "crystals",
                              "paracetamol.cif"))

    cry.cell.params == cell_param.params
    assert pytest.approx(cry.cell.volume, 1e-2) == pctml["volume"]

    assert len(cry.symops) == len(so)

    with pytest.warns(UserWarning):
        crystal_symop = cry.find_symmetry()

    assert crystal_symop['identifier'] == \
        individual_sg['identifier']

def test_crystal_change_cell():

    here = os.path.dirname(os.path.abspath(__file__))
    cry = informalff.Crystal("paracetamol")
    cry.read_cif(os.path.join(here,
                              "crystals",
                              "paracetamol.cif"))

    o_params = cry.cell.params

    side_displacement = 1
    delta_percent = 0.001

    points = 2 * side_displacement
    displaced = {
        "a" : [deepcopy(o_params) for _ in range(points)],
        "b" : [deepcopy(o_params) for _ in range(points)],
        "c" : [deepcopy(o_params) for _ in range(points)],
        "alpha" : [deepcopy(o_params) for _ in range(points)],
        "beta" : [deepcopy(o_params) for _ in range(points)],
        "gamma" : [deepcopy(o_params) for _ in range(points)]
    }

    new_crystals = {}

    for k in displaced.keys():
        for i in range(1, side_displacement + 1):
            displaced[k][i - 1][k] *= (1 - (side_displacement - i + 1) * delta_percent)
            displaced[k][-1 * i][k] *= (1 + (side_displacement - i + 1) * delta_percent)

            m_cell = informalff.Cell(list(displaced[k][i - 1].values()))
            p_cell = informalff.Cell(list(displaced[k][-1 * i].values()))

            new_crystals[f"{k}+{i}"] = deepcopy(cry)
            new_crystals[f"{k}{- 1 * i}"] = deepcopy(cry)
            new_crystals[f"{k}+{i}"].cell = m_cell
            new_crystals[f"{k}{-1 * i}"].cell = p_cell

    for k, v in new_crystals.items():

        assert cry.cell.params != v.cell.params
        
        for pair in zip(cry.rel_coords, v.rel_coords):
            assert not np.allclose(*pair)