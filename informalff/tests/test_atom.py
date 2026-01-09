"""
Unit and regression tests for the atom module of the informalff package.
"""

# Import this package, (test suite), and other packages as needed
import pytest
import numpy as np
import informalff

def test_atom_create():

    he1 = informalff.Atom(element="He")
    he2 = informalff.Atom(element="He")
    he1.set_coordinates(1.0, 0.0, 0.0)
    he2.set_coordinates(0.0, 0.0, 0.0)
    he2.move_atom(np.array([-1.0, 0.0, 0.0]))

    assert sum(he1.get_coordinates() + he2.get_coordinates()) == 0.0

def test_periodic_table():

    c = informalff.Atom(element="C")
    assert c.mass == 12.011
    assert pytest.approx(c.radius, 1e-3) == 0.8996

    assert informalff.PERIODIC_TABLE.loc["C", "AtomicNumber"] == 6

    with np.testing.assert_raises(KeyError):
        _ = informalff.PERIODIC_TABLE.loc["Xx", "AtomicNumber"]

    with np.testing.assert_raises(ValueError):
        _ = informalff.Atom(element="Xx")