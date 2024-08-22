"""
Unit and regression tests for the atom module of the informalff package.
"""

# Import this package, (test suite), and other packages as needed
import informalff

def test_atom_create():

    he1 = informalff.Atom(element="He")
    he2 = informalff.Atom(element="He")
    he1.set_coordinates(1.0, 0.0, 0.0)
    he2.set_coordinates(0.0, 0.0, 0.0)
    he2.move_atom(-1.0, 0.0, 0.0)

    assert sum(he1.get_coordinates() + he2.get_coordinates()) == 0.0