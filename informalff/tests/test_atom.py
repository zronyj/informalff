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
    he1.coordinates = (1.0, 0.0, 0.0)
    he2.coordinates = (0.0, 0.0, 0.0)
    he2.move_atom(np.array([-1.0, 0.0, 0.0]))

    assert sum(he1.coordinates + he2.coordinates) == 0.0

def test_periodic_table():

    c = informalff.Atom(element="C")
    assert c.mass == 12.011
    assert pytest.approx(c.radius, 1e-3) == 0.8996

    assert informalff.PERIODIC_TABLE.loc["C", "AtomicNumber"] == 6

    with np.testing.assert_raises(KeyError):
        _ = informalff.PERIODIC_TABLE.loc["Xx", "AtomicNumber"]

    with np.testing.assert_raises(ValueError):
        _ = informalff.Atom(element="Xx")

def test_get_valence():

    h = informalff.Atom(element="H")
    hv, hlp, hc = h.get_valence()

    assert hv == 1, "H should have 1 valence electron"
    assert hlp == 0, "H should have 0 lone pair"
    assert hc == 2, "H should have a 2-electron shell"

    b = informalff.Atom(element="B")
    bv, blp, bc = b.get_valence()

    assert bv == 3, "B should have 3 valence electrons"
    assert blp == 0, "B should have 0 lone pairs"
    assert bc == 8, "B should have an 8-electron shell"

    c = informalff.Atom(element="C")
    cv, clp, cc = c.get_valence()

    assert cv == 4, "C should have 4 valence electrons"
    assert clp == 0, "C should have 0 lone pairs"
    assert cc == 8, "C should have an 8-electron shell"

    n = informalff.Atom(element="N")
    nv, nlp, nc = n.get_valence()

    assert nv == 3, "N should have 3 valence electrons"
    assert nlp == 1, "N should have 1 lone pairs"
    assert nc == 8, "N should have an 8-electron shell"

    env, enlp, enc = n.get_valence(expand=1)

    assert env == 5, "expanded N should have 5 valence electrons"
    assert enlp == 0, "expanded N should have 0 lone pairs"
    assert enc == 8, "expanded N should have an 8-electron shell"

    s = informalff.Atom(element="S")
    sv, slp, sc = s.get_valence()

    assert sv == 2, "S should have 2 valence electrons"
    assert slp == 2, "S should have 2 lone pairs"
    assert sc == 8, "S should have an 8-electron shell"

    esv, eslp, esc = s.get_valence(expand=1)

    assert esv == 4, "first expansion of S should have 4 valence electrons"
    assert eslp == 1, "first expansion of S should have 1 lone pairs"
    assert esc == 8, "first expansion of S should have an 8-electron shell"

    eesv, eeslp, eesc = s.get_valence(expand=2)

    assert eesv == 6, "second expansion of S should have 6 valence electrons"
    assert eeslp == 0, "second expansion of S should have 0 lone pairs"
    assert eesc == 8, "second expansion of S should have an 8-electron shell"

    with np.testing.assert_raises(ValueError):
        _ = informalff.Atom(element="P").get_valence(expand=2)

    with np.testing.assert_raises(ValueError):
        _ = informalff.Atom(element="B").get_valence(expand=1)