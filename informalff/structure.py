from warnings import warn

from .atom import Atom
from .bonding import Bonding
from .molecule import Molecule
from .collection import Collection
from .graph import MolecularGraph

# ------------------------------------------------------- #
#                   The Structure Class                   #
# ------------------------------------------------------- #

class Structure:
    """ A class to represent a structure

    It is a list of Atom objects before they are
    converted into Molecule or Collection objects.
    """

    def __init__(self, name='structure'):
        self.name = name
        self.atoms = []
        self.bonds = []
        self.dist_mat = None
    
    def read_xyz(self, file_name : str) -> None:
        """ Get all atoms from an XYZ file

        Parameters
        ----------
        file_name : str
            Name of the XYZ file with the molecular coordinates.
        """
        # Empty the structure's atoms, bonds, and distance matrix
        self.atoms = []
        self.bonds = []
        self.dist_mat = None

        # Open the XYZ file and read the contents
        with open(file_name, 'r') as f:
            data = f.readlines()

        # Add atom by atom to this object, skipping blank lines
        for a in data[2:]:
            if not a.strip():
                continue
            temp = a.split()
            temp = [float(c) if i != 0 else c for i, c in enumerate(temp)]
            self.atoms.append(Atom(*temp))
    
    def add_atoms(self, *atoms : Atom | list) -> None:
        """ Method to add atoms to the structure

        Adds the specified atom(s) to the Structure object. It
        checks whether the object is empty, and if the elements
        in the list are actually instances of Atom.

        Raises
        ------
        TypeError
            If an empty list is added to the structure object.
            If any object in the added list is NOT an instance of Atom.

        Parameters
        ----------
        *atoms : Atom or list
            A list with all the Atom objects to be added to the
            Structure object.
        
        Raises
        ------
        TypeError
            - If the added object is not an instance of Atom.
            - If the added object is empty.
            - If the list of atoms is empty after unpacking.
        """
        # Check if no arguments were provided
        if len(atoms) == 0:
            raise TypeError("Structure.add_atoms() The added object is empty.")

        # If a single list argument was passed
        if len(atoms) == 1 and isinstance(atoms[0], list):
            atoms = atoms[0]

        # Check if the list of atoms is empty after unpacking
        if len(atoms) == 0:
            raise TypeError("Structure.add_atoms() The added object is empty.")

        # Iterate over all provided atoms and validate type
        for a in atoms:
            if not isinstance(a, Atom):
                raise TypeError("Structure.add_atoms() The added object is not "
                                f"an instance of Atom: {type(a)}")
            self.atoms.append(a)

        # Invalidate calculated bonds and distance matrix
        self.bonds = []
        self.dist_mat = None
    
    def distance_matrix(
            self,
            bond_tol : float = 0.2,
            box_tol : float = 0.5) -> None:
        """
        Method to get the distances between pairs of atoms

        The method also creates the bonds of the structure object

        Parameters
        ----------
        bond_tol : float
            The tolerance for the bond lengths. The default value is 0.2 times
            the sum of the covalent radii of the two atoms.
        box_tol : float
            The tolerance for the box dimensions. The default value is 0.5 Å.
        """
        # Initialize the bonding object
        bonding = Bonding(atoms=self.atoms,
                         bond_tolerance=bond_tol,
                         box_tolerance=box_tol)

        # Get the distance matrix
        try:
            self.dist_mat, self.bonds = bonding.distance_matrix()
        except ValueError as e:
            warn(f"Structure.distance_matrix() WARNING! {e}\n"
                 "The distance matrix could not be calculated. Attempting to "
                 "calculate the bonds without the distance matrix.",
                 UserWarning)
            self.dist_mat = None
            # Get the bonds
            self.bonds = bonding.find_bonds()
    
    def get_sub_structure(self, force : bool = False) -> Molecule | Collection:
        """ Method to get the sub-structures of the given structure

        This means, it will return a Molecule or a Collection object.
        If the connectivity list has more than one element,
        then it's a collection. However, if it has one element,
        it may be a molecule, or an atom.

        Parameters
        ----------
        force : bool
            If 'force' is True, then the non-bonded atoms will be ignored
            when constructing the molecules in the collection. (Useful if
            there's non-bonded atoms like ions)

        Returns
        -------
        sub_structure : Molecule | Collection
            An object representing the system being handled.
        """
        if len(self.atoms) == 0:
            raise ValueError("Structure.get_sub_structure() No atoms present in structure!")

        # Check if bonds/distance_matrix need to be calculated
        if self.dist_mat is None and len(self.bonds) == 0:
            self.distance_matrix()

        # Get the connectivity
        graph = MolecularGraph(len(self.atoms), self.bonds, force)
        sub_graphs = graph.get_connectivity()

        # Check if there are any substructures
        if len(sub_graphs) == 0:
            raise ValueError("Structure.get_sub_structure() No substructures "
                             "were found! Check your inputs!")
        # If there's only one substructure, then it's a molecule
        elif len(sub_graphs) == 1:
            sub_structure = Molecule(self.name)
            sub_structure.add_atoms(*self.atoms)
            sub_structure.get_bonds(True)
        # If there's more than one substructure, then it's a collection
        else:
            sub_structure = Collection(self.name)
            for i, sg in enumerate(sub_graphs):
                temp_mol = Molecule(f"mol_{i}")
                for ai in sg:
                    temp_mol.add_atoms(self.atoms[ai])
                temp_mol.get_bonds(True)
                sub_structure.add_molecule(f"mol_{i}", temp_mol)
        
        return sub_structure