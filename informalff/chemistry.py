import os                # To navigate the file system
import copy              # To copy objects
import warnings          # To throw warnings instead of raising errors
import numpy as np       # To do basic scientific computing
import pandas as pd      # To manage tables and databases
from pathlib import Path # To locate files in the file system
import scipy.constants as cts # Universal constants
# To be able to construct rotation matrices
from scipy.spatial.transform import Rotation as R

# bohr = 1 / (cts.physical_constants["Bohr radius"][0] * 1e10)

# ------------------------------------------------------- #
#              Setting up the Periodic Table              #
# ------------------------------------------------------- #
# National Center for Biotechnology Information. "Periodic Table of Elements"
# PubChem, https://pubchem.ncbi.nlm.nih.gov/periodic-table/.
# Accessed 20 February, 2024.
# ------------------------------------------------------- #
here = Path(globals().get("__file__", "./_")).absolute().parent
pte_file = os.path.join(here, "data", "PubChemElements_all.csv")
periodic_data = pd.read_csv(pte_file)
PERIODIC_TABLE = periodic_data.set_index("Symbol")
all_symbols = set(PERIODIC_TABLE.index.to_list())

# ------------------------------------------------------- #
#                     The Atom Class                      #
# ------------------------------------------------------- #

class Atom(object):
    """ Class to represent an Atom

    This class is used to represent an Atom, it
    handles its symbol and its coordinates.

    Attributes
    ----------
    element : str
        The symbol of the element of the current atom.
    coords : ndarray
        NumPy array with the X, Y and Z coordinates as `float`
    flag : bool
        A flag to denote if the atom has been selected or not
    """

    def __init__(self,
                 element:str = "H",
                 x : int = 0.0,
                 y : int = 0.0,
                 z : int = 0.0,
                 charge : float = 0.0,
                 flag : bool = False):
        """ Atom constructor method

        This is the method to construct the Atom object

        Parameters
        ----------
        element : str
            The symbol of the element of the current atom.
        x : float
            The atom's X coordinate
        y : float
            The atom's Y coordinate
        z : float
            The atom's Z coordinate
        charge : float
            The atom's charge
        flag : bool
            A flag to denote if the atom has been selected or not
        """
        if element in all_symbols:
            self.element = element
        else:
            raise TypeError((f"Atom.__init__() The symbol {element} does not "
                "correspond to any element in the Periodic Table."))

        self.coords = np.array([x,y,z])
        self.charge = charge
        self.flag = flag

    def __repr__(self) -> str:
        """ Atom representation method

        This method builds a string with the information
        of the Atom object. Said string will be displayed
        whenever someone prints this object.

        Returns
        -------
            text : str
                The atom's element symbol, its coordinates
                and its flag
        """
        text = (f" {self.element} {self.coords[0]:16.8f} "
                f"{self.coords[1]:16.8f} {self.coords[2]:16.8f}"
                f" q(+/-) {self.charge:16.8f} "
                f" [{'*' if self.flag else ' '}]")
        return text

    def set_coordinates(self,
                        x : float,
                        y : float,
                        z : float):
        """ Method to update the Atom's coordinates
        
        Parameters
        ----------
        x : float
            The atom's X coordinate
        y : float
            The atom's Y coordinate
        z : float
            The atom's Z coordinate
        """
        self.coords = np.array([x, y, z])

    def get_coordinates(self) -> np.ndarray:
        """ Method to get the Atom's coordinates
    
        Returns
        -------
        coords : ndarray
            NumPy array with the X, Y and Z coordinates as `float`
        """
        return self.coords

    def move_atom(self,
                  vx : float,
                  vy : float,
                  vz : float):
        """ Method to move the Atom's coordinates
        
        Parameters
        ----------
        vx : float
            The motion vector's X coordinate
        vy : float
            The motion vector's Y coordinate
        vz : float
            The motion vector's Z coordinate
        """
        self.coords += np.array([vx, vy, vz])

# ------------------------------------------------------- #
#                   The Molecule Class                    #
# ------------------------------------------------------- #

class Molecule(object):
    """ Class to represent a Molecule

    This class is used to represent a Molecule, it
    handles its coordinates and other properties.

    Note
    ----
    This class can ready any XYZ file and load it as
    a single molecule. This, however, doesn't mean that
    the loaded coordinates are from ONE single molecule.

    Attributes
    ----------
    name : str
        A name for the molecule (can be anything you choose)
    atoms : list of Atom
        A `list` with all the Atom objects of the molecule
    mol_weight : float
        The molecular weight
    charge : float
        The molecular charge
    """

    def __init__(self, name : str):
        """ Molecule constructor method

        This is the method to construct the Molecule object
        
        Parameters
        ----------
        name : str
            A name for the molecule (can be anything you choose)
        """
        self.name = name
        self.atoms = []
        self.mol_weight = 0.0
        self.charge = 0.0

    def __repr__(self) -> str:
        """ Method to represent a molecule

        This method builds a string with the information
        of the Molecule object. Said string will be displayed
        whenever someone prints this object.

        Returns
        -------
            text : str
                The atom's element symbol, its coordinates
                and its flag
        """
        # Show the number of atoms in the molecule
        temp = f"Atoms[{len(self.atoms)}] for {self.name}\n"

        # Add the information of every atom to the final string
        for i, a in enumerate(self.atoms):
            temp += f"{i:>4} {str(a)}\n"

        return temp

    def add_atoms(self, *atoms : Atom):
        """ Method to add atoms to the molecule

        Adds the specified atom(s) to the Molecule object. It
        checks whether the object is empty, and if the elements
        in the list are actually instances of Atom.

        Raises
        ------
        TypeError
            If an empty list is added to the molecule object.
            If any object in the added list is NOT an instance of Atom.

        Parameters
        ----------
        *atoms
            A `list` with all the Atom objects to be added to the
            Molecule object.
        """
        # Check if the provided list is empty
        if len(atoms) == 0:
            raise TypeError("Molecule.add_atoms() The added object is empty.")

        # If the provided list has only one element
        elif len(atoms) == 1:
            # Check if the provided element is a list
            if isinstance(atoms[0], list):
                atoms = atoms[0]

        # Iterate over all the provided atoms
        for a in atoms:
            # Check that the object is actually an Atom instance
            if not isinstance(a, Atom):
                raise TypeError(("Molecule.add_atoms() The added object is "
                                "not an instance of Atom."))
        
        # Iterate over all the provided atoms
        for a in atoms:
            # Add atoms to the molecule
            self.atoms.append(a)

        # Compute the molecular weight of the molecule
        self.get_mol_weight()
    
    def assign_charges(self, *charges : float):
        """ Method to assign charges to each atom in the molecule

        Raises
        ------
        ValueError
            If the number of charges is not the same as the number of
            atoms in the molecule.

        Parameters
        ----------
        *charges
            A `list` of floats with all the charges for the atoms.
        """
        # Sanity check
        if len(charges) != self.get_num_atoms():
            raise ValueError("Molecule.assign_charges() The number of charges"
                             " is not the same as the number of atoms in the "
                             "molecule!")
        
        # Assign the respective charges
        for c, a in enumerate(self.atoms):
            a.charge = charges[c]

    def get_mol_weight(self):
        """ Method to get the Molecule's mass

        Will compute the molecule's mass using the periodic table
        fetched at the beginning.

        Returns
        -------
        bool
            True if the molecular mass has been computed.
        """
        self.mol_weight = 0.0
        for a in self.atoms:
            symbol = a.element
            self.mol_weight += PERIODIC_TABLE.loc[symbol, "AtomicMass"]

        return True

    def get_coords(self) -> list:
        """ Method to get the molecule's coordinates

        Returns
        -------
        todos : list of list
            A list with the atoms represented by lists with
            the symbol and X, Y, Z coordinates.
        """
        todos = []

        for a in self.atoms:
            x, y, z = a.get_coordinates()
            todos.append([a.element, x, y, z, a.charge])

        return todos

    def get_num_atoms(self) -> int:
        """ Method to get the number of atoms in the molecule

        Returns
        -------
        int
            Number of atoms in the molecule.
        """
        return len(self.atoms)

    def move_molecule(self, direction : np.ndarray):
        """ Method to move the molecule

        Moves each atom of the molecule in a given direction.

        Parameters
        ----------
        directions : ndarray
            A NumPy array with the X, Y, Z coordinates of the motion
            vector, to be added to each atom to move the molecule.
        """
        # Iterate over all atoms in the molecule ...
        for a in self.atoms:
            # Extract the atomic coordinates
            position = a.get_coordinates()
            # Compute the new coordinates
            new_coords = position + direction
            # Set the new coordinates
            a.set_coordinates(  x=new_coords[0],
                                y=new_coords[1],
                                z=new_coords[2])
    
    def move_selected_atoms(self, direction : np.ndarray):
        """ Method to move some atoms of the molecule

        Moves some atom of the molecule in a given direction.

        Parameters
        ----------
        directions : ndarray
            A NumPy array with the X, Y, Z coordinates of the motion
            vector, to be added to each atom to move the molecule.
        """
        # Iterate over all atoms in the molecule ...
        for a in self.atoms:
            if a.flag:
                # Extract the atomic coordinates
                position = a.get_coordinates()
                # Compute the new coordinates
                new_coords = position + direction
                # Set the new coordinates
                a.set_coordinates(  x=new_coords[0],
                                    y=new_coords[1],
                                    z=new_coords[2])

    def rotate_molecule_over_center(self,
                                    euler_angles : np.ndarray,
                                    center : str = "geom"):
        """ Method to rotate the molecule around its center

        Rotates the molecule according to its Euler angles

        Parameters
        ----------
        euler_angles : ndarray
            A NumPy array with the X, Y, Z Euler angles (in degrees)
            for the rotation of the molecule. Keep in mind that the
            molecule has to be taken to the origin to make these
            rotations.
        center : str
            Specify which center of the molecule to use:
            - com  : center of mass
            - atom : atom closest to the center of mass
            - geom : geometric center (coordinates)
        """
        if center == "com":
            mol_center = self.get_center_of_mass()
        elif center == "atom":
            mol_center = self.get_center_atom()[2]
        else:
            mol_center = self.get_center()

        self.move_molecule(-1 * mol_center)

        # Create rotation operator
        r = R.from_euler('xyz', euler_angles.tolist(), degrees=True)

        # Iterate over all atoms in the molecule ...
        for a in self.atoms:
            # Extract the atomic coordinates
            position = a.get_coordinates()
            # Compute the new coordinates
            new_coords = r.as_matrix() @ position
            # Set the new coordinates
            a.set_coordinates(  x=new_coords[0],
                                y=new_coords[1],
                                z=new_coords[2])
        
        self.move_molecule(mol_center)
    
    def rotate_selected_atoms_over_center(self,
                                          euler_angles : np.ndarray,
                                          center : str = "geom"):
        """ Method to rotate some atoms over their center

        Rotates some atoms according to its Euler angles

        Parameters
        ----------
        euler_angles : ndarray
            A NumPy array with the X, Y, Z Euler angles (in degrees)
            for the rotation of the selected atoms. Keep in mind that
            the selected atoms have to be taken to the origin to make
            these rotations.
        center : str
            Specify which center of the molecule to use:
            - com  : center of mass
            - atom : atom closest to the center of mass
            - geom : geometric center (coordinates)
        """
        temp_mol = Molecule("__temporary_molecule__")
        for a in self.atoms:
            if a.flag:
                temp_mol.add_atoms(a)

        if center == "com":
            mol_center = temp_mol.get_center_of_mass()
        elif center == "atom":
            mol_center = temp_mol.get_center_atom()[2]
        else:
            mol_center = temp_mol.get_center()

        self.move_molecule(-1 * mol_center)

        # Create rotation operator
        r = R.from_euler('xyz', euler_angles.tolist(), degrees=True)

        # Iterate over all atoms in the molecule ...
        for a in self.atoms:
            # Extract the atomic coordinates
            position = a.get_coordinates()
            # Compute the new coordinates
            new_coords = r.as_matrix() @ position
            # Set the new coordinates
            a.set_coordinates(  x=new_coords[0],
                                y=new_coords[1],
                                z=new_coords[2])
        
        self.move_molecule(mol_center)
    
    def rotate_molecule_over_atom(self,
                                  euler_angles : np.ndarray,
                                  atom : int):
        """ Method to rotate the molecule around an atom

        Rotates the molecule according to its Euler angles over an atom

        Parameters
        ----------
        euler_angles : ndarray
            A NumPy array with the X, Y, Z Euler angles (in degrees)
            for the rotation of the molecule. Keep in mind that the
            molecule has to be taken to the origin to make these
            rotations.
        atom : int
            The number of atom to be used as rotation point.
        """
        rotation_point = np.array(self.get_coords()[atom][1:4])

        self.move_molecule(-1 * rotation_point)

        # Create rotation operator
        r = R.from_euler('xyz', euler_angles.tolist(), degrees=True)

        # Iterate over all atoms in the molecule ...
        for a in self.atoms:
            # Extract the atomic coordinates
            position = a.get_coordinates()
            # Compute the new coordinates
            new_coords = r.as_matrix() @ position
            # Set the new coordinates
            a.set_coordinates(  x=new_coords[0],
                                y=new_coords[1],
                                z=new_coords[2])
        
        self.move_molecule(rotation_point)
    
    def rotate_selected_atoms_over_atom(self,
                                        euler_angles : np.ndarray,
                                        atom : int):
        """ Method to rotate some atoms around an atom

        Rotates the selected atoms according to the Euler angles
        over an atom

        Raises
        ------
        ValueError
            If one of the selected atoms for rotation is selected as
            pivot for rotation.

        Parameters
        ----------
        euler_angles : ndarray
            A NumPy array with the X, Y, Z Euler angles (in degrees)
            for the rotation of the selected atoms. Keep in mind that
            the selected atoms have to be taken to the origin to make
            these rotations.
        atom : int
            The number of atom to be used as rotation point.
        """
        # Sanity check
        for i, a in enumerate(self.atoms):
            if i == atom and a.flag:
                raise ValueError("Molecule.rotate_selected_atoms_over_atom()"
                                 " The selected atom should not be one of"
                                 " the rotated atoms!")

        rotation_point = np.array(self.get_coords()[atom][1:4])

        self.move_molecule(-1 * rotation_point)

        # Create rotation operator
        r = R.from_euler('xyz', euler_angles.tolist(), degrees=True)

        # Iterate over all atoms in the molecule ...
        for a in self.atoms:
            if a.flag:
                # Extract the atomic coordinates
                position = a.get_coordinates()
                # Compute the new coordinates
                new_coords = r.as_matrix() @ position
                # Set the new coordinates
                a.set_coordinates(  x=new_coords[0],
                                    y=new_coords[1],
                                    z=new_coords[2])
        
        self.move_molecule(rotation_point)
    
    def rotate_molecule_over_bond(self,
                                  atom1 : int,
                                  atom2 : int,
                                  angle : float):
        """ Method to rotate the molecule around a bond

        Rotates the molecule over a specific bond

        Parameters
        ----------
        atom1 : int
            The atom working as the source of the rotation vector.
        atom2 : int
            The atom working as the end of the rotation vector.
        angle : float
            The angle of rotation (in degrees).
        """
        # Find the location of the base of the rotation vector
        base = np.array(self.get_coords()[atom1][1:4])

        # Move the molecule there
        self.move_molecule(-1 * base)

        # Get coordinates for the tip and tail of the arrow
        tail_point = np.array(self.get_coords()[atom1][1:4])
        tip_point = np.array(self.get_coords()[atom2][1:4])

        # Create the rotation vector (orthogonal to rotation)
        rotation_vector = tip_point - tail_point
        rotation_vector /= np.linalg.norm(rotation_vector)

        # Create rotation operator
        r = R.from_rotvec(angle * rotation_vector, degrees=True)

        # Iterate over all atoms in the molecule ...
        for a in self.atoms:
            # Extract the atomic coordinates
            position = a.get_coordinates()
            # Compute the new coordinates
            new_coords = r.as_matrix() @ position
            # Set the new coordinates
            a.set_coordinates(  x=new_coords[0],
                                y=new_coords[1],
                                z=new_coords[2])
        
        
        # Move the molecule back to its place
        self.move_molecule(base)
    
    def rotate_selected_atoms_over_bond(self,
                                        atom1 : int,
                                        atom2 : int,
                                        angle : float):
        """ Method to rotate selected atoms over a bond

        Rotates selected atoms over a specific bond

        Raises
        ------
        ValueError
            If one of the selected atoms for rotation is selected as
            start or end for rotation.

        Parameters
        ----------
        atom1 : int
            The atom working as the source of the rotation vector.
        atom2 : int
            The atom working as the end of the rotation vector.
        angle : float
            The angle of rotation (in degrees).
        """
        # Sanity check
        for i, a in enumerate(self.atoms):
            if ((i == atom1) or (i == atom2)) and a.flag:
                raise ValueError("Molecule.rotate_selected_atoms_over_bond()"
                                 " The selected atom should not be one of"
                                 " the rotated atoms!")
        
        # Find the location of the base of the rotation vector
        base = np.array(self.get_coords()[atom1][1:4])

        # Move the molecule there
        self.move_molecule(-1 * base)

        # Get coordinates for the tip and tail of the arrow
        tail_point = np.array(self.get_coords()[atom1][1:4])
        tip_point = np.array(self.get_coords()[atom2][1:4])

        # Create the rotation vector (orthogonal to rotation)
        rotation_vector = tip_point - tail_point
        rotation_vector /= np.linalg.norm(rotation_vector)

        # Create rotation operator
        r = R.from_rotvec(angle * rotation_vector, degrees=True)

        # Iterate over all atoms in the molecule ...
        for a in self.atoms:
            if a.flag:
                # Extract the atomic coordinates
                position = a.get_coordinates()
                # Compute the new coordinates
                new_coords = r.as_matrix() @ position
                # Set the new coordinates
                a.set_coordinates(  x=new_coords[0],
                                    y=new_coords[1],
                                    z=new_coords[2])
        
        # Move the molecule back to its place
        self.move_molecule(base)

    def get_distance(self, a1 : int, a2 : int) -> float:
        """ Method to get interatomic distance

        Method to get the distance between two atoms

        Parameters
        ----------
        a1 : int
            An integer representing an atom in the molecule.
        a2 : int
            An integer representing an atom in the molecule.

        Returns
        -------
        float
            The value of the distance between both atoms
        """
        # Get coordinates of atom 1
        v1 = self.atoms[a1].get_coordinates()
        # Get coordinates of atom 2
        v2 = self.atoms[a2].get_coordinates()

        return np.linalg.norm(v2 - v1)

    def get_angle(self, a1 : int, a2 : int, a3 : int) -> float:
        """ Method to get interatomic angle

        Method to get the angle between three atoms

        Parameters
        ----------
        a1 : int
            An integer representing an edge atom in the molecule.
        a2 : int
            An integer representing a pivot atom in the molecule.
        a3 : int
            An integer representing an edge atom in the molecule.

        Returns
        -------
        angle : float
            The value of the angle between all 3 atoms
        """
        # Get coordinates
        v1 = self.atoms[a1].get_coordinates()
        v2 = self.atoms[a2].get_coordinates()
        v3 = self.atoms[a3].get_coordinates()

        # Obtaining the direction vectors
        d1 = v1 - v2
        d2 = v3 - v2

        # Having everything as unit vectors
        d1 /= np.linalg.norm(d1)
        d2 /= np.linalg.norm(d2)

        # Compute the angle
        angle = np.arccos( np.dot(d1,d2) )

        # In degrees
        angle *= 180/np.pi

        return angle

    def get_dihedral(self,
                     a1 : int,
                     a2 : int,
                     a3 : int,
                     a4 : int) -> float:
        """ Method to get interatomic angle

        Method to get the angle between three atoms

        Parameters
        ----------
        a1 : int
            An integer representing an edge atom in the molecule.
        a2 : int
            An integer representing an atom in the molecule over
            the axis.
        a3 : int
            An integer representing an atom in the molecule over
            the axis.
        a4 : int
            An integer representing an edge atom in the molecule.

        Returns
        -------
        float
            The value of the angle between all 4 atoms
        """
        # Get coordinates
        v1 = self.atoms[a1].get_coordinates()
        v2 = self.atoms[a2].get_coordinates()
        v3 = self.atoms[a3].get_coordinates()
        v4 = self.atoms[a4].get_coordinates()

        # Obtaining the direction vectors
        d1 = v1 - v2
        d2 = v3 - v2
        d3 = v4 - v3

        # Having everything as unit vectors
        d1 /= np.dot(d1,d1)**0.5
        d2 /= np.dot(d2,d2)**0.5
        d3 /= np.dot(d3,d3)**0.5

        # Obtaining normal vectors
        n1 = np.cross(d1, d2)
        n2 = np.cross(d3, d2)

        # Having everything as unit vectors
        n1 /= np.dot(n1,n1)**0.5
        n2 /= np.dot(n2,n2)**0.5

        pre_num = np.cross(n1, n2)
        num = np.dot(pre_num, pre_num)**0.5
        denom = np.dot(n1, n2)

        # Compute dihedral angle
        dihedral = np.arctan2(num, denom)

        # In degrees
        dihedral *= 180/np.pi

        return dihedral

    def get_center_of_mass(self) -> np.ndarray:
        """ Method to get the molecule's center of mass

        Returns
        -------
        centro : ndarray
            The X, Y, Z coordinates of the center of mass
            of the molecule.
        """
        # Get the coordinates of all the atoms
        atoms = self.get_coords()

        # Get the molecular weight
        M = self.mol_weight

        # Start assuming that the center is at 0, 0, 0
        centro = np.array([0, 0, 0], dtype=np.float64)

        # Iterate over all atoms
        for a in atoms:
            # Weight the coordinates by the mass and add them to the center
            centro += np.array([*a[1:4]]) * PERIODIC_TABLE.loc[a[0], "AtomicMass"]

        # Divide the center by the molecular mass
        centro *= (1 / M)
        return centro

    def get_center_atom(self) -> list:
        """ Method to get the molecule's center atom

        Compute the atom closest to the center of mass.

        Returns
        -------
        distances[0] : list
            The number of the atom, its symbol and its distance
            to the center in a list.
        """
        # All atom distances relative to the COM
        distances = []

        # Compute the COM
        COM = self.get_center_of_mass()

        # Iterate over all atoms
        for i, atom in enumerate(self.atoms):
            dist = atom.coords - COM            # Compute distance for atom
            distances.append([i, atom.element, np.linalg.norm(dist)])

        # Sort all distances
        distances.sort(key=lambda s: s[2])

        return distances[0]

    def get_center(self) -> np.ndarray:
        """ Method to get the geometric center of the molecule

        Compute the center of the molecule solely as an average
        of the coordinates of its atoms.

        Returns
        -------
        centro : ndarray
            A NumPy array with the X, Y, Z coordinates of the
            geometric center of the molecule.
        """
        # Start assuming that the center is at 0, 0, 0
        centro = np.array([0,0,0], dtype=np.float64)

        # Iterate over all atoms
        for atom in self.atoms:

            # Take the coordinates of each atom and add them to the center
            centro += atom.coords

        # Scaling it down by the number of atoms
        centro *= (1.0/len(self.atoms))

        return centro

    def read_xyz(self, file_name : str):
        """ Get molecule info from XYZ file

        Parameters
        ----------
        file_name : str
            Name of the XYZ file with the molecular coordinates.
        """
        # Empty the molecule's atoms
        self.atoms = []

        # Open the XYZ file and read the contents
        with open(file_name, 'r') as f:
            data = f.readlines()

        # Add atom by atom to the Molecule object
        for a in data[2:]:
            temp = a.split()
            temp = [float(c) if i != 0 else c for i, c in enumerate(temp)]
            self.add_atoms(Atom(*temp))

    def save_as_xyz(self):
        """ Save molecule as an XYZ file

        This method does not return anything, nor it requires
        any parameters.
        """
        # Create a template for the XYZ coordinates
        template = " {s} {x:16.8f} {y:16.8f} {z:16.8f}\n"

        content = f"""{len(self.atoms)}
XYZ file of molecule: {self.name} - created by InformalFF
"""

        # Iterate over atoms
        for a in self.get_coords():
            content += template.format(s=a[0], x=a[1], y=a[2], z=a[3])

        with open(f"{self.name}.xyz", "w") as xyz:
            xyz.write(content)
    
    def get_limits(self) -> dict:
        """ Method to get the geometric limits of the molecule

        Compute find the limits of the molecule, considering the
        atomic radii of the atoms.

        Returns
        -------
        lims : dict
            A dictionary with X, Y, Z keys holding lists for each
            coordinate. Each list has:
             - the minumum (low = l)
             - the maximum (high = h)
             - length of the box
        """
        # Change the representation of the coordinates to
        # lists in each dimension
        q_trsp = { q : [] for q in "eXYZ" }

        for a in self.get_coords():
            q_trsp["e"].append(a[0])
            q_trsp["X"].append(a[1])
            q_trsp["Y"].append(a[2])
            q_trsp["Z"].append(a[3])
        
        # Build a new dictionary to hold the limits
        lims = {}

        for q in "XYZ":

            # Compute the minimum and maximum values
            low = min(q_trsp[q])
            high = max(q_trsp[q])

            # Find those values in the list of atoms
            id_l = q_trsp[q].index(low)
            id_h = q_trsp[q].index(high)

            # Get the atoms' atomic radius to pad the molecule
            pad_i = PERIODIC_TABLE.loc[q_trsp['e'][id_l], "AtomicRadius"]
            pad_a = PERIODIC_TABLE.loc[q_trsp['e'][id_h], "AtomicRadius"]

            # From pm to Angstrom
            pad_i /= 100
            pad_a /= 100

            # Compute the limits
            lims[q] = [low - pad_i,
                       high + pad_a,
                       high + pad_a - (low - pad_i)]

        return lims
    
    def charge_in_field(self,
                        x : float,
                        y : float,
                        z : float,
                        charge : float = -1) -> tuple:
        """ Method to get the value and vector of charge

        Putting a probe at a specific point in 3D, compute the value of
        the charge and the charge vector.

        Parameters
        ----------
        x : float
            The probe's X coordinate
        y : float
            The probe's Y coordinate
        z : float
            The probe's Z coordinate
        charge : float
            The probe's charge

        Returns
        -------
        final_charge : float
           The value of the probe at that particular point in space
        final_vector : ndarray
            The vector of the charge "perceived" by the probe
        """
        # Initialize charge and vector
        final_charge = 0.0
        final_vector = np.array([0, 0, 0], dtype = np.float64)

        # Loop over atoms in the molecule
        for a in self.get_coords():
            # Probe coords
            probe_c = np.array([x, y, z], dtype = np.float64)
            # Atom coords
            atom_c = np.array(a[1:4], dtype = np.float64)
            # Create the vector between the probe and the atom
            r_vect = atom_c - probe_c
            # Add probe-atom vector to the final vector
            final_vector += r_vect
            # Compute the distance of the vector
            r = np.linalg.norm(r_vect)
            # Compute the product of charges over distance
            final_charge += charge * a[4] / r
        
        # Normalize charge vector
        final_vector /= np.linalg.norm(final_vector)
        # Re-scale charge vector
        final_vector *= final_charge

        return final_charge, final_vector

    def compute_grid(self,
                     charge : float = -1,
                     mesh : float = 0.5,
                     limits : dict = {},
                     padding : float = 0.2):

        if len(limits) == 0:
            limits = self.get_limits()

        box = {}

        for q in "XYZ":
            temp_low = limits[q][0] - limits[q][2] * padding
            temp_high = limits[q][1] + limits[q][2] * padding
            box[q] = np.linspace(temp_low,
                                 temp_high,
                                 int((temp_high - temp_low) / mesh) + 1)
        
        # gX, gY, gZ = np.meshgrid(box['X'], box['Y'], box['Z'])

        grid, v_field = self.charge_in_field(box['X'][:,None,None], box['Y'][None,:,None], box['Z'][None,None,:], charge)

        return grid, v_field

# ------------------------------------------------------- #
#                    The Cluster Class                    #
# ------------------------------------------------------- #

class Cluster(object):
    """ Class to represent a molecular Cluster

    This class is used to represent a molecular cluster,
    it does so by considering each molecule with its name
    in a dictionary, and adding several methods for the
    analysis of the cluster.

    Note
    ----
    The original idea was to work with results from MD
    simulations of molecules in cubic solvation boxes.
    Therefore, the get_limits and sub_cluster methods are
    oriented towards cubic boxes.

    Attributes
    ----------
    name : str
        A name for the cluster (can be anything you choose)
    molecules : dict of Molecule
        A `dict` with all the Molecule objects of the cluster
    __nmols : int
        The number of molecules in the cluster
    __natoms : int
        The number of atoms in the cluster
    """

    def __init__(self, name='cluster'):
        """ Cluster constructor method

        This is the method to construct the Cluster object

        Parameters
        ----------
        name : str, optional
            A name for the cluster (can be anything you choose)
        """
        self.name = name
        self.molecules = {}
        self.__nmols = 0
        self.__natoms = 0

    def __repr__(self):
        """ Method to represent a Cluster

        This method builds a string with the information
        of the Cluster object. Said string will be displayed
        as a ticket whenever someone prints this object.

        Returns
        -------
            text : str
                A general description of the Cluster, its molecules,
                atoms and dimensions.
        """
        content = "\n=========================\n"
        content += f"    Molecular Cluster\n{self.name:^25}\n"
        content += "-------------------------\n"
        content += f" Total molecules: {self.__nmols}\n"
        content += f" Total atoms: {self.__natoms}\n\n"
        content += "        Molecules\n-------------------------\n"
        if self.__nmols > 0:
            frags = [m[:3] for m in self.molecules.keys()]
            ufrags = set(frags)
            for f in ufrags:
                content += f"{f}:\n"
                content += f"Number of molecules: {frags.count(f)}\n"
                for m in self.molecules.keys():
                    if f in m:
                        temp = m
                        break
                content += ("Atoms per molecule: "
                            f"{self.molecules[temp].get_atoms()}\n")
            content += "\n         Limits\n-------------------------\n"
            lims = self.get_limits()
            content += "     Lower    Upper  Side\n"
            content += (f"X:{lims['x'][0]:8.3f} {lims['x'][1]:8.3f}"
                        f" {lims['x'][2]:5.2f}\n")
            content += (f"Y:{lims['y'][0]:8.3f} {lims['y'][1]:8.3f}"
                        f" {lims['y'][2]:5.2f}\n")
            content += (f"Z:{lims['z'][0]:8.3f} {lims['z'][1]:8.3f}"
                        f" {lims['z'][2]:5.2f}\n")
        content += "\n         Density         \n"
        content += "-------------------------\n"
        content += f"    {self.get_density():>8.4f} g/cm^3      \n"
        content += "=========================\n"
        return content

    def __encoord(self, dims):
        """ Method to encode dimensions

        The method encodes the dimensions of a sub-cluster
        relative to a super-cluster, in a list of hexadecimals.

        Parameters
        ----------
        dims : dict
            A `dict` containing the X, Y, Z lower and upper limits
            of the sub-cluster.

        Returns
        -------
        into_hex : list
            A list with 3 `str` objects (hex numbers).
        """
        lims = self.get_limits()
        # Enconde the position of the sub cluster
        ratio5 = [round(dims[q][0]/lims[q][1] * 1E6) for q in 'xyz']
        into_hex = [hex(r)[2:] for r in ratio5]
        return into_hex

    def __decoord(self):
        """ Method to decode dimensions

        The method decodes the dimensions of the super-cluster
        relative to the sub-cluster.

        Returns
        -------
        coords : dict
            A `dict` with the X, Y, Z upper limits of the super-cluster.
        """

        # If the encoding in the name was done correctly ...
        if self.name.count(".") == 2:
            # Extract the hex-coordinates
            namx, y, z = self.name.split('.')
            x = namx[-5:]
            # Get the limits of the current sub-cluster
            l = self.get_limits()
            # Some structure
            qs = {'x':x, 'y':y, 'z':z}
            # Convert the hex-coordinates into relative coordinates
            dec = {i:int(q, 16) * 1E-6 for i, q in qs.items()}
            # Create the final coordinates of the upper limit of the
            # super-cluster.
            coords = {q:round(l[q][0]/dec[q], 3) for q in 'xyz'}
            return coords

    def add_molecule(self, idm, mol):
        """ Method to add a molecule to the cluster

        Adds the specified molecule to the Cluster object. It
        checks whether the object is actually an instance of
        Molecule.

        Raises
        ------
        TypeError
            If the added object is NOT an instance of Molecule.

        Parameters
        ----------
        idm : str
            The name of the molecule in the cluster.
        mol : Molecule
            A Molecule object to be added to the Cluster object.

        Returns
        -------
        bool
            True if everything works out.
        """

        # Check that the object is actually an Molecule instance
        if not isinstance(mol, Molecule):
            raise TypeError(("Cluster.add_molecule() The added object is not "
                            "an instance of Molecule."))
        
        # Add mols to the molecule
        self.molecules[idm] = mol

        # Increment the number of molecules and atoms
        self.__nmols += 1
        self.__natoms += mol.get_num_atoms()

        return True

    def rem_molecule(self, idm):
        """ Method to remove a molecule from the cluster

        Removes the specified molecule from the Cluster. It
        checks whether the provided id is actually in the Cluster.
        Otherwise, it warns the user.

        Parameters
        ----------
        idm : str
            The name of the molecule in the cluster.

        Returns
        -------
        bool
            True if everything works out.
        """

        # Check that the molecule actually exists
        if idm in self.molecules.keys():
            self.__natoms -= self.molecules[idm].get_atoms()
            self.__nmols -= 1
            # Remove the molecule from the cluster
            del self.molecules[idm]
            return True
        else:
            warnings.warn((f"Cluster: No molecule {idm} in the cluster; "
                           "no molecule deleted."))
            return False

    def read_pdb(self, file_name):
        """ Open a PDB file and read the data

        This function opens a PDB file and reads each atom,
        takes the coordinates and identity, and builds a
        Molecule object with each molecule, adding them all
        to a cluster dictionary.

        Parameters
        ----------
        file_name : str
            Name of the PDB file with the molecular coordinates.

        Returns
        -------
        bool
            True if everything works out.
        """

        # Read the data from the PDB
        with open(file_name, 'r') as pdb:
            data = pdb.readlines()

        # Filter out anything that's not an atom
        data = [l for l in data if ("ATOM" in l) or ("HETATM" in l)]

        # Create an empty list of molecules
        self.name = file_name[:-4]
        self.molecules = {}
        self.__nmols = 0
        self.__natoms = 0

        # Iterate over atoms
        for l in data:

            # The PDB file format is based on columns
            # https://www.biostat.jhsph.edu/~iruczins/teaching/260.655/links/pdbformat.pdf
            # Extract the atomic id
            pdb_ida = int(l[8:12].split()[0])
            # Extract the atom name (AMBER)
            pdb_anam = l[13:17].split()
            # Extract the name of the fragment in the PDB
            pdb_name = l[17:20].split()[0]
            # Extract the number of equal fragment in the PDB
            pdb_idm = int(l[22:27].split()[0])
            # Extract the X, Y, Z coordinates of each atom from the PDB
            pdb_x, pdb_y, pdb_z = [float(q) for q in l[31:55].split()]
            try:
                # Extract the element symbol of each atom from the PDB
                pdb_sym = l[76:].split()[0]
            except IndexError as e:
                pdb_sym = pdb_anam[0]

            # Create a molecule name
            mol_name = f"{pdb_name}_{pdb_idm:04}"

            # Create the atom object
            temp_atom = Atom(pdb_sym, pdb_x, pdb_y, pdb_z)
            temp_atom.amber_name = pdb_anam

            # Create a Molecule object
            if not (mol_name in self.molecules.keys()):
                self.molecules[mol_name] = Molecule(mol_name)

            # Add the atom
            self.molecules[mol_name].add_atoms(temp_atom)

        # Update the number of molecules
        self.__nmols = len(self.molecules)

        # Update the number of atoms
        self.__natoms = len(data)

        return True

    def get_limits(self):
        """ Get the limits of the molecular cluster

        The function finds the maximum and minimum values for
        each coordinate: X, Y, Z. It returns that and the
        distance of the cluster in each axis.

        Returns
        -------
        limits : dict
            The lowest and highest values for the coordinates of
            the atoms in the cluster, in each axis, and the size
            of the cluster in each axis.
        """

        # Initialize the limits
        limits = {'x':[], 'y':[], 'z':[]}
        
        # Iterate over molecules
        for idm, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.get_coords():
                # Lower limits
                if len(limits['x']) == 0:
                    limits['x'].append(a[1])
                    limits['y'].append(a[2])
                    limits['z'].append(a[3])
                else:
                    if a[1] < limits['x'][0]: limits['x'][0] = a[1]
                    if a[2] < limits['y'][0]: limits['y'][0] = a[2]
                    if a[3] < limits['z'][0]: limits['z'][0] = a[3]
                # Upper limits
                if len(limits['x']) == 1:
                    limits['x'].append(a[1])
                    limits['y'].append(a[2])
                    limits['z'].append(a[3])
                else:
                    if a[1] > limits['x'][1]: limits['x'][1] = a[1]
                    if a[2] > limits['y'][1]: limits['y'][1] = a[2]
                    if a[3] > limits['z'][1]: limits['z'][1] = a[3]

        # Compute the span of the molecules over each axis
        limits['x'].append(limits['x'][1] - limits['x'][0])
        limits['y'].append(limits['y'][1] - limits['y'][0])
        limits['z'].append(limits['z'][1] - limits['z'][0])

        return limits

    def sub_cluster(self, dims):
        """ Extract a cluster box from the larger cluster

        All molecules with at least one atom inside the box
        defined by the provided dimensions will be included
        and returned in a new cluster. It considers periodic
        boundary conditions, so if the sub-cluster box is
        slightly out of the cluster's boundaries, it the
        function will replicate the molecules to fill the box
        specified by the provided dimensions.

        Note
        ----
        The new cluster may have different dimensions than
        the ones defined by the provided dimensions. This
        happens because any molecule with at least one atom
        inside the box will be included. That molecule will
        change the final dimensions of the box.

        Parameters
        ----------
        dims : dict
            The lowest and highest values of the coordinates
            in each axis, for the atoms in the sub-cluster.

        Returns
        -------
        sub_c : Cluster
            A Cluster object with all the molecules within the
            provided dimensions.
        """

        # Setting everything correctly before subclustering
        self.fix_box()

        # Get the cluster limits
        lims = self.get_limits()

        # Check that the dimensions of the small box are within the cluster
        outside = [dims[q][1] > lims[q][1] for q in 'xyz']

        # Initialize dimensions of the sub-box
        sub_dims = {}

        # Dimensions of the sub-box
        for i, q in enumerate('xyz'):
            # If this dimension is outside, establish new limits (PBC)
            if outside[i]:
                sub_dims[q] = [[dims[q][0], lims[q][1]],
                        [lims[q][0], lims[q][0] + dims[q][1] - lims[q][1]]]
            # Else, just use the current limits
            else:
                sub_dims[q] = [[dims[q][0], dims[q][1]]]

        # Building the list of molecules within X
        possible_x = [[] for i in range(len(sub_dims['x']))]
        # Building the list of molecules within Y
        possible_y = [[] for j in range(len(sub_dims['y']))]
        # Building the list of molecules within Z
        possible_z = [[] for k in range(len(sub_dims['z']))]

        # Iterate over all molecules ...
        for idm, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.get_coords():
                # Iterate over the new limits

                # Check over all domains
                for i, x in enumerate(sub_dims['x']):
                    # Add molecule if within X
                    if (a[1] > x[0]) and (a[1] < x[1]):
                        possible_x[i].append(idm)

                # Check over all domains
                for j, y in enumerate(sub_dims['y']):
                    # Add molecule if within Y
                    if (a[2] > y[0]) and (a[2] < y[1]):
                        possible_y[j].append(idm)

                # Check over all domains
                for k, z in enumerate(sub_dims['z']):
                    # Add molecule if within Z
                    if (a[3] > z[0]) and (a[3] < z[1]):
                        possible_z[k].append(idm)
        
        # Removing duplicates
        possible_x = [set(i) for i in possible_x]
        possible_y = [set(j) for j in possible_y]
        possible_z = [set(k) for k in possible_z]

        # Intersect all sets to obtain all sub-boxes (octants?)
        mol_sets = {}
        for i, x in enumerate(possible_x):
            for j, y in enumerate(possible_y):
                for k, z in enumerate(possible_z):
                    unsorted_mols = x.intersection(y).intersection(z)
                    mol_sets[f"{i}{j}{k}"] = sorted(list(unsorted_mols))

        # Create a new cluster with the required molecules
        sub_c = Cluster()

        # Iterate over all sets
        for ids, mol_set in mol_sets.items():

            # Prepare to move the molecules
            motion = np.array([ int(ids[0]) * lims['x'][2],
                                int(ids[1]) * lims['y'][2],
                                int(ids[2]) * lims['z'][2],])

            # Iterate over all molecules
            for idm in mol_set:
                # Add the molecule to the new cluster
                # If the object is not deepcopied, then the original will
                # suffer the same fate as the copy
                sub_c.add_molecule(idm, copy.deepcopy(self.molecules[idm]))
                # Move the molecule
                sub_c.molecules[idm].move_molecule(motion)

        # Enconde the position of the sub cluster
        sub_lims = sub_c.get_limits()
        codes = self.__encoord(sub_lims)
        # Name the new cluster
        sub_c.name = f'{self.name}_{codes[0]}.{codes[1]}.{codes[2]}'

        return sub_c


    def is_in_box(self, idm, dims):
        """ Check if a molecule is in a given box region

        It will try to find the molecule in the box specified
        by the provided dimensions.

        Parameters
        ----------
        idm : str
            The name of the molecule to be found
        dims : dict
            The lowest and highest values of the coordinates
            in each axis, for the atoms in the sub-cluster

        Returns
        -------
        bool : 
            Returns true if the molecule is in the given
            coordinates, and false if not.
        """

        # Initialize the atom count
        inside_atoms = 0

        # Get the cluster limits
        lims = self.get_limits()

        # Check that the dimensions of the box are within the cluster
        inside = [dims[q][1] < lims[q][1] for q in 'xyz']

        limx = list(dims['x'])
        limy = list(dims['y'])
        limz = list(dims['z'])

        # If any of the 3 coordinates of the box are out of the cluster ...
        if sum(inside) != 3:
            if inside[0]: limx[1] = lims['x'][1]
            if inside[1]: limy[1] = lims['y'][1]
            if inside[2]: limz[1] = lims['z'][1]

        # Iterate over atoms
        for a in self.molecules[idm].get_coords():

            # Initialize detection control for all 3 dimensions
            is_in = [0,0,0]

            # Get all 3 coordinates
            x, y, z = a[1:]

            # Check if each coordinate of the atom is inside the box
            if (x > limx[0]) and (x < limx[1]): is_in[0] = 1
            if (y > limy[0]) and (y < limy[1]): is_in[1] = 1
            if (z > limz[0]) and (z < limz[1]): is_in[2] = 1

            # If they all are, count the atom in
            if sum(is_in) == 3: inside_atoms += 1

        # If there is at least 1 atom inside the box,
        # consider the molecule inside
        if inside_atoms >= 1:
            return True
        else:
            return False

    def get_density(self):
        """ Calculate the cluster's density

        The method will compute the total mass and volume of
        the cluster and divide them to obtain the density.
        Consider that several units have to be adjusted!

        Returns
        -------
        density : float
            The density of the cluster in g/cm^3
        """

        self.fix_box()

        # Molecules per mol
        avogadro = 6.022E23

        # Get the mass of all molecules in g/mol
        mass = 0
        for mol in self.molecules.keys():
            self.molecules[mol].get_mol_weight()
            mass += self.molecules[mol].mol_weight

        # Gram per mol to kilogram
        mass /= (1000 * avogadro)

        # Get side lengths and compute the volume in Angstrom
        lims = self.get_limits()
        volume = lims['x'][2] * lims['y'][2] * lims['z'][2]

        # Cubic Angstrom to cubic meter
        volume *= (1E-10)**3

        # Density in kg/m^3
        density = mass / volume

        # Density in g/cm^3 (g/mL)
        density /= 1000

        return density


    def fix_box(self):
        """ Re-position the cluster putting an edge on the origin

        The lower limits (in the X, Y, Z axes) of the cluster will
        be re-positioned to the origin. The idea is not to have
        negative coordinates.

        Note
        ----
            This method doesn't require any parameters and will not
            return anything. The change is done to the Cluster object
            itself.
        """

        lims = self.get_limits()
        mins = np.array([lims['x'][0], lims['y'][0], lims['z'][0]])

        # Iterate over molecules
        for idm, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.atoms:
                # Get the atom's current coordinates
                coords = a.get_coordinates()
                # Compute the new coordinates
                new_coords = coords - mins
                # Move the atom ...
                a.set_coordinates(new_coords[0],new_coords[1],new_coords[2])

    def save_as_pdb(self, f_nam="cluster"):

        # Initialize the PDB file content
        content = ("CRYST1    0.000    0.000    0.000  "
                    "90.00  90.00  90.00 P 1           1\n")

        # Create a template for the PDB coordinates
        pdb_template = ("ATOM {num:>6} {s:>2}   {nam} X{molnum:>4}     "
                        "{x:7.3f} {y:7.3f} {z:7.3f}  1.00  0.00          "
                        "{s:>2}\n")

        atom_counter = 0

        # Iterate over molecules
        for idm, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.atoms:
                # Get the atom's current coordinates
                coords = a.get_coordinates()

                # Increment the number of atoms
                atom_counter += 1

                # Check if there's an Amber name
                try:
                    anam = a.amber_name
                except AttributeError as e:
                    anam = a.element

                # Build atom line
                content += pdb_template.format(
                    num=atom_counter,
                    ana=anam,
                    nam=idm[:3],
                    molnum=int(idm[4:]),
                    x=coords[0],
                    y=coords[1],
                    z=coords[2],
                    s=a.element)

        content += "END\n"

        with open(f"{f_nam}.pdb", "w") as xyz:
            xyz.write(content)


    def save_as_xyz(self, f_nam="cluster"):

        # Create a template for the XYZ coordinates
        template = " {s} {x:16.8f} {y:16.8f} {z:16.8f}\n"

        content = f"""{self.__natoms}
XYZ file of cluster: {self.name} - created by InformalFF
"""

        # Iterate over molecules
        for idm, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.get_coords():
                content += template.format(s=a[0], x=a[1], y=a[2], z=a[3])

        with open(f"{f_nam}.xyz", "w") as xyz:
            xyz.write(content)


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("This library was not intended as a standalone program.")