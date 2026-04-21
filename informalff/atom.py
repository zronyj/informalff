
#import matplotlib.pyplot as plt # Plotting library
import numpy as np       # To do basic scientific computing
import os                # To navigate the file system
import pandas as pd      # To manage tables and databases
from pathlib import Path # To locate files in the file system
import scipy.constants as cts # Universal constants

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
BOHR = 1 / (cts.physical_constants["Bohr radius"][0] * 1e10)

def fibonacci_grid_shell(center : np.ndarray,
                         radius : float = 1.0,
                         dots : int = 70):
    """ Method to create a 3D spherical shell around a point in space

    A list of Fibonacci grid shells is used to create a grid of a
    filled sphere to be able to probe properties.

    Note
    ----
    Cylindrical coordinates are used here to construct the
    spherical grid.

    Parameters
    ----------
    center : ndarray
        The X, Y and Z coordinates of the point in space which
        serves as the shell's center
    radius : float
        The radius of the outer shell of the sphere
    dots : int
        The amount of dots in the grid

    Returns
    -------
    grid : dict
        The X, Y and Z coordinates of all dots in that shell
    """
    # Initialize rho and the angles
    phi = np.zeros(dots)
    zeta = np.zeros(dots)
    pr = np.zeros(dots)

    # Create the rho, thetha and phi angles for each point
    for d in range(dots):
        # Center all the dots around 0
        i_centered = 2 * d - (dots - 1)
        # Compute the azimuthal angle
        phi[d] = (2 * np.pi * i_centered) / cts.golden
        # Compute the altitude
        zeta[d] = i_centered / dots
        # Compute rho, or radial distance as a function
        # of the altitude (this should look like a circle)
        pr[d] = np.sqrt(1 - (zeta[d])**2)
        # Adjust for a sphere with a given radius
        pr[d] *= radius

    # Create a dictionary which will hold the coordinates
    # of each grid-point in each dimension
    grid = { q : [0] * dots for q in 'XYZ' }

    # Fill in the coordinates of the grid as X, Y and Z
    for i in range ( 0, dots ) :
        grid['X'][i] = pr[i] * np.sin(phi[i]) + center[0]
        grid['Y'][i] = pr[i] * np.cos(phi[i]) + center[1]
        grid['Z'][i] = radius * zeta[i] + center[2]
    
    return grid

# def plot_shell(grid):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection = '3d')
#     ax.set_box_aspect((np.ptp(grid['X']), np.ptp(grid['Y']), np.ptp(grid['Z'])))
#     ax.scatter(grid['X'], grid['Y'], grid['Z'], 'b')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Grid in sphere')
#     ax.grid(True)
#     plt.show()

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
    charge : float
        The atom's charge
    radius : float
        The atom's radius
    mass : float
        The atom's mass
    bonded_atoms : list
        A list with the ID of the bonded atoms
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
            self._element = element
        else:
            raise ValueError((f"Atom.__init__() The symbol {element} does not "
                "correspond to any element in the Periodic Table."))
        
        self._coords = np.array([x,y,z])
        self.charge = charge
        self.flag = flag

        self._update_properties()

        self.bonded_atoms = []

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
        text = (f" {self._element} {self._coords[0]:16.8f} "
                f"{self._coords[1]:16.8f} {self._coords[2]:16.8f}"
                f" q(+/-) {self.charge:16.8f} "
                f" [{'*' if self.flag else ' '}]")
        return text
    
    def __get_mass(self) -> float:
        """
        Get the mass of an atom.

        Returns
        -------
        mass : float
            Mass of the atom.
        """
        return PERIODIC_TABLE.loc[self._element, "AtomicMass"]
    
    def __get_electron_configuration(self) -> dict:
        """
        Get the electron configuration of an atom.

        Returns
        -------
        electron_configuration : dict
            Electron configuration of the atom.
        """
        # Get the electron configuration of the atom
        ec = PERIODIC_TABLE.loc[self._element, "ElectronConfiguration"]

        # If it's not H or He, remove the lower shells
        if "]" in ec:
            ec = ec.split("]")[1]
        
        # If the electron configuration is not fully defined
        if "(" in ec:
            ec = ec.split("(")[0]
        
        # Split the electron configuration into orbital types
        orb_types = ec.split(" ")

        # Remove any empty strings
        orb_types = [orb for orb in orb_types if orb != ""]

        # Create a dictionary with the orbital types as keys
        # and the number of electrons as values
        electron_configuration = {}
        for orb_type in orb_types:
            electron_configuration[orb_type] = int(orb_type[-1])
            
        return electron_configuration
    
    def __get_electronegativity(self) -> float:
        """
        Get the electronegativity of an atom.

        Parameters
        ----------
        element : str
            The symbol of the element of the atom.

        Returns
        -------
        electronegativity : int
            Valence of the atom.
        """
        # Get the electronegativity of the atom
        en = PERIODIC_TABLE.loc[self._element, "Electronegativity"]

        # Check if the value exists
        if en != "":
            return float(en)
        else:
            raise ValueError(
                    "Atom.__get_electronegativity(): "
                    f"Electronegativity not found for {self._element}")
    
    def __get_atomic_radius(self) -> float:
        """
        Get the atomic radius of an atom.

        Returns
        -------
        atomic_radius : float
            Atomic radius of the atom.
        """
        # Get the atomic radius of the atom
        ar = PERIODIC_TABLE.loc[self._element, "AtomicRadius"]

        # Check if the value exists
        if ar != "":
            return (float(ar) / BOHR) / 100
        else:
            raise ValueError(
                    "Atom.__get_atomic_radius(): "
                    f"Atomic radius not found for {self._element}")
    
    def __get_oxidation_states(self) -> list:
        """
        Get the oxidation states of an atom.

        Returns
        -------
        oxidation_states : list
            Oxidation states of the atom.
        """
        # Get the oxidation states of the atom
        oxs = PERIODIC_TABLE.loc[self._element, "OxidationStates"]

        # Check if the value exists
        if oxs != "":
            return [int(ox) for ox in oxs.split(",")]
        else:
            raise ValueError(
                    "Atom.__get_oxidation_states(): "
                    f"Oxidation states not found for {self._element}")
    
    def _update_properties(self):
        """
        Update the properties of the atom.
        """
        self.radius = self.__get_atomic_radius()
        self.mass = self.__get_mass()
        self.electronegativity = self.__get_electronegativity()
        self.oxidation_states = self.__get_oxidation_states()
    
    @property
    def element(self) -> str:
        """ Method to get the Atom's element symbol

        Returns
        -------
        element : str
            The atom's element symbol
        """
        return self._element
    
    @element.setter
    def element(self, element : str):
        """ Method to update the Atom's element symbol

        Parameters
        ----------
        element : str
            The atom's element symbol
        
        Raises
        ------
        TypeError
            If the element symbol does not correspond to any element
            in the Periodic Table
        """
        if element in all_symbols:
            self._element = element
        else:
            raise TypeError((f"Atom.element.setter() The symbol {element} "
                "does not correspond to any element in the Periodic Table."))
        
        self._update_properties()
    
    @property
    def coordinates(self) -> np.ndarray:
        """ Method to get the Atom's coordinates
    
        Returns
        -------
        coords : ndarray
            NumPy array with the X, Y and Z coordinates as `float`
        """
        return self._coords
    
    @coordinates.setter
    def coordinates(self, *args):
        """ Method to update the Atom's coordinates
        
        Parameters
        ----------
        coords : numpy.ndarray, dictionary, list, tuple or individual entries
                - If it is a NumPy array, it must have the X, Y and Z
                coordinates as `float`
                - If it is a dictionary, it must have keys for X, Y and Z
                coordinates, and all values as `float`
                - If it is a list, it must have the X, Y and Z coordinates
                as `float`
                - If it is a tuple, it must have the X, Y and Z coordinates
                as `float`
                - If it is individual entries, it must be 3 entries representing
                the X, Y and Z coordinates as `float`
        
        Raises
        ------
        TypeError
            If the provided coordinates are not a NumPy array, a dictionary,
            a list, a tuple or 3 entries
        """
        if len(args) == 1:
            if len(args[0]) != 3:
                raise TypeError("Atom.coordinates.setter(): "
                                "Coordinates must be a 3D vector. "
                                f"Received {len(args[0])} entries.")
            else:
                if isinstance(args[0], np.ndarray):
                    coords = args[0]
                elif isinstance(args[0], dict):
                    if "x" not in args[0] or \
                        "y" not in args[0] or \
                        "z" not in args[0]:
                        raise TypeError("Atom.coordinates.setter(): "
                                        "The dictionary should contain keys "
                                        "for x, y and z coordinates.")
                    coords = np.array([args[0]["x"], args[0]["y"], args[0]["z"]])
                elif isinstance(args[0], list):
                    coords = np.array(args[0])
                elif isinstance(args[0], tuple):
                    coords = np.array(args[0])
                else:
                    raise TypeError("Atom.coordinates.setter(): "
                                    "Coordinates must be a NumPy vector, "
                                    "a list or a tuple.")
        elif len(args) == 3:
            if isinstance(args[0], float) and \
                isinstance(args[1], float) and \
                isinstance(args[2], float):
                coords = np.array([args[0], args[1], args[2]])
            else:
                raise TypeError("Atom.coordinates.setter(): "
                                "Coordinates must be 3 float entries.")
        else:
            raise TypeError("Atom.coordinates.setter(): "
                            "Coordinates must be one 3D NumPy vector, "
                            "one 3-item dictionary with 'x', 'y' and 'z' keys, "
                            "one 3-item list, one 3-item tuple, or "
                            "3 float entries. "
                            f"Received {len(args)} entries.")

        self._coords = coords

    def move_atom(self, v_move : np.ndarray) -> None:
        """ Method to move the Atom's coordinates
        
        Parameters
        ----------
        v_move : ndarray
            NumPy array with the X, Y and Z coordinates as `float`
        """
        self._coords += v_move
    
    def sphere_grid(self,
                    r : float = 1,
                    dots : int = 200,
                    delta_r : float = 0.3):
        """ Method to create a 3D spherical grid around an atom

        A list of Fibonacci grid shells is used to create a grid of a
        filled sphere to be able to probe atomic properties.

        Raises
        ------
        ValueError
            If the number of shells that can be created is less,
            than 2, an error will be raised, since this method is
            not intended to create a single shell.

        Parameters
        ----------
        r : float
            The radius of the outer shell of the sphere
        dots : int
            The amount of dots in the grid
        delta_r : float
            The distance between the radius of one shell an the other

        Returns
        -------
        grid : dict
            The X, Y and Z coordinates of all dots in that sphere
        """
        # Compute the range of shells
        shells = range(1, int(np.ceil(r / delta_r)) + 1)
        if len(shells) < 2:
            raise ValueError('Atom.sphere_grid() The number of shells that '
                             'can be generated with this radius and delta is'
                             'not enough for a filled sphere to be made.')
        # Compute the area of each shell
        shell_areas = [4 * np.pi * (delta_r * i)**2 for i in shells]
        # Compute the dot density over shell area
        surf_dens = dots / sum(shell_areas)
        # Compute the number of dots in each shell
        shell_dots = [int(np.ceil(area * surf_dens)) for area in shell_areas]
        # How many dots are there now?
        temp_dots = sum(shell_dots)
        # How off are we?
        delta_dots = int(abs(temp_dots - dots))
        # Rescaling ...
        if delta_dots != 0:
            for d in range(1, delta_dots+1):
                shell_dots[- d] -= 1
        # Create grid
        grid = { 'X':[], 'Y':[], 'Z':[] }
        for s in range(len(shells)):
            temp_grid = fibonacci_grid_shell(self.get_coordinates(),
                                             delta_r * s,
                                             shell_dots[s])
            for q in "XYZ":
                grid[q] = grid[q] + temp_grid[q]

        return grid
    
    def get_bonded_atoms(self) -> list:
        """ Method to get the atom's first partners

        Returns
        -------
        bonded_atoms : list
            A list with the ID of the bonded atoms
        """
        return self.bonded_atoms
    
    def get_valence(self, expand : int = 0) -> int:
        """ Method to compute the valence of an atom

        Parameters
        ----------
        expand : int
            The number of lone pairs that should be added to the valence

        Returns
        -------
        valence : int
            The valence of the atom
        lone_pairs : int
            How many pairs of electrons should not be bonded
        valence_orbital_capacity : int
            The number of electrons that fit in the valence shell
        """
        # Number of electrons in each orbital
        orbital_capacity = {
            "s" : 2,
            "p" : 6,
            "d" : 10,
            "f" : 14,
            "g" : 18
        }

        # Orbitals per shell
        orbital_filling = {
            "1s" : ["1s"],
            "2s" : ["2s", "2p"],
            "3s" : ["3s", "3p"],
            "4s" : ["4s", "3d", "4p"],
            "5s" : ["5s", "4d", "5p"],
            "6s" : ["6s", "4f", "5d", "6p"],
            "7s" : ["7s", "5f", "6d", "7p"],
            "8s" : ["8s", "5g", "6f", "7d", "8p"]
        }

        # Get the electron configuration
        e_config = self.__get_electron_configuration()

        # Calculate how many valence electrons does the atom have
        all_valence_electrons = 0

        # ... and also get the shell
        shell = ""

        for ke, ve in e_config.items():
            all_valence_electrons += ve
            if "s" in ke:
                shell = ke[:-1]

        # Calculate how many electrons I can put in this shell
        valence_orbital_capacity = 0
        
        for orb in orbital_filling[shell]:
            valence_orbital_capacity += orbital_capacity[orb[-1]]
        
        # Calculate the valence
        # (or how many bonds does it take to fill up the shell)
        valence = valence_orbital_capacity - all_valence_electrons

        # If the capacity of the shell is larger than the number of
        # electrons I can get from covalently-bonding all valence electrons,
        # set the valence to the number of valence electrons
        if valence > all_valence_electrons:
            valence = all_valence_electrons

        # If I did as many covalent bonds as active valence electrons,
        # how many lone pairs do I have?
        lone_pairs = (all_valence_electrons - valence) // 2

        # Expand the valence
        if expand > 0:

            # Check if there are enough lone pairs
            if lone_pairs == 0:
                raise ValueError("Atom.get_valence() There are no lone pairs "
                                "to expand the valence with.")
        
            if expand > lone_pairs:
                raise ValueError("Atom.get_valence() There are not enough lone "
                                "pairs to expand the valence with.")
        
            # Expand the valence
            valence += 2 * expand
            lone_pairs -= expand
        
        assert valence + 2 * lone_pairs == all_valence_electrons, (
            "Atom.get_valence() The valence and the number of lone pairs "
            "do not add up to the number of valence electrons."
        )

        return valence, lone_pairs, valence_orbital_capacity

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("This library was not intended as a standalone program.")