
import matplotlib.pyplot as plt # Plotting library
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

def plot_shell(grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_box_aspect((np.ptp(grid['X']), np.ptp(grid['Y']), np.ptp(grid['Z'])))
    ax.scatter(grid['X'], grid['Y'], grid['Z'], 'b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Grid in sphere')
    ax.grid(True)
    plt.show()

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

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("This library was not intended as a standalone program.")