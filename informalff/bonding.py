import numpy as np
from warnings import warn
from itertools import product
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import cdist

from .elements import PTE

# ------------------------------------------------------- #
#                The Simple Bonding Class                 #
# ------------------------------------------------------- #

class Bonding:
    """ Class to find the bonds between atoms
    
    This class is used to figure out which atoms are bonded in a given
    structure. It uses the element of each atom, their positions and their
    radii to determine if they are bonded or not.
    
    Attributes
    ----------
    __elements : list
        List of atom elements
    __coords : list
        List of atom coordinates
    __radii : dict
        Dictionary of atom radii
    __bonds : list
        List of bonds
    __tolerance : float
        Tolerance used together with the atomic radii to compute the
        maximum distance for two atoms to be considered bonded.
    __cube_tolerance : float
        Tolerance used to compute the cube size for the grid-based
        neighbor search.
    __multi : bool
        If True, the bond tolerance is multiplicative. If False, the bond
        tolerance is additive.
    __verbose : bool
        Verbose flag
    """
    ADDRESSES = list(product([-1, 0, 1], repeat=3))

    def __init__(self,
                 atoms : list,
                 bond_tolerance : float = 0.3,
                 box_tolerance : float = 0.5,
                 multiplicative : bool = True,
                 verbose : bool = False):
        """
        Bonding constructor method

        Parameters
        ----------
        atoms : list
            List of atoms
        bond_tolerance : float
            Tolerance for the bond distance. If multiplicative is True,
            the bond distance is multiplied by (1 + bond_tolerance). If
            multiplicative is False, the bond distance is increased by
            bond_tolerance. Default is 0.3.
        box_tolerance : float
            Tolerance for the box size. The box size is computed as the
            maximum atomic radius plus the bond tolerance. Default is 0.5.
        multiplicative : bool
            If True, the bond tolerance is multiplicative
        verbose : bool
            If True, print messages
        """

        self.__elements = []
        self.__coords = []
        self.__radii = {}
        self.__bonds = []
        self.__tolerance = bond_tolerance
        self.__cube_tolerance = box_tolerance
        self.__multi = multiplicative
        self.__verbose = verbose

        if len(atoms) == 0:
            raise ValueError("Bonding.__init__(): No atoms provided.")

        for atom in atoms:
            self.__elements.append(atom.element)
            self.__coords.append(atom.coordinates)
            self.__radii[atom.element] = atom.covalent_radius
        
        self.__cube_size = self.__get_cube_size()
        self.__pair_distance = self.__get_possible_distances()

        if self.__verbose:
            print(f"Initialized Bonding object with {len(atoms)} atoms.")
            print(f"Cube size: {self.__cube_size:.2f} Å")
            print("Possible distances:")
            for pair, dist in self.__pair_distance.items():
                print(f"  {pair}: {dist:.2f} Å")
        
    def __get_cube_size(self):
        """ Method to get the cube size

        Returns
        -------
        cube_size : float
            The cube size
        """
        return max(self.__radii.values()) * 2 + self.__cube_tolerance

    def __get_possible_distances(self):
        """ Method to get the possible distances between atoms
        
        This method computes the possible distances between atoms based
        on their radii and the provided tolerance. The idea is to create
        a dictionary of possible distances for each pair of elements,
        which can be used to quickly check if two atoms are bonded based on
        their distance.

        Returns
        -------
        possible_distances : dict
            A dictionary with the possible distances between atoms
        """
        # Get the unique elements and their radii
        elements = set(self.__elements)

        # Compute the possible distances
        possible_distances = {}

        # Iterate over all pairs of elements
        for e1 in elements:
            for e2 in elements:
                # Sort the pair of elements to avoid duplicates
                i, j = sorted((e1, e2))
                if (i, j) not in possible_distances:
                    distance = self.__radii[i] + self.__radii[j]

                    # If the tolerance is multiplicative, multiply the
                    # distance by the tolerance. Otherwise, add the
                    # tolerance to the distance
                    if self.__multi:
                        distance *= (1 + self.__tolerance)
                    else:
                        distance += self.__tolerance

                    # Add the distance to the dictionary
                    possible_distances[(i, j)] = distance

        return possible_distances
    
    def _get_grid_information(self):
        """ Method to get the limits of the bounding box

        Returns
        -------
        grid_info : tuple
            A tuple with the minimum and maximum coordinates and
            the number of cubes in each direction
        """
        # Get the minimum and maximum coordinates
        min_coords = np.min(self.__coords, axis=0)
        max_coords = np.max(self.__coords, axis=0)

        # Calculate the side distances of the bounding box
        sides = max_coords - min_coords

        # Get the number of cubes in each direction
        num_cubes = np.ceil(sides / self.__cube_size).astype(int)

        # Get the new maximum coordinates
        max_coords = min_coords + num_cubes * self.__cube_size

        return min_coords, max_coords, num_cubes
    
    def _put_atoms_in_cubes(self):
        """ Method to put atoms in cubes

        This method creates a dictionary of cubes, where each cube is
        defined by its indices in the grid and contains a list of atoms
        that are located in that cube. This allows for efficient neighbor
        searching when determining bonds.

        Returns
        -------
        cubes : dict
            A dictionary with the cubes and the atoms contained in each cube
        """
        # Get the minimum coordinates
        mi = np.min(self.__coords, axis=0)

        # Create the cubes dictionary
        cubes = {}

        # Iterate over all atoms
        for i, coord in enumerate(self.__coords):
            # Get the cube indices for the atom
            cube_indices = np.floor((coord - mi) / self.__cube_size).astype(int)

            # Convert the cube indices to a tuple to use as a key in the dictionary
            cube_key = tuple(cube_indices)

            # Add the atom to the corresponding cube in the dictionary
            if cube_key not in cubes:
                cubes[cube_key] = []
            cubes[cube_key].append(i)

        if self.__verbose:
            print(f"Put {len(self.__coords)} atoms in {len(cubes)} cubes.")
            for cube_key, atom_indices in cubes.items():
                print(f"Cube {cube_key}: {len(atom_indices)} atoms")

        return cubes
    
    def _check_bonds(self, atoms1 : list, atoms2 : list) -> list:
        """ Method to check the bonds between two lists of atoms

        This method checks the distances between all pairs of atoms in
        the two lists and determines if they are bonded based on the
        possible distances computed earlier.

        Parameters
        ----------
        atoms1 : list
            List of indices of the first set of atoms
        atoms2 : list
            List of indices of the second set of atoms

        Returns
        -------
        bonds : list of tuples
            A list of tuples, where each tuple contains the indices of
            two bonded atoms
        """
        # Initialize the list of bonds
        bonds = []

        # Get the coordinates of the atoms as NumPy arrays
        coords1 = np.array([self.__coords[i] for i in atoms1])
        coords2 = np.array([self.__coords[j] for j in atoms2])

        # Compute the distances between the two sets of atoms
        distances = cdist(coords1, coords2, metric='euclidean')

        # Iterate over all pairs of atoms
        for idi, i in enumerate(atoms1):
            for idj, j in enumerate(atoms2):

                # Check if the atoms are the same
                if i == j:
                    continue

                # Get the elements and coordinates of the atoms
                e1, e2 = self.__elements[i], self.__elements[j]
                pair = tuple(sorted((e1, e2)))
                
                # Check if the atoms are bonded
                if distances[idi, idj] <= self.__pair_distance[pair]:
                    bonds.append(tuple(sorted([i, j])))
        
        return bonds
    
    def distance_matrix(self) -> tuple[np.ndarray, list]:
        """ Method to compute the distance matrix

        This method computes the distance matrix between all pairs of atoms
        and determines the bonds based on the possible distances. The distance
        matrix is return as a NumPy array, and the bonds are also returned as
        a list of tuples.

        Returns
        -------
        dist_mat : np.ndarray
            The distance matrix between all pairs of atoms
        bonds : list of tuples
            A list of tuples, where each tuple contains the indices of
            two bonded atoms
        """
        # Get the number of atoms
        num_atoms = len(self.__elements)

        if self.__verbose:
            print(f"Computing distance matrix for {num_atoms} atoms ...")

        # Check if there are any atoms
        if num_atoms == 0:
            raise ValueError("Bonding.distance_matrix() No atoms were "
                            "provided.")

        # Check if there are too many atoms
        if num_atoms > 15000:
            raise ValueError("Bonding.distance_matrix() The number of atoms "
                             "is too large to compute the distance matrix. "
                             "Consider using a more efficient method for "
                             "systems over 15000 atoms.")

        # Initialize the list of bonds
        self.__bonds = []

        # Check if there is only one atom
        if num_atoms == 1:
            return np.zeros((1, 1)), self.__bonds
        
        # Get the coordinates as a NumPy array
        coords = np.array(self.__coords)

        # Initialize the distance matrix with zeros
        dist_mat = cdist(coords, coords, metric='euclidean')

        # Iterate over all pairs of atoms
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):

                # Get the elements and coordinates of the atoms
                e1, e2 = self.__elements[i], self.__elements[j]
                pair = tuple(sorted((e1, e2)))

                # Check if the atoms are bonded
                if dist_mat[i][j] <= self.__pair_distance[pair]:
                    self.__bonds.append(tuple(sorted([i, j])))
        
        if self.__verbose:
            print("Distance matrix computed.")
            print(dist_mat)
            print("\nBonds found:")
            for bond in self.__bonds:
                print(f"  Atoms {bond[0]} and {bond[1]} are bonded.")

        return dist_mat, self.__bonds
    
    def find_bonds(self, force : bool = False) -> list:
        """ Method to find the bonds between atoms

        This method uses the cube information and the possible distances
        to efficiently determine which pairs of atoms are bonded.

        Parameters
        ----------
        force : bool
            If True, force the computation of the bonds even if the bond
            list is already computed. Default is False.

        Returns
        -------
        bonds : list of tuples
            A list of tuples, where each tuple contains the indices of
            two bonded atoms
        """
        # Check if the bonds have already been computed
        if not force and len(self.__bonds) > 0:
            return self.__bonds
        
        # Check if there are any atoms
        if len(self.__elements) == 0:
            raise ValueError("Bonding.find_bonds() No atoms were "
                            "provided.")
        
        if self.__verbose:
            print("Finding bonds between atoms ...")
            grid_info = self._get_grid_information()
            print(f"Lower corner: {grid_info[0]}")
            print(f"Upper corner: {grid_info[1]}")
            print(f"Number of cubes: {grid_info[2]}")

        # Put the atoms in cubes
        cubes = self._put_atoms_in_cubes()

        # Initialize the list of bonds
        self.__bonds = []

        # Iterate over all cubes
        for cube_key, atom_indices in cubes.items():
            
            # Iterate over all neighboring cubes, including the current one
            for address in self.ADDRESSES:

                # Get the neighbor key
                neighbor_key = tuple(np.array(cube_key) + np.array(address))

                # Check if the neighbor key is in the dictionary of cubes
                if neighbor_key in cubes:
                    neighbor_indices = cubes[neighbor_key]

                    # Check the bonds between the atoms in the current cube
                    # and the atoms in the neighbor cube
                    self.__bonds += self._check_bonds(atom_indices,
                                                      neighbor_indices)

        return self.__bonds