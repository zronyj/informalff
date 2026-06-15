import numpy as np
from hashlib import md5
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from warnings import warn

from .elements import PTE
from .molecule import Molecule

class AtomNeighborhood:
    """ Class to represent the neighborhood of an atom
    
    Attributes
    ----------
    neighbors_spher : tuple[np.ndarray]
        The spherical coordinates of the neighbors
        around the central atom
    neighbors_cart : tuple[np.ndarray]
        The cartesian coordinates of the neighbors
        around the central atom
    pivot : int
        The index of the central atom
    """
    def __init__(self,
                 center : np.ndarray,
                 neighbors_cart : tuple[np.ndarray],
                 pivot : tuple[int],
                 verbose : bool = False):
        
        self.center = center
        self.neighbors_cart = neighbors_cart
        self.__verbose = verbose
        self.__num_ngbrs = len(neighbors_cart)

        # Center the group of atoms
        self.__move_to_origin()

        # Calculate the pivot's coordinates
        if len(pivot) == 0:
            raise ValueError("AtomNeighborhood.__init__() No pivot "
                             "atoms selected")
        elif len(pivot) == 1:
            self.pivot = self.neighbors_cart[pivot[0]]
        else:
            self.pivot = np.zeros(3)
            for p in pivot:
                self.pivot += self.neighbors_cart[p]
            self.pivot /= len(pivot)

        # Rotate the group of atoms
        self.__rotate_to_pivot()

        # Compute the spherical coordinates
        transformed = [self._cart_to_spher(n) for n in self.neighbors_cart]
        self.neighbors_spher = tuple(transformed)
    
    def __str__(self):
        """ Return a string representation of the atomic neighborhood
        
        Returns
        -------
        output : str
            A string representation of the atomic neighborhood
        """
        output = "-" * 82
        output += f"\n{'AtomNeighborhood':^82}\n"
        output += "-" * 82 + "\n"

        output += f"{'Center':>13} | "
        output += f"{self.center[0]:10.6f} "
        output += f"{self.center[1]:10.6f} "
        output += f"{self.center[2]:10.6f}\n"

        sp_pivot = self._cart_to_spher(self.pivot)

        output += f"{'Pivot':>13} | "
        output += f"{self.pivot[0]:10.6f} "
        output += f"{self.pivot[1]:10.6f} "
        output += f"{self.pivot[2]:10.6f} "
        output += f"{sp_pivot[0]:10.6f} "
        output += f"{sp_pivot[1]:10.6f} "
        output += f"{sp_pivot[2]:10.6f}\n"

        for i in range(self.__num_ngbrs):
            output += f"{'Neighbor':>9}[{i:>2}] | "
            output += f"{self.neighbors_cart[i][0]:10.6f} "
            output += f"{self.neighbors_cart[i][1]:10.6f} "
            output += f"{self.neighbors_cart[i][2]:10.6f} "
            output += f"{self.neighbors_spher[i][0]:10.6f} "
            output += f"{self.neighbors_spher[i][1]:10.6f} "
            output += f"{self.neighbors_spher[i][2]:10.6f}\n"
        output += "-" * 82
        return output
    
    def __move_to_origin(self):
        """ Move the group of atoms to the origin
        """
        self.neighbors_cart = [n - self.center for n in self.neighbors_cart]
        self.center = np.array([0, 0, 0])
    
    def __rotate_to_pivot(self):
        """ Rotate the group of atoms to the pivot
        """
        # Compute the rotation matrix
        t_vec = np.array([0.0, 0.0, 1.0])
        p_vec = deepcopy(self.pivot) / np.linalg.norm(self.pivot)
        cross = np.cross(self.pivot, t_vec)
        dot = np.dot(self.pivot, t_vec)
        mag_cross = np.linalg.norm(cross)

        # Compute the angle
        angle = np.arctan2(mag_cross, dot)
        
        # Computing the axis and ensuring numerical stability
        if mag_cross < 1e-12:
            axis = np.zeros(3, dtype=np.float64)
        else:
            axis = cross * angle / mag_cross

        # Build the rotation matrix (plus a sanity check)
        rot_mat = R.from_rotvec(axis).as_matrix()
        if np.abs(np.dot(rot_mat @ p_vec, t_vec) - 1) > 1e-9:
            rot_mat = R.from_rotvec(axis * -1).as_matrix()
        if np.abs(np.dot(rot_mat @ p_vec, t_vec) - 1) > 1e-9:
            raise ValueError("AtomNeighborhood.__rotate_to_pivot() "
                       "Rotation failed")

        # Rotate the group of atoms
        self.neighbors_cart = [rot_mat @ n for n in self.neighbors_cart]
        
        # Rotate the pivot
        self.pivot = rot_mat @ p_vec
    
    @staticmethod
    def _cart_to_spher(coord : np.ndarray) -> np.ndarray:
        """Transform coordinates from cartesian to spherical
        
        Parameters
        ----------
        coord : np.ndarray
            The cartesian coordinates to be transformed
        
        Returns
        -------
        np.ndarray
            The spherical coordinates after transformation
        """
        x, y, z = coord

        rho = np.hypot(x, y)
        r = np.hypot(rho, z)
        t = np.arctan2(rho, z)
        p = np.arctan2(y, x)

        return np.array([r, t, p])

# ------------------------------------------------------- #
#                   The Atom Types Class                  #
# ------------------------------------------------------- #
class AtomTypes:
    """ A class to determine the types of atoms in a molecule

    The types of atoms are determined by their hybridization and other properties.
    """

    def __init__(self,
                 molecule : Molecule,
                 verbose : bool = False):
        self.__molecule = molecule
        self.__verbose = verbose

        self.__bonds, self.__bondtypes = self.__molecule.get_bond_types()
    
    def __compute_neighbor_priority(self,
                                    idx : int,
                                    neighbors : list) -> list:
        """Find the lowest priority neighbor
        
        Parameters
        ----------
        idx : int
            The index of the atom
        neighbors : list
            The neighbors of the atom

        Returns
        -------
        priority : list
            The priority of the neighbors
        """
        # Keep track of the atoms that have already been visited
        old_branches = [[] for _ in range(len(neighbors))]

        # Iterate over 50 levels deep in the graph
        for level in range(50):

            # Get all the branches stemming from the neighbors
            branches = []
            for ing, ng in enumerate(neighbors):

                # Get the branch
                branch = self.__molecule.graph.get_branch(idx, ng, level, [])

                # Remove the atoms that have already been visited in previous
                # levels
                branch = [b for b in branch if b not in old_branches[ing]]

                # Compute the priority as the negative of the atomic number
                prio = [-1 * self.__molecule.atoms[b].number for b in branch]

                # Safety check to see if the branch is empty (and therefore
                # the priority should be -1000)
                if len(prio) == 0:
                    branches.append(-1000)
                else:
                    branches.append(max(prio))
                
                # Update the list of visited atoms
                old_branches[ing] += branch
            
            # If NOT all branches have the same priority, break
            if not all(b == branches[0] for b in branches):
                break
        
        # Return the indices of the atoms with the highest priority
        return [i for i, b in enumerate(branches) if b == max(branches)]
    
    def __build_neighbor_types(self,
                               idx : int,
                               previous : int = -1,
                               neighbors : list = []) -> str:
        """Build the type of the atom and its neighbors
        
        Parameters
        ----------
        idx : int
            The index of the atom
        previous : int
            The index of the previous atom (origin atom)
        neighbors : list
            The neighbors of the atom

        Returns
        -------
        output : str
            The type of the atom
        """
        # Initialize the output
        output = ""
        
        # If there is no origin atom or neighbors, raise an error
        if previous == -1 and len(neighbors) == 0:
            raise ValueError("AtomTypes.__build_neighbor_types() "
                             "No origin atom or neighbors provided")
        
        # If there is no origin atom, just prepare the element
        if previous == -1 and len(neighbors) != 0:
            output = f"{self.__molecule.atoms[idx].element}"
        
        # If there is an origin atom, prepare the bond to the previous
        # atom and the element
        if previous != -1:
            idb = self.__bonds.index(tuple(sorted([idx, previous])))
            order = self.__bondtypes[idb]
            output = f"<{order}>{self.__molecule.atoms[idx].element}"
        
        # If there are no neighbors, return
        if len(neighbors) == 0:
            return output
        
        # Extract the indices of the neighbors
        flat_ngbrs = []
        for n in neighbors:
            if isinstance(n, int):
                flat_ngbrs.append(n)
            elif isinstance(n, dict):
                flat_ngbrs.append(list(n.keys())[0])
            else:
                raise TypeError("AtomTypes.__build_neighbor_types() "
                                f"Unknown type of neighbor: {n} : {type(n)}")

        # Get the neighbors in the right order
        ngbr_order = self._ordering(idx)

        # Remove the neighbor(s) where we are coming from
        ngbr_order = [i for i in ngbr_order if i in flat_ngbrs]

        # Sort the neighbors
        sorted_neighbors = [neighbors[flat_ngbrs.index(i)] for i in ngbr_order]

        # Start adding the neighbors (this is a recursive function that
        # depends on the depth of the surrounding graph)
        output += "["
        for idn, n in enumerate(sorted_neighbors):
            if isinstance(n, int):
                output += self.__build_neighbor_types(n, idx, [])
            elif isinstance(n, dict):
                output += self.__build_neighbor_types(list(n.keys())[0],
                                                      idx,
                                                      [idx] + list(n.values())[0])
            else:
                raise TypeError("AtomTypes.__build_neighbor_types() "
                                f"Unknown type of neighbor: {n} : {type(n)}")
            if idn < len(neighbors) - 1:
                output += ","
        output += "]"

        return output
    
    def _ordering(self,
                   idx : int,
                   disc_angle : float = 20,
                   reverse : bool = False) -> list:
        """Compute the correct ordering of the neighbor atoms
        
        Parameters
        ----------
        idx : int
            The index of the atom to use as reference
        disc_angle : float
            The angle of each azimutal segment of the sphere around
            the reference atom. Default is 20 degrees.
        reverse : bool
            If the neighbors should be considered in reverse
            order. Default is False.
        
        Returns
        -------
        new_order : list
            The list of atom indices indicating the correct
            ordering of the atoms around the reference atom
        """
        # Get all the neighbors around the atom
        ngbrs = self.__molecule.graph.get_neighbors(idx, depth=1)

        # Find the lowest priority neighbor
        priority = self.__compute_neighbor_priority(idx, ngbrs)

        # Create an atom neighborhood object to more easily handle the
        # calculation of the angles
        atm_pk = AtomNeighborhood(
            self.__molecule.atoms[idx].coordinates,
            tuple([self.__molecule.atoms[ng].coordinates for ng in ngbrs]),
            tuple(priority),
            verbose=True)
    
        # Get the angle in radians and how many segments are
        # to be used
        segments = int(np.ceil(np.pi / (disc_angle / 180 * np.pi)))
        delta = np.pi / segments
        
        # Iterate over all disc segments
        index_order = []
        for s in range(segments):

            # Create an empty dictionary to keep track of the key, value
            # pairs. Key: atom number, Value: theta angle
            d_atoms = {}

            # Iterate over neighbors
            for i, spher in enumerate(atm_pk.neighbors_spher):

                # If the atom is in that disc, add it to the dictionary
                if spher[1] >= s * delta and spher[1] < (s+1) * delta:
                    d_atoms[i] = spher[2]
            
            # If there's no atoms in the dictionary, skip
            if len(d_atoms) == 0:
                continue
            # If there are atoms in the dictionary, sort and add them to the
            # final (ordered) list
            else:
                # Sort the dictionary
                for k, v in dict(sorted(d_atoms.items(),
                                        key=lambda x: x[1])).items():
                    index_order.append(k)
        
        # Transform the indices of the neighbors to the indices of
        # the atoms in the molecule
        pre_order = []
        ele_order = []
        for i in index_order:
            pre_order.append([self.__molecule.atoms[ngbrs[i]].number, ngbrs[i]])
            ele_order.append(self.__molecule.atoms[ngbrs[i]].element)
        pre_order = np.array(pre_order)

        # Cycle the list to ensure the highest priority neighbor is last
        highest = np.max(pre_order[:, 0])
        for i in range(len(pre_order)):
            temp_order = np.roll(pre_order, i, axis=0)
            if temp_order[-1,0] == highest and temp_order[0,0] != highest:
                break
        
        # Transform the indices of the neighbors to the indices of
        # the atoms in the molecule
        new_order = temp_order[:,1].tolist()

        if reverse:
            return new_order[::-1]
        else:
            return new_order
    
    def surrounding_atoms(self, idx : int, depth : int = 1) -> str:
        """Encode the atom and its neighbors in a readable format
        
        Parameters
        ----------
        idx : int
            The index of the atom to use as reference
        depth : int
            The depth of the neighborhood to consider. Default is 1.
        
        Returns
        -------
        output : str
            A string representation of the atom and its neighbors
        """
        # Get the neighbors
        ngbrs = self.__molecule.graph.get_neighbors(idx, depth=depth)

        # Build the string
        output = self.__build_neighbor_types(idx, -1, ngbrs)

        return output
    
    def surrounding_hash(self, idx : int, depth : int = 1) -> str:
        """Get the hash of the atom and its neighbors
        
        Parameters
        ----------
        idx : int
            The index of the atom to use as reference
        depth : int
            The depth of the neighborhood to consider. Default is 1.
        
        Returns
        -------
        hash : str
            The hash of the atom and its neighbors
        """
        # Get the neighbors
        readable = self.surrounding_atoms(idx, depth=depth)

        # Get the hash
        hash = md5(readable.encode()).hexdigest()

        return hash