import numpy as np
from hashlib import md5
from copy import deepcopy
from warnings import warn
from abc import ABC, abstractmethod  # To be able to create several drivers
from scipy.spatial.transform import Rotation as R

from .elements import PTE
from .molecule import Molecule

# ------------------------------------------------------- #
#               The Atom Neighborhood Class               #
# ------------------------------------------------------- #
class AtomNeighborhood:
    """ Class to represent the neighborhood of an atom
    
    Attributes
    ----------
    center : np.ndarray
        The cartesian coordinates of the central atom
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
                 center : list,
                 indices : list,
                 neighbors : list[list]):
        
        if not isinstance(center, list):
            raise TypeError("AtomNeighborhood.__init__() center should be "
                            f"a list, but got {type(center)} instead")        
        if len(center) != 2:
            raise ValueError("AtomNeighborhood.__init__() center should be "
                             "a list containing the element and the cartesian "
                             "coordinates")
        if not isinstance(center[0], str):
            raise ValueError("AtomNeighborhood.__init__() The first entry of "
                             "center should be the element symbol as a string")
        if not isinstance(center[1], np.ndarray):
            raise ValueError("AtomNeighborhood.__init__() The second entry of"
                             "center should be the cartesian coordinates as "
                             "a NumPy array")

        if len(indices) != len(neighbors):
            raise ValueError("AtomNeighborhood.__init__() indices and "
                             "neighbors should be lists of the same length")
        if not isinstance(indices, list):
            raise TypeError("AtomNeighborhood.__init__() indices should be "
                            f"a list, but got {type(indices)} instead")
        if not all([isinstance(i, int) for i in indices]):
            raise TypeError("AtomNeighborhood.__init__() indices should be "
                            "a list of integers")
        
        if not isinstance(neighbors, list):
            raise TypeError("AtomNeighborhood.__init__() neighbors should be "
                            f"a list, but got {type(neighbors)} instead")
        if not all([isinstance(n, list) for n in neighbors]):
            raise TypeError("AtomNeighborhood.__init__() neighbors should be "
                            "a list of lists")
        if not all([len(n) == 2 for n in neighbors]):
            raise ValueError("AtomNeighborhood.__init__() each neighbor should"
                             " be a list containing the element symbol and "
                             "the cartesian coordinates")
        if not all([isinstance(n[0], str) for n in neighbors]):
            raise ValueError("AtomNeighborhood.__init__() The first entry of "
                             "each neighbor should be the element symbol as a "
                             "string")
        if not all([isinstance(n[1], np.ndarray) for n in neighbors]):
            raise ValueError("AtomNeighborhood.__init__() The second entry of "
                             "each neighbor should be the cartesian "
                             "coordinates as a NumPy array")

        self.__symbol = center[0]
        self.center = center[1]

        self.__indices = indices
        self.__symbols = [n[0] for n in neighbors]
        self.__numbers = [PTE[n[0]].number for n in neighbors]
        self.neighbors_cart = [n[1] for n in neighbors]
        self.__num_ngbrs = len(neighbors)

        self.__pivot = np.array([0.0, 0.0, 1.0])
        self.__change = False

        # Center the group of atoms
        self.__move_to_origin()

        # Compute the spherical coordinates
        self.__to_spherical()
    
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
    
    def __to_spherical(self):
        """ Compute the spherical coordinates of the group of atoms """
        transformed = [self._cart_to_spher(n) for n in self.neighbors_cart]
        self.neighbors_spher = tuple(transformed)
    
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

    @property
    def pivot(self):
        return self.__pivot
    
    @pivot.setter
    def pivot(self, new_pivot : int | np.ndarray | tuple):
        """ Rotate the group of atoms to the pivot

        Parameters
        ----------
        new_pivot : int | np.ndarray | tuple | list
            The pivot to rotate the group of atoms to
        """
        if isinstance(new_pivot, int):
            if new_pivot < 0 or new_pivot >= self.__num_ngbrs:
                raise ValueError("AtomNeighborhood.pivot() Invalid pivot "
                                 f"atom index: {new_pivot}")
            self.__pivot = self.neighbors_cart[new_pivot]
        elif isinstance(new_pivot, np.ndarray):
            if new_pivot.shape != (3,):
                raise ValueError("AtomNeighborhood.pivot() Invalid pivot "
                                 f"shape {new_pivot.shape}")
            self.__pivot = new_pivot
        elif isinstance(new_pivot, tuple) or isinstance(new_pivot, list):
            if len(new_pivot) == 0:
                raise ValueError("AtomNeighborhood.pivot() The provided "
                                 f"list is empty!")
            if all([isinstance(p, np.ndarray) for p in new_pivot]):
                if any([p.shape != (3,) for p in new_pivot]):
                    raise ValueError("AtomNeighborhood.pivot() Invalid pivot "
                                     f"shape")
                self.__pivot = np.zeros(3)
                for p in new_pivot:
                    self.__pivot += p
                self.__pivot /= len(new_pivot)

            elif all([isinstance(p, int) for p in new_pivot]):
                if any([p < 0 or p >= self.__num_ngbrs for p in new_pivot]):
                    raise ValueError("AtomNeighborhood.pivot() Invalid pivot "
                                     f"atom index")
                self.__pivot = np.zeros(3)
                for p in new_pivot:
                    self.__pivot += self.neighbors_cart[p]
                self.__pivot /= len(new_pivot)
        else:
            raise ValueError("AtomNeighborhood.pivot() Invalid pivot "
                             "type")

        # Compute the rotation matrix
        t_vec = np.array([0.0, 0.0, 1.0])
        p_vec = deepcopy(self.__pivot) / np.linalg.norm(self.__pivot)
        cross = np.cross(self.__pivot, t_vec)
        dot = np.dot(self.__pivot, t_vec)
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
        self.__pivot = rot_mat @ p_vec

        # Recalculate the spherical coordinates
        self.__to_spherical()

        # Register the change
        self.__change = True

    def order(self,
              azimuth_angle : float = 20,
              reverse : bool = False) -> list:
        """Compute the correct order of the neighboring atoms
        
        Parameters
        ----------
        azimuth_angle : float
            The angle of each azimuthal segment of the sphere around
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
        if not self.__change:
            warn("AtomNeighborhood.order() WARNING! The pivot atom has "
                 f"probably not been established for {self.__symbol} "
                 f"with neighbors {self.__indices}. "
                 f"Working with the default: {self.__pivot}")

        # Get the angle in radians and how many segments are
        # to be used
        segments = int(np.ceil(np.pi / (azimuth_angle / 180 * np.pi)))
        delta = np.pi / segments
        
        # Iterate over all disc segments
        index_order = []
        for s in range(segments):

            # Create an empty dictionary to keep track of the key, value
            # pairs. Key: atom number, Value: theta angle
            d_atoms = {}

            # Iterate over neighbors
            for i, spher in enumerate(self.neighbors_spher):

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
            pre_order.append([self.__numbers[i], self.__indices[i]])
            ele_order.append(self.__symbols[i])
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

# ------------------------------------------------------- #
#                   The Atom Types Class                  #
# ------------------------------------------------------- #
class AtomTypes(ABC):
    """ A class to determine the types of atoms in a molecule

    The types of atoms are determined by their hybridization and other properties.
    """

    def __init__(self, molecule : Molecule):

        if not isinstance(molecule, Molecule):
            raise TypeError("AtomTypes() Invalid type for molecule")

        self._molecule = molecule

        self._bonds, self._bondtypes = self._molecule.get_bond_types()

        self._atom_nhs = []
        for ida in range(len(self._molecule)):
            # Get all the neighbors around the atom
            ngbrs = self._molecule.graph.get_neighbors(ida, depth=1)

            # Find the lowest priority neighbor
            priority = self.__compute_neighbor_priority(ida, ngbrs)

            # Create an atom neighborhood object to more easily handle the
            # calculation of the angles
            self._atom_nhs.append(
                AtomNeighborhood(
                    self._molecule[ida],
                    ngbrs,
                    [self._molecule[ng] for ng in ngbrs]
                )
            )

            self._atom_nhs[-1].pivot = priority

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
                branch = self._molecule.graph.get_branch(idx, ng, level, [])

                # Remove the atoms that have already been visited in previous
                # levels
                branch = [b for b in branch if b not in old_branches[ing]]

                # Compute the priority as the negative of the atomic number
                prio = [-1 * self._molecule.atoms[b].number for b in branch]

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
    
    @abstractmethod
    def get_atom_type(self, idx : int) -> str:
        pass

    @abstractmethod
    def get_atom_hash(self) -> str:
        pass

    def get_atom_types(self) -> list:
        """Get the types of all the atoms in the molecule
        
        Returns
        -------
        atom_types : list
            A list with the types of all the atoms in the molecule
        """
        return [self.get_atom_type(i) for i in range(len(self._molecule))]
    
    def __as_xyz__(self) -> list:
        output = [str(len(self._molecule))]
        comment = f"Molecule {self._molecule.name} with atom types - "
        comment += "created by InformalFF"
        output.append(comment)
        for i in range(len(self._molecule)):
            atom = self._molecule[i]
            line = f"{atom[0]:<2} "
            line += f"{atom[1][0]:14.8f} {atom[1][1]:14.8f} {atom[1][2]:14.8f}"
            line += f"\t{self.get_atom_type(i)}"
            output.append(line)
        return output
    
    def __str__(self):
        
        lines = self.__as_xyz__()
        width = len(str(len(self._molecule)))
        output = []
        for i, l in enumerate(lines[2:]):
            output.append(f"{i:>{width}}. {l}")

        return "\n".join(output)

class SurroundingAtomTypes(AtomTypes):
    """ A class to build the atom types in a molecule
    
    The types are determined by each atom's element, its bonds,
    and the elements of its neighbors.
    
    Attributes
    ----------
    _molecule : Molecule
        A Molecule object
    __depth : int
        The depth of the neighborhood to consider
    """
    def __init__(self, molecule : Molecule, depth : int = 1):
        """Constructor method for the SurroundingAtomTypes class

        Parameters
        ----------
        molecule : Molecule
            A Molecule object
        depth : int
            The depth of the neighborhood to consider. Default is 1.
        """
        super().__init__(molecule)
        self.__depth = depth

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
            output = f"{self._molecule.atoms[idx].element}"
        
        # If there is an origin atom, prepare the bond to the previous
        # atom and the element
        if previous != -1:
            idb = self._bonds.index(tuple(sorted([idx, previous])))
            order = self._bondtypes[idb]
            output = f"<{order}>{self._molecule.atoms[idx].element}"
        
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
        ngbr_order = self._atom_nhs[idx].order()

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
    
    def get_atom_type(self, idx : int) -> str:
        """Encode the atom and its neighbors in a readable format
        
        Parameters
        ----------
        idx : int
            The index of the atom to use as reference
        
        Returns
        -------
        output : str
            A string representation of the atom and its neighbors
        """
        # Get the neighbors
        ngbrs = self._molecule.graph.get_neighbors(idx, depth=self.__depth)

        # Build the string
        output = self.__build_neighbor_types(idx, -1, ngbrs)

        return output
    
    def get_atom_hash(self, idx : int) -> str:
        """Get the hash of the atom and its neighbors
        
        Parameters
        ----------
        idx : int
            The index of the atom to use as reference
        
        Returns
        -------
        hash : str
            The hash of the atom and its neighbors
        """
        # Get the neighbors
        readable = self.get_atom_type(idx)

        # Get the hash
        hash = md5(readable.encode()).hexdigest()

        return hash

class Mol2AtomTypes(AtomTypes):
    """Class to determine the MOL2 types of atoms in a molecule
    
    The types are determined by each atom's element, its bonds,
    and the elements of its neighbors.
    
    Attributes
    ----------
    _molecule : Molecule
        A Molecule object
    """
    def __init__(self, molecule : Molecule):
        super().__init__(molecule = molecule)

    # TODO: Build a massive if else tree to handle all the cases
    
    def __build_neighbor_bonds(self, idx : int) -> dict:

        ngbr_bonds = {}
        for bond, bond_type in zip(self._bonds, self._bondtypes):
            if idx in bond:
                ngbr_bonds[bond] = bond_type

        return ngbr_bonds
    
    def __get_sigma_pi(self, idx : int) -> dict:
        
        sigma = 0
        bonds = self.__build_neighbor_bonds(idx)
        valence, lps, cap = self._molecule.atoms[idx].get_valence()
        deloc = []
        strange = []
        coord = len(bonds)
        for bond, bond_type in bonds.items():
            if bond_type >= 1:
                sigma += 1
            if bond_type > 1 and bond_type < 2:
                deloc.append(bond)
            if bond_type > 0 and bond_type < 1:
                strange.append(bond)
        
        pi = valence - sigma

        return {'coordination' : coord,
                'sigma' : sigma,
                'pi' : pi,
                'delocalized' : deloc,
                'strange' : strange}
    
    def get_atom_type(self, idx : int) -> str:
        """Encode the atom in the MOL2 format
        
        Parameters
        ----------
        idx : int
            The index of the atom to use as reference
        
        Returns
        -------
        output : str
            A string representation of the atom and its neighbors
        """
        special = ["C", "N", "O", "P", "S"]
        typ = self._molecule.atoms[idx].element
        bond_info = self.__get_sigma_pi(idx)

        if typ in special:
            if len(bond_info['delocalized']) == 2:
                return f"{typ}.ar"
            valence, lps, cap = self._molecule.atoms[idx].get_valence()
            nlp = (valence - bond_info['sigma'] - bond_info['pi'] ) / 2
            steric = bond_info['coordination'] + nlp - 1
            return f"{typ}.{int(steric)}"
        
        return typ

    def get_atom_hash(self, idx : int) -> str:
        """Get the hash of the atom and its neighbors
        
        Parameters
        ----------
        idx : int
            The index of the atom to use as reference
        
        Returns
        -------
        hash : str
            The hash of the atom and its neighbors
        """
        # Get the neighbors
        readable = self.get_atom_type(idx)

        # Get the hash
        hash = md5(readable.encode()).hexdigest()

        return hash