import os                                          # To navigate the file system
import copy                                        # To copy objects
import json                                        # To parse json files
import warnings                                    # To throw warnings instead of raising errors
import numpy as np                                 # To do basic scientific computing
import pandas as pd                                # To manage tables and databases
from pathlib import Path                           # To locate files in the file system
from functools import lru_cache                    # To cache functions
from multiprocessing import Pool, Process, Manager # To parallelize jobs
import scipy.constants as cts                      # Universal constants
from scipy.spatial.transform import Rotation as R  # To be able to construct rotation matrices

from .atom import Atom, PERIODIC_TABLE, fibonacci_grid_shell

BOHR = 1 / (cts.physical_constants["Bohr radius"][0] * 1e10)

# ------------------------------------------------------- #
#                The Molecular Graph Class                #
# ------------------------------------------------------- #

class MolecularGraph:
    """ Class to represent the graph of a molecule 

    This class is used to represent the graph of a molecule
    by considering each atom with its id in a dictionary,
    and adding several methods for the analysis of the graph.

    Attributes
    ----------
    atoms : list
        A list with all the atoms in the molecule
    bonds : list of list
        A list with all the pairs of atoms defining the bonds of the molecule
    graph : dict
        A dictionary with the connectivity of the molecule
    """
    def __init__(self, atoms : int, bonds : list):
        """
        MolecularGraph constructor method
        
        Parameters
        ----------
        atoms : int
            The number of atoms in the molecule
        bonds : list of list
            A list of pairs of atoms defining the bonds of the molecule
        """
        self.atoms = list(range(atoms))
        self.bonds = bonds

        # Initialize the dictionary for the graph
        self.graph = {a : [] for a in self.atoms}

        # Iterate over all bonds
        for b in self.bonds:
            self.graph[b[0]].append(b[1])
            self.graph[b[1]].append(b[0])
        
        # Iterate over all atoms
        for a in self.atoms:
            # Remove any duplicates
            self.graph[a] = set(self.graph[a])

    def get_neighbors(self, atom : int) -> list:
        """ Method to get the neighbors of a given atom
        
        Parameters
        ----------
        atom : int
            The atom
        
        Returns
        -------
        neighbors : list
            The neighbors of the atom"""
        # Return the neighbors
        return list(self.graph[atom])
    
    def get_branch(self,
                   atom1 : int,
                   atom2 : int,
                   depth : int,
                   path : list = []) -> list:
        """ Method to get the molecular branch stemming from two atoms

        Parameters
        ----------
        atom1 : int
            The first atom
        atom2 : int
            The second atom
        depth : int
            The depth of the branch (how many bonds away should it be)
        path : list
            The path that has been walked already

        Returns
        -------
        branch : list
            Tree structure of atoms in the branch"""
        # Check that atom1 and atom2 are not the same
        if atom1 == atom2:
            raise ValueError("MolecularGraph.get_branch() The two atoms"
                             " should be different!")
        
        # Check that atom1 and atom2 are in the molecule
        if (atom1 not in list(range(len(self.atoms))) or
                atom2 not in list(range(len(self.atoms)))):
            raise ValueError("MolecularGraph.get_branch() The two atoms"
                             " should be in the molecule!")

        # If no bonds are found, something is wrong
        if len(self.bonds) == 0:
            raise ValueError("MolecularGraph.get_branch() No bonds found!")

        # Create a list for the next level
        next_level = [a for a in self.get_neighbors(atom2) if a != atom1]

        # Add the current atom to the path
        if len(path) == 0:
            path.append(atom2)

        # If the depth is 0 or if the next level is empty, return emptiness
        if depth == 0 or len(next_level) == 0:
            return path

        # If the depth is greater than 0, return the next level
        else:
            # Advance to the next level of neighbors
            for nl in next_level:
                # If the neighbor is already in the path, return the path
                if nl in path:
                    return path
                # If the neighbor is not in the path, continue the search
                else:
                    path.append(nl)
                    path = self.get_branch(atom2, nl, depth - 1, path)
            return list(set(path))
    
    def get_graph(self) -> dict:
        """ Method to get the graph of the molecule
        
        Returns
        -------
        graph : dict
            The graph of the molecule as a dictionary"""
        return self.graph
    
    def _find_rings(self,
                   path : list = [],
                   rings : list = []) -> list:
        """ Method to find the rings in the molecule
        
        Detects if there are rings in the molecule, and
        keeps track of them in a list.

        Returns
        -------
        path : list
            The path that has been walked already
        rings : dict
            The rings of the molecule as a dictionary"""
        
        # Check if the path is empty
        if len(path) == 0:
            # If it is, start with the first atom
            path = [0]

            # Create a list for the next level
            next_level = self.get_neighbors(path[-1])
        else:
            # Create a list for the next level (excluding the last atom)
            next_level = []
            for a in self.get_neighbors(path[-1]):
                if a != path[-2]:
                    next_level.append(a)

        # If there's nothing in the next level, return the path without this
        # last atom
        if len(next_level) == 0:
            return path[:-1], rings

        else:
            # The current path has a length of
            until_now = len(path)

            # Advance to the next level of neighbors
            for nl in next_level:
                # If the neighbor is already in the path, return the path
                if nl in path and nl != path[-1]:
                    starting_index = path.index(nl)
                    new_ring = path[starting_index:until_now]
                    new_ring.sort()
                    if new_ring not in rings:
                        rings.append(new_ring)
                # If the neighbor is not in the path, continue the search
                else:
                    path, rings = self._find_rings(path + [nl], rings)
            
            return path[:until_now - 1], rings
    
    def get_rings(self) -> list:
        """ Method to get the rings in the molecule

        This is a much more refined method to find the
        actual rings in the molecule, considering that
        it checks whether a ring is a subset of another.
        This leads to a removal of the potential bigger
        rings.

        Example: A naphtalene molecule has two rings, but
        the combination of both is, topologically, also
        a ring. This method will remove the bigger ring.
        
        Returns
        -------
        rings : list
            The rings of the molecule as a list"""
        
        # Find the rings in the molecule
        _, rings = self._find_rings([], [])

        # Turn the rings into sets
        sets = [set(r) for r in rings]

        # Remove sets that are supersets of other sets
        for s1 in sets:
            for s2 in sets:
                if s1 != s2 and s1.issubset(s2):
                    sets.remove(s2)

        # Small rings only, as lists
        list_rings = [list(s) for s in sets]
        list_rings.sort()

        # Return the rings
        return list_rings

    def shortest_path(self,
                    atom1 : int,
                    atom2 : int,
                    path : list = []) -> list:
        """Method to find shortest path between two atoms
        
        Method to find the path with the least amount of
        connecting bonds between two atoms in the molecule
        
        Parameters
        ----------
        atom1 : int
            The starting atom
        atom2 : int
            The finishing atom
        path : list
            The path that has been walked already

        Returns
        -------
        path : list
            The path with least steps in the molecule"""
        # Check that atom1 and atom2 are not the same
        if atom1 == atom2:
            raise ValueError("MolecularGraph.shortest_path() The two atoms"
                             " should be different!")
        
        # Check that atom1 and atom2 are in the molecule
        if (atom1 not in list(range(len(self.atoms))) or
                atom2 not in list(range(len(self.atoms)))):
            raise ValueError("MolecularGraph.shortest_path() The two atoms"
                             " should be in the molecule!")

        # If no bonds are found, something is wrong
        if len(self.bonds) == 0:
            raise ValueError("MolecularGraph.shortest_path() No bonds found!")

        # Add the current atom to the path
        if len(path) == 0:
            path.append(atom1)

        # Create a list for the next level
        next_level = [a for a in self.get_neighbors(atom1) if a not in path]

        # If there's nothing in the next level, return the path without this
        # last atom
        if len(next_level) == 0:
            return path[:-1]

        else:
            # The current path has a length of
            until_now = len(path)
            new_paths = []

            # Advance to the next level of neighbors
            for nl in next_level:
                # If the neighbor is already in the path, return the path
                if nl in path:
                    continue
                # If the neighbor is the finishing atom, return the path
                elif nl == atom2:
                    new_paths.append(path + [nl])
                # If the neighbor is not in the path, continue the search
                else:
                    temp_path = self.shortest_path(nl, atom2, path + [nl])
                    if temp_path[-1] == atom2:
                        new_paths.append(temp_path)
            
            if len(new_paths) > 0:
                # Sort the paths by the shortest one
                new_paths = sorted(new_paths, key=len)
                # Keep only the paths with the same short length
                shortest = len(new_paths[0])
                new_paths = [p for p in new_paths if len(p) == shortest]
                # Sort the paths by the sum of number of atom
                new_paths = sorted(new_paths, key=sum)
                return new_paths[0]
            else:
                # This path didn't lead to the finishing atom, remove the last
                return path[:until_now - 1]
    
    # TODO:
    # - Add method to find the longest chain in the molecule

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
    bonds : list of list
        A `list` with all the pairs of atoms creating bonds
    angles : list of list
        A `list` with all the trios of atoms creating angles
    dihedrals : list of list
        A `list` with all the quartets of atoms creating dihedrals
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
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.mol_weight = 0.0
        self.charge = 0.0
        self.volume = 0.0
        self.graph = None

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

    def add_atoms(self, *atoms : Atom) -> None:
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
    
    def remove_atoms(self, *atoms : int) -> None:
        """ Method to remove atoms from the molecule

        Removes the specified atom(s) of the Molecule object. It
        checks whether indices are correct.

        Raises
        ------
        TypeError
            If an empty list is provided to remove atoms.
        ValueError
            If there are no atoms to remove.
            If the number(s) for the atom(s) don't match.

        Parameters
        ----------
        *atoms : int
            A `list` with the indices of the atoms in the Molecule
            object.
        """
        # Check if the provided list is empty
        if len(atoms) == 0:
            raise TypeError("Molecule.remove_atoms() No atom indices "
                            "were provided.")
        # Check if there are any atoms to remove
        if self.get_num_atoms() == 0:
            raise ValueError("Molecule.remove_atoms() There are no "
                             "atoms to be removed.")
        # Check if all the provided atom indices are within range
        for w in atoms:
            if (w < 1) or (w > self.get_num_atoms()):
                raise ValueError("Molecule.remove_atoms() The provided "
                                 f"atom index {w} is out of range!")
        
        # Sort the given indices
        atoms = [*atoms]
        atoms.sort()

        # Remove the atoms from the back to the front
        for w in atoms[::-1]:
            self.atoms.pop(w)
    
    def toggle_selection(self) -> None:
        """ Method to activate/deactivate atoms in the molecule

        The selection of the atoms in the molecule will be flipped
        """
        # Go over list of atoms
        for a in self.atoms:

            # Flip the selection of the atoms in the molecule
            if a.flag == True:
                a.flag = False
            else:
                a.flag = True
    
    def select(self) -> None:
        """ Method to select all atoms in the molecule

        All atoms of the molecule will be selected
        """
        # Go over list of atoms
        for a in self.atoms:

            # Select the atoms
            a.flag = True
    
    def deselect(self) -> None:
        """ Method to deselect all atoms in the molecule

        All atoms of the molecule will be deselected
        """
        # Go over list of atoms
        for a in self.atoms:

            # Select the atoms
            a.flag = False
    
    def assign_charges(self, *charges : float) -> None:
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

    def get_mol_weight(self) -> bool:
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
            self.mol_weight += a.mass

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
    
    def _montecarlo_volume(self,
                            pid : int,
                            return_dict : dict,
                            dots : int = 1000) -> None:
        """ Method to evaluate the molecular volume via Monte Carlo

        Parameters
        ----------
        pid : int
            Process ID
        return_dict : dict
            Dictionary to store the results
        dots : int
            Number of dots to generate
        """
        # Get the limits of the molecule
        lims = self.get_limits()
        box_volume = lims['X'][2] * lims['Y'][2] * lims['Z'][2]

        # Random number generator
        rng = np.random.default_rng()
        
        # Generate random coordinates
        x = rng.random(dots)
        y = rng.random(dots)
        z = rng.random(dots)

        hits = 0
        for j in range(dots):

            # Build the dot
            dot = np.array([x[j] * lims['X'][2],
                            y[j] * lims['Y'][2],
                            z[j] * lims['Z'][2]])
            dot += np.array([lims['X'][0], lims['Y'][0], lims['Z'][0]])
            
            # Iterate over all atoms
            for a in self.atoms:

                # Compute distance between the dot and the atom
                distance = np.linalg.norm(dot - a.get_coordinates())

                # If the distance is less than the radius of the atom
                if distance < a.radius:
                    hits += 1
                    break
        
        # Compute the relative volume of the molecule as the
        # ratio of (dots - hits) to the number of dots
        ratio = hits / dots

        return_dict[pid] = box_volume * ratio
    
    @lru_cache(maxsize=1)
    def get_molecular_volume(self,
                             dots : int = 1000,
                             iterations : int = 10) -> float:
        """ Method to get the volume of the molecule

        The method will compute the volume of the molecule using
        a Monte Carlo approach.

        Parameters
        ----------
        dots : int
            Number of dots to be used in the MonteCarlo method
        iterations : int
            Number of iterations to be used in the MonteCarlo method

        Returns
        -------
        self.volume : float
            Volume of the molecule
        """
        # If < 3 iterations are used, the volume is computed in serial
        if iterations > 2:
            # Split the task accross the number of cores

            # Prepare a list of processes
            processes = []
            # Set a process manager to get the results
            manager = Manager()
            # Initialize a dictionary for the results
            return_dict = manager.dict()
            # Add each process to the list
            for i in range(iterations):
                processes.append(Process(target=self._montecarlo_volume,
                                        args=(i, return_dict, dots)))
            # Start the processes
            [t.start() for t in processes]
            # Join the processes
            [t.join() for t in processes]
            # Get the results
            mol_vols = list(return_dict.values())
        else:
            # Initialize a list for the results
            mol_vols = []
            # Compute the volume in serial
            for i in range(iterations):
                mol_vols.append(self._montecarlo_volume(dots))
        self.volume = np.round(np.mean(mol_vols), 3)

        return self.volume
    
    def _set_bonded_atoms(self) -> None:
        """ Method to set the bonded atoms of the molecule to its atoms

        This method will assign the bonded atoms of the molecule
        to each of the atoms
        """
        if self.graph is None:
            self.get_distance_matrix()
               
        # Iterate over all possible bonds
        for ia, pb in self.graph.get_graph().items():
            # Cases of 1, 2 or 3 bonded atoms (i.e. chirality doesn't matter)
            if len(pb) < 4:
                self.atoms[ia].bonded_atoms = list(pb)
            # Chirality could matter
            else:
                # TODO:
                # - Get the atomic geometry by number of neighbors
                # - Decide case based on geometry
                # - Get the symbols for all neighbors
                # - Count the number of each symbol
                # - If symbol = H, F, Cl or Br, and number of symbol = neighbors - 2, add the bonded atoms
                # Get the masses of the bonded atoms
                masses = {}
                for j, ja in enumerate(pb):
                    masses[j] = self.atoms[ja].mass
                if set(masses.values()) != list(masses.values()):
                    pass
                # Sort the masses
                sorted_masses = {}
                for k, v in sorted(masses.items(), key=lambda x: x[1]):
                    sorted_masses[k] = v
                
                

    def get_distance_matrix(self) -> np.ndarray:
        """ Method to get the distances between pairs of atoms

        The method also creates the bonds of the molecule object

        Returns
        -------
        dist_mat : ndarray
            Distance matrix of the molecule
        """
        # Need the number of atoms
        num_atoms = self.get_num_atoms()

        # Fill the distance matrix with zeros
        dist_mat = np.zeros((num_atoms, num_atoms), dtype=np.float64)

        # Temporary bond table
        bonds = []

        # Iterate over all atoms ... twice
        for i, ai in enumerate(self.atoms):
            for j, aj in enumerate(self.atoms):
                # If it's the same atom, the distance is zero
                if i != j:
                    # Compute distance
                    dist_mat[i][j] = self.get_distance(i, j)
                    # Atomic radius 1
                    d1 = ai.radius
                    # Atomic radius 2
                    d2 = aj.radius
                    # Maximal bond distance
                    max_dist = d1 + d2
                    # If the position of both atoms is within bonding distance
                    if dist_mat[i][j] <= max_dist:
                        # Add them to a temporary list
                        temp = sorted([i,j])
                        bonds.append(f"{temp[0]},{temp[1]}")

        # Re-create the bond table for the molecule
        for b in set(bonds):
            self.bonds.append([ int(i) for i in b.split(",") ])
        
        self.graph = MolecularGraph(num_atoms, self.bonds)

        return dist_mat
    
    def get_bonds(self, force : bool = False) -> list:
        """ Method to get the list of bonds in the molecule

        Parameters
        ----------
        force : bool
            Force the recalculation of the list of bonds?

        Returns
        -------
        bonds : list
            List of bonds
        """
        if not force and len(self.bonds) != 0:
            return self.bonds

        # If the bond list hasn't been created,
        # or force is applied, do it first
        self.get_distance_matrix()

        return self.bonds
    
    def get_angles(self, force : bool = False) -> list:
        """ Method to get the list of angles in the molecule

        Parameters
        ----------
        force : bool
            Force the recalculation of the angles?
        
        Returns
        -------
        angles : list
            List of angles
        """

        if not force and len(self.angles) != 0 and len(self.bonds) != 0:
            return self.angles

        if len(self.bonds) == 0:
            self.get_bonds()

        self.angles = []

        # Iterate over all bonded atoms (reference atoms)
        for b1 in self.bonds:
            # Iterate over all bonded atoms (comparison atoms)
            for b2 in self.bonds:
                # Only one of the bonded atoms should be in the other bond
                if not (b1[0] in b2) and (b1[1] in b2):
                    # Produce a list with the atoms in the angle
                    temp_a = b1 + [ b2[ (b2.index(b1[1]) + 1) % 2 ] ]
                    # Check that the angle hasn't been added already
                    if (temp_a not in self.angles and
                            temp_a[::-1] not in self.angles):
                        self.angles.append(temp_a)
                 # Only one of the bonded atoms should be in the other bond
                if (b1[0] in b2) and not (b1[1] in b2):
                    # Produce a list with the atoms in the angle
                    temp_b = [ b2[ (b2.index(b1[0]) + 1) % 2 ] ] + b1
                    # Check that the angle hasn't been added already
                    if (temp_b not in self.angles and
                            temp_b[::-1] not in self.angles):
                        self.angles.append(temp_b)

        return self.angles
    
    def get_dihedrals(self, force : bool = False) -> list:
        """ Method to get the list of dihedrals in the molecule

        Parameters
        ----------
        force : bool
            Force the recalculation of the dihedrals?

        Returns
        -------
        dihedrals : list
            List of dihedrals
        """

        if (not force and 
            len(self.dihedrals) != 0 and 
            len(self.angles) != 0 and 
            len(self.bonds) != 0):
            return self.dihedrals
        
        if len(self.angles) == 0:
            self.get_angles()

        self.dihedrals = []

        # Iterate over all angles (reference atoms)
        for a1 in self.angles:
            # Iterate over all angles (comparison atoms)
            for b1 in self.bonds:
               # The dihedral should be an angle with an additional atom
               # at one of the sides
               if (b1[0] == a1[0] or b1[0] == a1[2]) and b1[1] != a1[1]:
                    # If it's the atom on the left ...
                    if b1[0] == a1[0]:
                       # Produce a list with the atoms in the dihedral
                       temp_a = [ b1[1] ] + a1
                       # Check that the dihedral hasn't been added already
                       if (temp_a not in self.dihedrals and
                           temp_a[::-1] not in self.dihedrals):
                            self.dihedrals.append(temp_a)
                    # If it's the atom on the right ...
                    if b1[0] == a1[2]:
                       # Produce a list with the atoms in the dihedral
                       temp_b = a1 + [ b1[1] ]
                       # Check that the dihedral hasn't been added already
                       if (temp_b not in self.dihedrals and
                           temp_b[::-1] not in self.dihedrals):
                            self.dihedrals.append(temp_b)

        return self.dihedrals

    def move_molecule(self, direction : np.ndarray) -> None:
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
    
    def move_selected_atoms(self, direction : np.ndarray) -> None:
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
                                    center : str = "geom") -> None:
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
                                          center : str = "geom") -> None:
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
                                  atom : int) -> None:
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
                                        atom : int) -> None:
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
                                  angle : float) -> None:
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
                                        angle : float) -> None:
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
        for ia, a in enumerate(atoms):
            # Weight the coordinates by the mass and add them to the center
            atom_coords = np.array([*a[1:4]])
            centro += atom_coords * self.atoms[ia].mass

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

    def read_xyz(self, file_name : str) -> None:
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

    def save_as_xyz(self) -> None:
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
    
    def save_selection_as_xyz(self) -> None:
        """ Save selected atoms in the molecule as an XYZ file

        This method does not return anything, nor it requires
        any parameters.
        """
        # Create a template for the XYZ coordinates
        template = " {s} {x:16.8f} {y:16.8f} {z:16.8f}\n"

        num_atoms = 0
        selected_coords = ""

        # Iterate over atoms
        for a in self.atoms:
            if a.flag:
                num_atoms += 1
                selected_coords += template.format(
                                            s=a.element,
                                            x=a.coords[1],
                                            y=a.coords[2],
                                            z=a.coords[3])
        
        header = f"""{len(num_atoms)}
XYZ file of atom selection from molecule: {self.name} - created by InformalFF
"""

        with open(f"sel_{self.name}.xyz", "w") as xyz:
            xyz.write(header + selected_coords)
    
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

            # From Bohr to pm
            pad_i /= BOHR
            pad_a /= BOHR

            # From pm to Angstrom
            pad_i /= 100
            pad_a /= 100

            # Compute the limits
            lims[q] = [low - pad_i,
                       high + pad_a,
                       high + pad_a - (low - pad_i)]

        return lims
    
    def max_distance_to_center(self, center : str = "geom") -> float:
        """ Method to get the radial distance from the molecule out

        Compute find the distance from the center of the molecule to
        its furthest atom (plus VdW radius).

        Parameters
        ----------
        center : str
            Specify which center of the molecule to use:
            - com  : center of mass
            - atom : atom closest to the center of mass
            - geom : geometric center (coordinates)

        Returns
        -------
        atom : int
            The index of the atom which is the furthest from the
            selected center.
        dist : float
            The distance from the selected center to the furthest
            atom plus its VdW radius.
        """
        # Select the kind of center to be used
        if center == "com":
            mol_center = self.get_center_of_mass()
        elif center == "atom":
            mol_center = self.get_center_atom()[2]
        else:
            mol_center = self.get_center()

        # If there's only one atom in the molecule
        if len(self.atoms) == 1:
            symbol = self.get_coords()[0][0]
            dist = PERIODIC_TABLE.loc[symbol, "AtomicRadius"] / BOHR / 100
            return 0, dist
        # If there's more than one atom
        else:
            dist = 0
            atom = -1
            for i, a in enumerate(self.get_coords()):
                # Compute vector between atoms
                dist_vec = mol_center - np.array(a[1:4])
                # Compute norm of the vector = distance between nuclei
                nuclear_dist = np.linalg.norm(dist_vec)
                # Get the VdW radius of that atom
                vdw_radius = PERIODIC_TABLE.loc[a[0], "AtomicRadius"] / BOHR / 100
                # Combine everything
                temp_dist = nuclear_dist + vdw_radius
                # If this newly computed distance is greater than the last
                # save it (and the atom), otherwise, just ignore it
                if temp_dist > dist:
                    dist = temp_dist
                    atom = i
            
            return atom, dist
    
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
    
    def create_box_grid(self,
                        mesh : float = 0.1,
                        limits : dict = {},
                        padding : float = 0.2) -> list:
        """ Method to create an imaginary grid

        Creating a box-shaped imaginary grid around the molecule,
        considering the VdW radii and the padding.

        Parameters
        ----------
        mesh : float
            The space between points in the grid
        limits : dict
            The lower and upper limits of each side of the box
        padding : float
            Additional space to be left on the sides of the box

        Returns
        -------
        grid : ndarray
            A 3D grid with the position vector for each point
        """
        
         # If the limits were not provided, compute them
        if len(limits) == 0:
            limits = self.get_limits()

        # Place to store the lists to create the grid
        box = {}

        # Create the lists for the grid
        for q in "XYZ":
            temp_low = limits[q][0] - limits[q][2] * padding
            temp_high = limits[q][1] + limits[q][2] * padding
            box[q] = np.linspace(temp_low,
                                 temp_high,
                                 int((temp_high - temp_low) / mesh) + 1)
        
        # Create empty grid
        grid = []
        # Iterate over x coordinate
        for x in box['X']:
            # Iterate over y coordinate
            for y in box['Y']:
                # Iterate over z coordinate
                for z in box['Z']:
                    grid.append([x,y,z])
        
        return grid
    
    def create_sphere_grid(self,
                    r : float = 5,
                    center : str = "geom",
                    dots : int = 1000,
                    delta_r : float = 0.3):
        """ Method to create a 3D spherical grid around a molecule

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
        center : str
            Specify which center of the molecule to use:
            - com  : center of mass
            - atom : atom closest to the center of mass
            - geom : geometric center (coordinates)
        dots : int
            The amount of dots in the grid
        delta_r : float
            The distance between the radius of one shell an the other

        Returns
        -------
        grid : dict
            The X, Y and Z coordinates of all dots in that sphere
        """
        # Select the kind of center to be used
        if center == "com":
            mol_center = self.get_center_of_mass()
        elif center == "atom":
            mol_center = self.get_center_atom()[2]
        else:
            mol_center = self.get_center()

        # Compute the range of shells
        shells = range(1, int(np.ceil(r / delta_r)) + 1)
        if len(shells) < 2:
            raise ValueError('Molecule.sphere_grid() The number of shells '
                             'that can be generated with this radius and '
                             'delta is not enough for a filled sphere to '
                             'be made.')
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
            temp_grid = fibonacci_grid_shell(mol_center,
                                             delta_r * s,
                                             shell_dots[s])
            for q in "XYZ":
                grid[q] = grid[q] + temp_grid[q]

        return grid

    def compute_charge_box_grid(self,
                                charge : float = -1,
                                mesh : float = 0.5,
                                limits : dict = {},
                                padding : float = 0.2) -> tuple:
        """ Method to create an imaginary grid and compute the charge

        Creating an imaginary grid and using each point in space as a
        probe to compute the charge and charge vector.

        Parameters
        ----------
        charge : float
            The probe's charge
        mesh : float
            The space between points in the grid
        limits : dict
            The lower and upper limits of each side of the box
        padding : float
            Additional space to be left on the sides of the box

        Returns
        -------
        grid : ndarray
            A 3D grid with the charge evaluated at each point
        v_field : ndarray
            A 3D grid of vectors pointing towards the given charged
            point.
        """

        raw_grid = self.create_box_grid(mesh=mesh,
                                        limits=limits,
                                        padding=padding)
        
        refined_grid = [g + [charge] for g in raw_grid]
        
        # Compute the list of z coordinates for a given x and y
        # using all available processors
        with Pool() as p:
            output = p.starmap(self.charge_in_field,
                                            refined_grid)
            
        # Open the result into charges and vectors
        x_grid, x_vfield = zip(*output)
        
        # Turn into ndarray
        grid = np.array(x_grid)
        v_field = np.array(x_vfield)

        return grid, v_field

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("This library was not intended as a standalone program.")