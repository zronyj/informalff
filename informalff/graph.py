import warnings                                    # To throw warnings instead of raising errors

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

        if atoms < 0:
            raise ValueError("MolecularGraph.__init__() The number of atoms"
                             " should be positive!")

        pre_bonds = []
        for b in bonds:
            pre_bonds += b
        
        pre_bonds = set(pre_bonds)
        if len(pre_bonds) != atoms:
            raise ValueError("MolecularGraph.__init__() The number of atoms"
                             " and the atoms in the bonds do not match!")

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
            self.graph[a] = set(sorted(self.graph[a]))

    def get_neighbors(self,
                      atom : int,
                      depth : int = 1) -> list:
        """ Method to get the neighbors of a given atom
        
        Parameters
        ----------
        atom : int
            The atom
        depth : int
            How many next neighbors to look for
        
        Returns
        -------
        neighbors : list
            The neighbors of the atom"""

        if depth == 1:
            # Return the neighbors
            return list(self.graph[atom])
        else:
            # Initialize the neighbor list
            neighbors = []

            # Iterate over all neighbors
            for n in self.graph[atom]:

                # Get the neighbors of the neighbor
                n_ngbrs = self.get_neighbors(n, depth - 1)

                # Prepare a filtered version of the neighbors
                # of the neighbor
                f_neigbors = []

                # Iterate over all neighbors of the neighbor
                for nn in n_ngbrs:

                    # If it's an integer, check if it's not the source atom
                    if isinstance(nn, int):
                        if nn != atom:
                            f_neigbors.append(nn)
                    # If it's a dictionary, check if it doesn't contain
                    # the source atom
                    else:
                        if atom not in list(nn.keys()):
                            f_neigbors.append(nn)

                # Add the filtered neighbors to the list
                if len(f_neigbors) > 0:
                    neighbors.append({n: f_neigbors})
                else:
                    neighbors.append(n)
                
            return neighbors
    
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
    
    def _follow_bonds(self,
                     atom : int,
                     path : list) -> list:
        """ Method to follow the bonds in the molecule

        This method collects a list of atoms which are
        connected to each other in the molecule.

        Parameters
        ----------
        atom : int
            The atom to be used as starting point
        path : list
            The atoms from the network that have been
            seen already

        Returns
        -------
        path : list
            A list of the atoms that are bonded together
            as a molecule."""
        if len(self.bonds) == 0:
            warnings.warn("MolecularGraph._follow_bonds() No bonds were "
                          "provided! Just the atom will be returned.")
            return [atom]

        # Add the atom to the path if it isn't there yet
        if atom not in path:
            path.append(atom)

        # Get the atom's neighbors
        neighbors = self.get_neighbors(atom)
        
        if (len(neighbors) == 0) or \
            (len(neighbors) == 1 and neighbors[0] in path):
            return path
        
        next_stop = []
        # Add all neighbors if not there yet
        for item in neighbors:
            # Check that the atom hasn't been added to the path
            if item not in path:
                # If so, add the atom to the path
                path.append(item)
                next_stop.append(item)
        
        # Loop over the new neighbors
        if len(next_stop) != 0:
            for item in next_stop:
                path = self._follow_bonds(item, path)

        path.sort()
        return path
                            
    def get_connectivity(self) -> list:
        """ Method to get the connectivity of the molecule

        This means, it will return a list of lists of atoms
        which are connected to each other. If the connecitivity
        list has more than one element, then it's not a single
        Molecule object.However, if it has one element, it may be
        a Molecule, or an Atom.

        Returns
        -------
        connectivity : list
            Connectivity of the molecule"""
        connectivity = []
        reference_pool = []

        for atom in self.atoms:

            # If the atom has been included as a substructure
            # before, just move on
            if atom in reference_pool:
                continue

            # Get the network of the current atom
            mol_net = self._follow_bonds(atom, [])

            # Check the new network
            for n in mol_net:
                if n in reference_pool:
                    continue
            
            connectivity.append(mol_net)
            reference_pool += mol_net
    
        return connectivity
