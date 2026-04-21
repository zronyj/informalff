import numpy as np
from warnings import warn

# ------------------------------------------------------- #
#                 The Chemical Bond Class                 #
# ------------------------------------------------------- #

class ChemicalBond:
    """ Class to find the bond type between all provided atoms
    
    This class is used to figure which kind of bonds are present in
    a given structure. It uses the element of each atom, the structure's
    bond table and each atom's electron configuration.

    Attributes
    ----------
    atoms : list
        List of atoms
    bonds : list of list
        A list with all the paris of atoms defining the bonds
    """
    def __init__(self,
                 atoms : list,
                 bonds : list,
                 charge : int = 0,
                 verbose : bool = False):
        """
        ChemicalBond constructor method

        Parameters
        ----------
        atoms : list
            List of atoms
        bonds : list
            A list with all the paris of atoms defining the bonds
        charge : int
            The charge of the structure
        verbose : bool
            If True, print messages
        """
        self.__atoms = atoms
        self.__bonds = bonds
        self.__charge = charge
        self.__verbose = verbose
    
    def __generate_bonding_configs(self,
                                   possible_expansions : list,
                                   current_config : list = []) -> list:
        """ Method to generate all possible bond order configurations

        Parameters
        ----------
        possible_expansions : list
            List of possible bond orders
        current_config : list
            Current bond order configuration

        Returns
        -------
        list
            List of all possible bond orders
        """
        if len(possible_expansions) == 0:
            return [current_config]
        else:
            new_configs = []
            # Iterate over all possible expansions
            for pe in possible_expansions[0]:

                # Add the current expansion plus all the other possible
                # expansions
                new_configs += self.__generate_bonding_configs(
                    possible_expansions[1:],
                    current_config + [pe]
                )
            return new_configs
    
    def __nice_bond_orders(self, bond_orders : list) -> list:
        """ Method to get a list of ideally nice bonds

        The idea is to prioritize bond orders that are integers,
        and to de-prioritize bond orders with zero (i.e. bond breakage)

        Parameters
        ----------
        bond_orders : list
            List of bond orders

        Returns
        -------
        list
            List of bonds
        """
        # Initialize the lists
        has_zeros = []
        has_decimals = []
        nice_bos = []

        # Iterate over all bond orders
        for bo in bond_orders:

            # If the bond order contains zero
            if np.any(bo < 1e-2):
                has_zeros.insert(0, bo)

            # If the bond order contains at least one non-integer
            elif np.any([ibo % int(ibo) != 0 for ibo in bo]):
                has_decimals.append(bo)
            
            # If the bond order is nice
            else:
                nice_bos.append(bo)
        
        return nice_bos + has_decimals + has_zeros
    
    def __compute_bond_orders(self,
                              velec : int,
                              mat : np.ndarray,
                              imat : np.ndarray,
                              vvec : np.ndarray) -> tuple:
        """ Method to compute the bond orders

        Parameters
        ----------
        velec : int
            The number of valence electrons
        mat : np.ndarray
            The atom-bond matrix
        imat : np.ndarray
            The pseudo inverse of the atom-bond matrix
        vvec : np.ndarray
            The valence electron vector

        Returns
        -------
        tuple
            Tuple containing the following 2 items:
            - A list of bond orders
            - A boolean indicating if the calculation was successful
        """
        # Compute the bond type vector
        bond_orders = imat @ vvec

        bond_orders = np.round(bond_orders, 4)
        
        # Negative bond orders are not allowed
        if np.any(bond_orders < 0):
            return bond_orders, False
        
        # If the number of bonds is the same as the number of
        # valence electrons halved, then the calculation is
        # successful
        if np.abs(np.sum(bond_orders) - velec / 2) > 1e-2:
            return bond_orders, False

        # Check if the bond type vector is correct (i.e. the
        # multiplication of the atom-bond matrix with the bond
        # order vector is equal to the valence electron vector)
        if not np.all(np.abs(mat @ bond_orders - vvec) < 1e-3):
            return bond_orders, False
        
        # Otherwise, the calculation was successful
        return bond_orders, True
    
    def __clean_bond_orders(self, bond_orders : np.ndarray) -> tuple:
        """ Method to clean the bond orders

        Parameters
        ----------
        bond_orders : np.ndarray
            The bond orders

        Returns
        -------
        tuple
            Tuple containing the following 2 items:
            - A list of bonds
            - A list of bond orders
        """
        # Get the number of bonds
        num_bonds = len(self.__bonds)

        # Initialize the lists
        new_bonds = []
        new_bond_orders = []

        # Iterate over all bonds
        for i in range(num_bonds):

            # If the bond order is greater than 0.1, it's a bond!
            if np.abs(bond_orders[i]) > 0.1:
                new_bonds.append(self.__bonds[i])
                new_bond_orders.append(bond_orders[i])

        return new_bonds, new_bond_orders
    
    def __try_expansions(self,
                         valence_electrons : np.ndarray,
                         lone_pairs : np.ndarray,
                         mat : np.ndarray,
                         imat : np.ndarray) -> np.ndarray:
        
        # Generate the template of all possible expansions
        pre_configs = []
        for lp in lone_pairs:
            if lp > 0:
                pre_configs.append(list(range(lp + 1)))
            elif lp == 0:
                pre_configs.append([0])
            else:
                raise ValueError("ChemicalBond.__try_expansions() "
                                 "Lone pairs cannot be negative.")
        
        # Actually generate all possible expansions
        possible_expansions = self.__generate_bonding_configs(pre_configs)
        
        # Compute the bond orders for all possible expansions
        # De-prioritize bond orders with zero, to avoid bond breakage
        all_bond_orders = []
        for pe in possible_expansions:

            # Classic valence electrons + expansion
            ve = valence_electrons + np.array(pe) * 2

            # Compute the bond orders
            bo, success = self.__compute_bond_orders(np.sum(ve), mat, imat, ve)
            if success:
                all_bond_orders.append(bo)
        
        if len(all_bond_orders) == 0:
            raise ValueError("ChemicalBond.__try_expansions() "
                             "failed to calculate bond types. "
                             "No bond order configurations leading to "
                             "a satisfactory bonding scheme was found. "
                             "Make sure that the provided geometry is "
                             "correct.")
        
        # Prioritize nice bond orders
        all_bond_orders = self.__nice_bond_orders(all_bond_orders)

        # Return the bond orders
        return all_bond_orders
    
    def __ionize(self,
                 charge : int,
                 valence_electrons : np.ndarray,
                 lone_pairs : np.ndarray) -> np.ndarray:
        """ Method to ionize a structure

        Parameters
        ----------
        charge : int
            The charge of the structure
        valence_electrons : np.ndarray
            The valence electrons
        lone_pairs : np.ndarray
            The number of lone pairs

        Returns
        -------
        np.ndarray
            The ionized valence electrons
        """
        # Get atom electronegativities
        electronegativities = []
        for i, atom in enumerate(self.__atoms):
            electronegativities.append(
                (
                    i,
                    atom.electronegativity
                )
            )

        # Sort by electronegativity
        electronegativities.sort(key=lambda s: s[1])
        electropositivities = electronegativities.copy()
        electronegativities.reverse()

        # Backup
        e_negatives = electronegativities.copy()
        e_positives = electropositivities.copy()

        if self.__verbose:
            print("Electronegativities:")
            print([(self.__atoms[en[0]].element, en[1]) for en in e_negatives])
            print("Electropositivities:")
            print([(self.__atoms[ep[0]].element, ep[1]) for ep in e_positives])

        # Which charge do I have?
        if charge > 0: # I'm a cation

            # Loop until I'm neutral: I'm being reduced
            while charge > 0:

                any_change = False
                # Iterate over electropositivities
                for ep in electropositivities:

                    # If there is an atom that is highly electropositive
                    # and has a free lone pair of electrons that can be
                    # used to ionize the system without breaking a bond,
                    # remove the lone pair, add an extra valence electron,
                    # reduce the charge and remove the atom from the list
                    # of electropositivities
                    if lone_pairs[ep[0]] > 0:
                        lone_pairs[ep[0]] -= 1
                        valence_electrons[ep[0]] += 1
                        charge -= 1
                        electropositivities.remove(ep)
                        any_change = True
                        break
                
                # If no change was made
                if charge > 0 and not any_change:

                    # Iterate over electropositives again
                    for ep in e_positives:

                        # If there is an atom that is highly electropositive
                        # and has a free valence electron that can be
                        # used to ionize the system further, even by breaking
                        # a bond, remove the valence electron, reduce the charge
                        # and remove the atom from the list of electropositivities
                        if valence_electrons[ep[0]] > 0:
                            valence_electrons[ep[0]] -= 1
                            charge -= 1
                            e_positives.remove(ep)
                            any_change = True
                            break
                
                # If nothing happened
                if charge > 0 and not any_change:
                    raise ValueError("ChemicalBond.__ionize() Reduction "
                                     "failed to ionize the structure. "
                                     "No ionization scheme leading to "
                                     "a neutral structure was found. "
                                     "Make sure that the provided geometry "
                                     "is correct.")


        elif charge < 0: # I'm an anion

            # Loop until I'm neutral: I'm being oxidized
            while charge < 0:

                any_change = False
                # Iterate over electronegativities
                for en in electronegativities:

                    # If there is an atom that is highly electronegative
                    # and has free space in the valence shell for another
                    # electron that can be used to ionize the system without
                    # breaking a bond, remove a valence electron, add an extra
                    # lone pair, increase the charge and remove the atom from
                    # the list of electronegativities
                    electrons_in_shell = valence_electrons[en[0]] +\
                                         2 * lone_pairs[en[0]]
                    _, _, shell_capacity = self.__atoms[en[0]].get_valence()
                    if electrons_in_shell + 1 < shell_capacity:
                        valence_electrons[en[0]] -= 1
                        lone_pairs[en[0]] += 1
                        charge += 1
                        electronegativities.remove(en)
                        any_change = True
                        break
                
                # If no change was made
                if charge < 0 and not any_change:

                    # Iterate over electronegatives again
                    for en in e_negatives:

                        # If there is an atom that is highly electronegative
                        # and has free space in the valence shell for another
                        # electron that can be used to ionize the system further,
                        # even by breaking a bond, remove a valence electron,
                        # increase the charge and remove the atom from
                        # the list of electronegativities
                        electrons_in_shell = valence_electrons[en[0]] +\
                                             2 * lone_pairs[en[0]]
                        _, _, shell_capacity = self.__atoms[en[0]].get_valence()
                        if electrons_in_shell < shell_capacity:
                            valence_electrons[en[0]] -= 1
                            charge += 1
                            e_negatives.remove(en)
                            any_change = True
                            break
                
                # If nothing happened
                if charge < 0 and not any_change:
                    raise ValueError("ChemicalBond.__ionize() Oxidation "
                                     "failed to ionize the structure. "
                                     "No ionization scheme leading to "
                                     "a neutral structure was found. "
                                     "Make sure that the provided geometry "
                                     "is correct.")

        return valence_electrons, lone_pairs
        

    def get_bond_types(self, allow_multiconfig : bool = False) -> tuple | list:
        """ Method to get the bond type of each pair of atoms

        Parameters
        ----------
        allow_multiconfig : bool
            The method might return multiple configurations, if found

        Returns
        -------
        tuple or list of tuples
            Tuple containing the following 2 items:
            - A list of bond_types
            - A list of bond orders
        
        Raises
        ------
        ValueError
            If the bond type cannot be computed
        """
        # Get the number of atoms and bonds
        num_atoms = len(self.__atoms)
        num_bonds = len(self.__bonds)

        # Create the atom-bond matrix
        atom_bond_matrix = np.zeros((num_atoms, num_bonds))

        # Fill the matrix with ones where an atom takes part in a bond
        for i in range(num_atoms):
            for j in range(num_bonds):
                if i in self.__bonds[j]:
                    atom_bond_matrix[i][j] = 1
            
        if self.__verbose:
            print("Atom-bond matrix:")
            print(atom_bond_matrix)

        # Compute the pseudo inverse
        i_atom_bond_matrix = np.linalg.pinv(atom_bond_matrix)

        # Create the valence electron vector and lone pairs vectors
        valence_electrons = []
        lone_pairs = []

        # Fill the vector with the number of valence electrons
        for i in range(num_atoms):
            ve, lp, sc = self.__atoms[i].get_valence()
            valence_electrons.append(ve)
            lone_pairs.append(lp)

        valence_electrons = np.array(valence_electrons)
        lone_pairs = np.array(lone_pairs)

        # If the structure has a non-zero charge, ionize it
        if self.__charge != 0:
            valence_electrons, lone_pairs = self.__ionize(self.__charge,
                                                          valence_electrons,
                                                          lone_pairs)

        if self.__verbose:
            print("Valence electron vector:")
            print(valence_electrons)
            print("Lone pairs vector:")
            print(lone_pairs)

        # Compute the bond type vector
        bond_orders, success = self.__compute_bond_orders(
                                                np.sum(valence_electrons),
                                                atom_bond_matrix,
                                                i_atom_bond_matrix,
                                                valence_electrons)
        
        # If the number of bonds is the same as the number of
        # valence electrons halved, then the calculation is
        # successful and the bond type vector can be returned
        if success:
            new_bonds, new_bond_orders = self.__clean_bond_orders(bond_orders)
            return new_bonds, new_bond_orders
    
        else:
            if self.__verbose:
                warn("ChemicalBond.get_bond_types() failed to "
                     "calculate bond types. "
                     "Expanding the valence electron vector to "
                     "find possible bond orders.")

            # If the number of bonds is not the same as the number
            # of valence electrons halved, then try expanding the
            # valence electron vector
            if np.sum(lone_pairs) == 0:
                raise ValueError(
                            "ChemicalBond.get_bond_types() "
                            "failed to calculate bond types. "
                            "The number of valence electrons is "
                            "not the same as the number of "
                            "bonds, and there are no lone "
                            "pairs to expand the valence.")
            
            all_bond_orders = self.__try_expansions(valence_electrons,
                                                    lone_pairs,
                                                    atom_bond_matrix,
                                                    i_atom_bond_matrix)
            
            if len(all_bond_orders) == 1:
                bond_orders = all_bond_orders[0]
            else:
                if allow_multiconfig:
                    multi = []
                    for bo in all_bond_orders:
                        new_bonds, new_bond_orders = self.__clean_bond_orders(bo)
                        multi.append((new_bonds, new_bond_orders))
                    return multi
                
                else:
                    if self.__verbose:
                        warn("ChemicalBond.get_bond_types() "
                             "Multiple bond order configurations leading to "
                             "a satisfactory bonding scheme were found. "
                             "Returning the first one.")
                    bond_orders = all_bond_orders[0]

            new_bonds, new_bond_orders = self.__clean_bond_orders(bond_orders)
            return new_bonds, new_bond_orders
            