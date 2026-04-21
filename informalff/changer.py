import copy
import numpy as np
from abc import ABC, abstractmethod # To be able to create several drivers
from .molecule import Molecule
from .collection import Collection
from scipy.spatial.transform import Rotation as R

class Changer(ABC):
    """ Abstract class to change the sub_structure
    
    This is a base class to make a change in a molecule or
    collection. The idea is to make the geometrical modifications
    such that, after energy evaluation, a Force Field can be
    parametrized.
    """

    @abstractmethod
    def change(self,
               sub_structure : Molecule | Collection,
               magnitude : float) -> Molecule | Collection:
        """ Abstract method to change the molecule or collection
    
        This method must be implemented in the derived classes to
        perform the actual change to the molecule or collection.

        Parameters
        ----------
        sub_structure : Molecule | Collection
            The molecule or collection to be changed
        magnitude : float
            The magnitude of the change
        
        Returns
        -------
        changed_molecule : Molecule | Collection
            The modified molecule or collection
        """
        pass

class Stretch(Changer):
    """ Class to represent a Stretch
    
    This class is used to represent the stretch of a bond; it
    does so by moving the atoms that make up the bond.

    Attributes
    ----------
    molecule_key : str
        The key of the molecule to be stretched
    bond_index : int
        The index of the bond to be stretched
    atom1_idx : int
        The index of the first atom in the bond
    atom2_idx : int
        The index of the second atom in the bond
    branch1_idx : list
        The indices of the atoms in the first branch
    branch2_idx : list
        The indices of the atoms in the second branch
    exclusion : list
        The indices of the atoms in the ring that are not to be moved
    """

    def __init__(self,
                 sub_structure : Molecule | Collection,
                 molecule_key : str,
                 bond_index : int) -> None:
        """ Constructor for Stretch class
        
        This is the constructor for the Stretch class. The idea
        is not to keep the Molecule or Collection object as an
        attribute. This way, the stretch can be prepared, and
        only carried out when needed, without making this object
        too heavy.
        
        Parameters
        ----------
        sub_structure : Molecule | Collection
            The molecule or collection to be changed
        molecule_key : str
            The key of the molecule to be stretched
        bond_index : int
            The index of the bond to be stretched
        """
        self.molecule_key = molecule_key
        self.bond_index = bond_index

        if isinstance(sub_structure, Molecule):
            subs = sub_structure
        elif isinstance(sub_structure, Collection):
            subs = sub_structure.molecules[molecule_key]
        else:
            raise TypeError("Stretch() The structure must be "
                            "a Molecule or a Collection.")

        # Get the indices of the atoms
        self.atom1_idx = subs.bonds[bond_index][0]
        self.atom2_idx = subs.bonds[bond_index][1]

        # Check that the atoms are not in the same ring(s)
        rings = subs.graph.get_rings()

        self.exclusion = []

        for ring in rings:
            if self.atom1_idx in ring and self.atom2_idx in ring:
                self.exclusion = self.exclusion + ring
                self.exclusion.remove(self.atom1_idx)
                self.exclusion.remove(self.atom2_idx)

        # Get indices of the branches
        self.branch1_idx = (subs
                            .graph
                            .get_branch(self.atom2_idx,
                            self.atom1_idx,
                            subs.get_num_atoms(),
                            []))
        self.branch2_idx = (subs
                            .graph
                            .get_branch(self.atom1_idx,
                            self.atom2_idx,
                            subs.get_num_atoms(),
                            []))

    def change(self,
               sub_structure : Molecule | Collection,
               magnitude : float) -> Molecule | Collection:
        """
        Method to perform the actual change to the molecule or collection
        
        This method moves the atoms in the molecule or collection
        according to the magnitude of the stretch.
        
        Parameters
        ----------
        sub_structure : Molecule | Collection
            The molecule or collection to be changed
        magnitude : float
            The magnitude of the stretch
        
        Returns
        -------
        subs : Molecule | Collection
            The modified molecule or collection
        """
        if isinstance(sub_structure, Molecule):
            subs = copy.deepcopy(sub_structure)

            # Get the coordinates of the atoms
            left_atom = subs.atoms[self.atom1_idx].coordinates
            right_atom = subs.atoms[self.atom2_idx].coordinates

            # Get the vectors
            left_dir = left_atom - right_atom
            right_dir = right_atom - left_atom

            left_dir /= np.linalg.norm(left_dir)
            right_dir /= np.linalg.norm(right_dir)

            # Move the atoms
            for atom_idx in self.branch1_idx:
                if atom_idx not in self.exclusion:
                    subs.atoms[atom_idx].flag = True
            
            subs.move_selected_atoms(left_dir * magnitude * 0.5)
            subs.deselect()

            for atom_idx in self.branch2_idx:
                if atom_idx not in self.exclusion:
                    subs.atoms[atom_idx].flag = True

            subs.move_selected_atoms(right_dir * magnitude * 0.5)
            subs.deselect()

            return subs
        
        elif isinstance(sub_structure, Collection):
            subs = copy.deepcopy(sub_structure)

            # Get the coordinates of the atoms
            left_atom = (subs
                        .molecules[self.molecule_key]
                        .atoms[self.atom1_idx]
                        .coordinates)
            right_atom = (subs
                        .molecules[self.molecule_key]
                        .atoms[self.atom2_idx]
                        .coordinates)

            # Get the vectors
            left_dir = left_atom - right_atom
            right_dir = right_atom - left_atom

            left_dir /= np.linalg.norm(left_dir)
            right_dir /= np.linalg.norm(right_dir)

            # Move the atoms
            for atom_idx in self.branch1_idx:
                if atom_idx not in self.exclusion:
                    (subs
                     .molecules[self.molecule_key]
                     .atoms[atom_idx]
                     .flag) = True
            
            (subs
             .molecules[self.molecule_key]
             .move_selected_atoms(self.left_dir * magnitude * 0.5))
            subs.molecules[self.molecule_key].deselect()

            for atom_idx in self.branch2_idx:
                if atom_idx not in self.exclusion:
                    (subs
                     .molecules[self.molecule_key]
                     .atoms[atom_idx]
                     .flag) = True

            (subs
             .molecules[self.molecule_key]
             .move_selected_atoms(self.right_dir * magnitude * 0.5))
            subs.molecules[self.molecule_key].deselect()

            return subs
        else:
            raise TypeError("Stretch.change() The structure must be "
                            "a Molecule or a Collection.")

class Bend(Changer):
    """ Class to represent a Bend
    
    This class is used to represent the bend of an angle; it
    does so by moving the atoms that make up the angle.

    Attributes
    ----------
    molecule_key : str
        The key of the molecule to be bent
    angle_index : int
        The index of the angle to be bent
    atom1_idx : int
        The index of the first atom in the angle
    atom2_idx : int
        The index of the second atom in the angle
    atom3_idx : int
        The index of the third atom in the angle
    branch1_idx : list
        The indices of the atoms in the first branch
    branch2_idx : list
        The indices of the atoms in the second branch
    exclusion : list
        The indices of the atoms in the ring that are not to be moved
    """

    def __init__(self,
                 sub_structure : Molecule | Collection,
                 molecule_key : str,
                 angle_index : int) -> None:
        """ Constructor for Bend class
        
        This is the constructor for the Bend class. The idea
        is not to keep the Molecule or Collection object as an
        attribute. This way, the bend can be prepared, and
        only carried out when needed, without making this object
        too heavy.
        
        Parameters
        ----------
        sub_structure : Molecule | Collection
            The molecule or collection to be changed
        molecule_key : str
            The key of the molecule to be bent
        angle_index : int
            The index of the angle to be bent
        """
        self.molecule_key = molecule_key
        self.angle_index = angle_index

        if isinstance(sub_structure, Molecule):
            subs = sub_structure
        elif isinstance(sub_structure, Collection):
            subs = sub_structure.molecules[molecule_key]
        else:
            raise TypeError("Bend() The structure must be "
                            "a Molecule or a Collection.")

        # Get the indices of the atoms
        subs.get_angles()
        self.atom1_idx = subs.angles[angle_index][0]
        self.atom2_idx = subs.angles[angle_index][1]
        self.atom3_idx = subs.angles[angle_index][2]

        # Check that the atoms are not in the same ring(s)
        rings = subs.graph.get_rings()

        self.exclusion = []

        for ring in rings:
            if self.atom1_idx in ring and self.atom3_idx in ring:
                self.exclusion = self.exclusion + ring
                self.exclusion.remove(self.atom1_idx)
                self.exclusion.remove(self.atom3_idx)

        # Get indices of the branches
        self.branch1_idx = (subs
                            .graph
                            .get_branch(self.atom2_idx,
                            self.atom1_idx,
                            subs.get_num_atoms(),
                            []))
        self.branch2_idx = (subs
                            .graph
                            .get_branch(self.atom2_idx,
                            self.atom3_idx,
                            subs.get_num_atoms(),
                            []))

    def change(self,
               sub_structure : Molecule | Collection,
               magnitude : float) -> Molecule | Collection:
        """
        Method to perform the actual change to the molecule or collection
        
        This method moves the atoms in the molecule or collection
        according to the magnitude of the bend.
        
        Parameters
        ----------
        sub_structure : Molecule | Collection
            The molecule or collection to be changed
        magnitude : float
            The magnitude of the stretch
        
        Returns
        -------
        subs : Molecule | Collection
            The modified molecule or collection
        """
        if isinstance(sub_structure, Molecule):
            subs = copy.deepcopy(sub_structure)

            # Get the coordinates of the atoms
            left_atom = subs.atoms[self.atom1_idx].coordinates
            center_atom = subs.atoms[self.atom2_idx].coordinates
            right_atom = subs.atoms[self.atom3_idx].coordinates

            # Get the vectors
            left_dir = left_atom - center_atom
            right_dir = right_atom - center_atom
            left_dir /= np.linalg.norm(left_dir)
            right_dir /= np.linalg.norm(right_dir)

            # Rotation axis
            axis = np.cross(left_dir, right_dir)
            axis /= np.linalg.norm(axis)

            # Move the atoms
            for atom_idx in self.branch1_idx:
                if atom_idx not in self.exclusion:
                    subs.atoms[atom_idx].flag = True
            
            subs.rotate_selected_atoms_over_atom_axis(
                axis,
                magnitude * (-0.5),
                self.atom2_idx
            )
            subs.deselect()

            for atom_idx in self.branch2_idx:
                if atom_idx not in self.exclusion:
                    subs.atoms[atom_idx].flag = True

            subs.rotate_selected_atoms_over_atom_axis(
                axis,
                magnitude * 0.5,
                self.atom2_idx
            )
            subs.deselect()

            return subs
        
        elif isinstance(sub_structure, Collection):
            subs = copy.deepcopy(sub_structure)

            # Get the coordinates of the atoms
            left_atom = (subs
                        .molecules[self.molecule_key]
                        .atoms[self.atom1_idx]
                        .coordinates)
            center_atom = (subs
                        .molecules[self.molecule_key]
                        .atoms[self.atom2_idx]
                        .coordinates)
            right_atom = (subs
                        .molecules[self.molecule_key]
                        .atoms[self.atom3_idx]
                        .coordinates)

            # Get the vectors
            left_dir = left_atom - center_atom
            right_dir = right_atom - center_atom
            left_dir /= np.linalg.norm(left_dir)
            right_dir /= np.linalg.norm(right_dir)

            # Rotation axis
            axis = np.cross(left_dir, right_dir)
            axis /= np.linalg.norm(axis)

            # Move the atoms
            for atom_idx in self.branch1_idx:
                if atom_idx not in self.exclusion:
                    (subs
                    .molecules[self.molecule_key]
                    .atoms[atom_idx]
                    .flag) = True
            
            (subs
            .molecules[self.molecule_key]
            .rotate_selected_atoms_over_atom_axis(
                axis,
                magnitude * (-0.5),
                self.atom2_idx
            ))
            subs.molecules[self.molecule_key].deselect()

            for atom_idx in self.branch2_idx:
                if atom_idx not in self.exclusion:
                    (subs
                    .molecules[self.molecule_key]
                    .atoms[atom_idx]
                    .flag) = True

            (subs
            .molecules[self.molecule_key]
            .rotate_selected_atoms_over_atom_axis(
                axis,
                magnitude * 0.5,
                self.atom2_idx
            ))
            subs.molecules[self.molecule_key].deselect()

            return subs
        else:
            raise TypeError("Stretch.change() The structure must be "
                            "a Molecule or a Collection.")

class Torsion(Changer):
    """ Class to represent a Torsion
    
    This class is used to represent the torsion of a dihedral;
    it does so by moving the atoms that make up the dihedral.

    Attributes
    ----------
    molecule_key : str
        The key of the molecule to be twisted
    dihedral_index : int
        The index of the dihedral to be twisted
    atom1_idx : int
        The index of the first atom in the dihedral
    atom2_idx : int
        The index of the second atom in the dihedral
    atom3_idx : int
        The index of the third atom in the dihedral
    atom4_idx : int
        The index of the fourth atom in the dihedral
    branch1_idx : list
        The indices of the atoms in the first branch
    branch2_idx : list
        The indices of the atoms in the second branch
    exclusion : list
        The indices of the atoms in the ring that are not to be moved
    """

    def __init__(self,
                 sub_structure : Molecule | Collection,
                 molecule_key : str,
                 dihedral_index : int) -> None:
        """ Constructor for Torsion class
        This is the constructor for the Torsion class. The idea
        is not to keep the Molecule or Collection object as an
        attribute. This way, the torsion can be prepared, and
        only carried out when needed, without making this object
        too heavy.
        Parameters
        ----------
        sub_structure : Molecule | Collection
            The molecule or collection to be changed
        molecule_key : str
            The key of the molecule to be twisted
        dihedral_index : int
            The index of the dihedral to be twisted
        """
        self.molecule_key = molecule_key
        self.dihedral_index = dihedral_index
        if isinstance(sub_structure, Molecule):
            subs = sub_structure
        elif isinstance(sub_structure, Collection):
            subs = sub_structure.molecules[molecule_key]
        else:
            raise TypeError("Torsion() The structure must be "
                            "a Molecule or a Collection.")
        # Get the indices of the atoms
        subs.get_dihedrals()
        self.atom1_idx = subs.dihedrals[dihedral_index][0]
        self.atom2_idx = subs.dihedrals[dihedral_index][1]
        self.atom3_idx = subs.dihedrals[dihedral_index][2]
        self.atom4_idx = subs.dihedrals[dihedral_index][3]

        # Check that the atoms are not in the same ring(s)
        rings = subs.graph.get_rings()
        self.exclusion = []
        for ring in rings:
            if (self.atom2_idx in ring and
                self.atom3_idx in ring):
                self.exclusion = self.exclusion + ring
                self.exclusion.remove(self.atom2_idx)
                self.exclusion.remove(self.atom3_idx)

class ChangeFinder:
    """ Class to find the main sites for moving the molecule

    This class holds a copy of the Molecule or Collection object
    and extracts information of the bonds, angles and dihedrals
    in it. The objective is to prepare the coordinates for the
    scan.
    """

    def __init__(self, sub_structure : Molecule | Collection):
        """ Constructor for Change Finder class

        When initialized, this class will extract the information
        from the molecule or collection. This information will be
        used to find the main sites to move the molecule or
        collection.

        Parameters
        ----------
        sub_structure : Molecule | Collection
            The molecule or collection to be changed
        """
        self.sub_structure = sub_structure
        self.locations = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.distance_matrix = {}
        self.mol_col = None

        # Get the main sites to move the molecule
        if isinstance(self.sub_structure, Molecule):
            mol_name = self.sub_structure.name
            self.distance_matrix[mol_name] = self.sub_structure.get_distance_matrix()
            self.bonds[mol_name] = self.sub_structure.get_bonds()
            self.angles[mol_name] = self.sub_structure.get_angles()
            self.dihedrals[mol_name] = self.sub_structure.get_dihedrals()
            self.mol_col = True

        # Get the main sites to move the collection
        elif isinstance(self.sub_structure, Collection):
            self.mol_col = False

            for n, m in self.sub_structure.molecules.items():
                self.locations[n] = m.get_center_of_mass()
                self.distance_matrix[n] = m.get_distance_matrix()
                self.bonds[n] = m.get_bonds()
                self.angles[n] = m.get_angles()
                self.dihedrals[n] = m.get_dihedrals()

        else:
            raise TypeError("The structure must be a Molecule or a Collection")

    def stretch_bonds(self) -> list:

        # Initialize the list
        changes = []
        
        # Iterate over all molecules ...
        for mol, bonds in self.bonds.items():

            # ... and all bonds
            for ib, b in enumerate(bonds):
                # Get the length of the bond
                length = self.distance_matrix[mol][b[0]][b[1]]

                # If the bond is bigger than 2, it's not yet a bond
                if length > 2:
                    continue
                
                # Initialize the changer
                changes.append(Stretch(self.sub_structure, mol, ib))

        return changes


    def bend_angles(self) -> list:
        
        # Initialize the list
        changes = []
        
        # Iterate over all molecules ...
        for mol, angles in self.angles.items():

            # ... and all angles
            for ia, a in enumerate(angles):
                
                # Initialize the changer
                changes.append(Bend(self.sub_structure, mol, ia))

        return changes

    def twist_dihedrals(self) -> list:
        pass