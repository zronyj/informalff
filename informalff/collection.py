import numpy as np                   # To do basic scientific computing
import scipy.constants as cts        # Universal constants
from copy import deepcopy            # To copy objects
from warnings import warn            # To throw warnings instead of raising errors
from multiprocessing import Pool     # To parallelize jobs
from functools import lru_cache      # To cache functions
from scipy.special import gamma      # To compute the gamma function
from scipy.optimize import curve_fit # To fit the Subbotin function

from .elements import PTE
from .molecule import Molecule

def _subbotin(x, alpha, sigma, mu):
    """ Subbotin function

    This distribution will be used to model the mass
    distribution of the molecules in the collection when
    they are in a cube.

    Parameters
    ----------
    x : float
        The x coordinate
    alpha : float
        A parameter to change the height of the function
    sigma : float
        The sigma parameter
    mu : float
        The mean of the data

    Returns
    -------
    value : float
        The value of the subbotin function
    """
    beta = 30
    preExp = alpha * beta / (2 * sigma * gamma(1/beta))
    exp = np.exp(-((x - mu)/sigma)**beta)
    return preExp * exp

# ------------------------------------------------------- #
#                  The Collection Class                   #
# ------------------------------------------------------- #
class Collection(object):
    """ Class to represent a molecular Collection

    This class is used to represent a simple molecular
    systems by considering each molecule with its name
    in a dictionary, and adding several methods for the
    analysis of the collection.

    Attributes
    ----------
    name : str
        A name for the collection (can be anything you choose)
    molecules : dict of Molecule
        A `dict` with all the Molecule objects of the collection
    __nmols : int
        The number of molecules in the collection
    __natoms : int
        The number of atoms in the collection
    """

    def __init__(self, name='collection'):
        """ Collection constructor method

        This is the method to construct the Collection object

        Parameters
        ----------
        name : str, optional
            A name for the collection (can be anything you choose)
        """
        self.name = name
        self.molecules = {}
        self.__nmols = 0
        self.__natoms = 0
        self.__ref_atoms = []
    
    def __str__(self):
        """ Method to represent a Collection as a string

        This method builds a string with the information
        of the Collection object. Said string will be displayed
        as a ticket whenever someone prints this object.

        Returns
        -------
            text : str
                A general description of the Collection, its molecules,
                atoms and dimensions.
        """
        width = 40
        dline = "=" * width
        sline = "-" * width

        content = f"\n{dline}\n"
        content += f"     Molecular Collection\n{self.name:^{width}}\n"
        content += sline + "\n"
        content += f" Total molecules: {self.__nmols}\n"
        content += f" Total atoms: {self.__natoms}\n\n"
        content += "        Molecules\n" + sline + "\n"
        if self.__nmols > 0:
            frags = [m[:3] for m in self.molecules.keys()]
            ufrags = set(frags)
            for f in ufrags:
                content += f"> {f}:\n"
                content += f"    Number of molecules: {frags.count(f)}\n"
                temp = None
                for m in self.molecules.keys():
                    if f in m:
                        temp = m
                        break
                if temp is not None:
                    content += ("    Atoms per molecule: "
                                f"{self.molecules[temp].num_atoms()}\n")
                    content += ("    Formula: "
                                f"{self.molecules[temp].get_formula()}\n")
            content += "\n         Limits\n" + sline + "\n"
            lims = self.get_limits()
            content += "     Lower    Upper  Side\n"
            content += (f"X:{lims['x'][0]:8.3f} {lims['x'][1]:8.3f}"
                        f" {lims['x'][2]:5.2f}\n")
            content += (f"Y:{lims['y'][0]:8.3f} {lims['y'][1]:8.3f}"
                        f" {lims['y'][2]:5.2f}\n")
            content += (f"Z:{lims['z'][0]:8.3f} {lims['z'][1]:8.3f}"
                        f" {lims['z'][2]:5.2f}\n")
        content += "\n            Density           \n"
        content += sline + "\n"
        content += f"    {self.get_density():>8.4f} g/cm^3      \n"
        content += f"{dline}\n"
        return content
    
    def __remap(self) -> None:
        """ Method to keep a reference of all atoms in the collection
        """

        # If no molecules are present, just clear the list
        if len(self.molecules) == 0:
            self.__ref_atoms = []
        else:
            # Else, save a key, index pair to locate the atoms
            self.__ref_atoms = []
            for mk, mv in self.molecules.items():
                for ia, a in enumerate(mv.atoms):
                    self.__ref_atoms.append((mk, ia))

    def __getitem__(self, idx : int | slice):
        """ Retrieve an atom's information based on index.

        Parameters
        ----------
        idx : int or slice
            - If an integer, it retrieves the atom at that index.
            - If a slice, it retrieves a list of atoms defined by the slice.
        
        Returns
        -------
        list or list of lists
            - If an integer is provided, it returns a list containing
              the atom's symbol, and Cartesian coordinates.
            - If a slice is provided, it returns a list of lists, each
              containing the symbol, and Cartesian coordinates of
              the atoms defined by the slice.
        
        Raises
        ------
        TypeError
            If the provided argument is not an integer or slice.
        """
        if isinstance(idx, int):
            ref_code = self.__ref_atoms[idx]
            tmp_atom = self.molecules[ref_code[0]].atoms[ref_code[1]]
            return [
                tmp_atom.element,
                tmp_atom.coordinates
            ]
        elif isinstance(idx, slice):
            piece = []
            i = idx.start if idx.start != None else 0
            o = idx.stop if idx.stop != None else len(self.__ref_atoms)
            e = idx.step if idx.step != None else 1
            for j in range(i, o, e):
                ref_code = self.__ref_atoms[j]
                tmp_atom = self.molecules[ref_code[0]].atoms[ref_code[1]]
                piece.append(
                    [
                        tmp_atom.element,
                        tmp_atom.coordinates
                    ]
                )
            return piece
        else:
            raise TypeError(
                    f"Collection.__getitem__() The argument {idx} is not an "
                    "integer or a slice object."
            )
    
    def __setitem__(self, idx : int | slice , data : list):
        """ Set an atom's information based on index or slice.

        Parameters
        ----------
        idx : int or slice
            - If an integer, it sets the atom at that index.
            - If a slice, it sets a list of atoms defined by the slice.
        data : list
            - If an integer is provided, data should be a list containing
              the atom's symbol, and Cartesian coordinates.
            - If a slice is provided, data should be a list of lists, each
              containing the symbol, and Cartesian coordinates of
              the atoms to be set defined by the slice.
        
        Raises
        ------
        TypeError
            If the provided arguments are not of the expected types.
        """
        if not isinstance(data, list):
            raise TypeError(
                    f"Collection.__setitem__() Expected a list, got {type(data)}"
            )
        if len(data) == 0:
            raise ValueError(
                "Collection.__setitem__() The provided list is empty"
            )
        if isinstance(idx, int):
            if len(data) != 2:
                raise ValueError(
                        "Collection.__setitem__() Expected list with the "
                        f"symbol and the 3D coordinates, got {data}"
                )
            if not isinstance(data[0], str) or \
                not isinstance(data[1], np.ndarray):
                raise TypeError(
                        f"Collection.__setitem__() The item "
                        f"{data} is not shaped as a list of: str "
                        "and np.ndarray."
                )
            if len(data[1]) != 3:
                raise ValueError(
                        "Collection.__setitem__() Expected a 3D vector, "
                        f"got an array with shape {data[1].shape}"
                )
            ref_code = self.__ref_atoms[idx]
            self.molecules[ref_code[0]].atoms[ref_code[1]].element = data[0]
            self.molecules[ref_code[0]].atoms[ref_code[1]].coordinates = data[1]
        elif isinstance(idx, slice):
            i = idx.start if idx.start != None else 0
            o = idx.stop if idx.stop != None else len(self.__ref_atoms)
            e = idx.step if idx.step != None else 1

            for jdx, op in enumerate(range(i, o, e)):
                if not isinstance(data[jdx], list):
                    raise TypeError(
                            "Collection.__setitem__() Expected a list, got "
                            f"{type(data[jdx])}"
                    )
                if len(data[jdx]) != 2:
                    raise ValueError(
                            "Collection.__setitem__() Expected list with the "
                            f"symbol and the 3D coordinates, got {data}"
                    )
                if not isinstance(data[jdx][0], str) or \
                    not isinstance(data[jdx][1], np.ndarray):
                    raise TypeError(
                            f"Collection.__setitem__() The item {jdx}: "
                            f"{data[jdx]} is not shaped as a list of: str "
                            "and np.ndarray."
                    )
                if len(data[jdx][1]) != 3:
                    raise ValueError(
                            "Collection.__setitem__() Expected a 3D vector, "
                            f"got an array with shape {data[jdx][1].shape}"
                    )
            for jdx, op in enumerate(range(i, o, e)):
                ref_code = self.__ref_atoms[op]
                self.molecules[ref_code[0]].atoms[ref_code[1]].element = data[jdx][0]
                self.molecules[ref_code[0]].atoms[ref_code[1]].coordinates = data[jdx][1]
        else:
            raise TypeError(
                    f"Collection.__setitem__() The argument {idx} is not an "
                    "integer or a slice object."
            )
    
    def __len__(self):
        """ Method to get the number of atoms in the collection

        Returns
        -------
        int
            The number of atoms in the collection.
        """
        return self.__natoms

    def add_molecule(self, idm : str, mol : Molecule) -> bool:
        """ Method to add a molecule to the collection

        Adds the specified molecule to the Collection object. It
        checks whether the object is actually an instance of
        Molecule.

        Raises
        ------
        TypeError
            If the added object is NOT an instance of Molecule.
        ValueError
            If the name (ID) of the molecule already exists.

        Parameters
        ----------
        idm : str
            The name of the molecule in the collection.
        mol : Molecule
            A Molecule object to be added to the Collection object.

        Returns
        -------
        bool
            True if everything works out.
        """

        # Check that the object is actually an Molecule instance
        if not isinstance(mol, Molecule):
            raise TypeError(("Collection.add_molecule() The added object is "
                            "not an instance of Molecule."))
        
        if idm in self.molecules.keys():
            raise ValueError("Collection.add_molecule() The name provided "
                             "for this molecule already exists.")
        
        # Add mols to the molecule
        self.molecules[idm] = mol

        # Increment the number of molecules and atoms
        self.__nmols += 1
        self.__natoms += mol.num_atoms()

        # Update the atoms reference map
        self.__remap()

        return True

    def remove_molecule(self, idm : str) -> bool:
        """ Method to remove a molecule from the collection

        Removes the specified molecule from the Collection. It
        checks whether the provided id is actually in the Collection.
        Otherwise, it warns the user.

        Parameters
        ----------
        idm : str
            The name of the molecule in the collection.

        Returns
        -------
        bool
            True if everything works out.
        """

        # Check that the molecule actually exists
        if idm in self.molecules.keys():
            self.__natoms -= self.molecules[idm].num_atoms()
            self.__nmols -= 1
            # Remove the molecule from the collection
            del self.molecules[idm]
            # Update the atoms reference map
            self.__remap()
            return True
        else:
            warn((f"Collection.remove_molecule() No molecule {idm} "
                  "in the collection; no molecule deleted."))
            return False
    
    @property
    def atoms(self) -> list:
        """ Method to get the collection's atoms

        Returns
        -------
        todos : list of Atom
            A list with all the atoms in the collection.
        """
        todos = []

        for mol in self.molecules.keys():
            for a in self.molecules[mol].atoms:
                todos.append(a)

        return todos

    def num_atoms(self) -> int:
        """ Method to get the number of atoms in the collection

        Returns
        -------
        int
            The number of atoms in the collection.
        """
        return self.__natoms
    
    def get_coords(self) -> list:
        """ Method to get the collection's coordinates

        Returns
        -------
        todos : list of list
            A list with the atoms represented by lists with
            the symbol and X, Y, Z coordinates.
        """
        todos = []

        for mol in self.molecules.keys():
            for a in self.molecules[mol].atoms:
                x, y, z = a.coordinates
                todos.append([a.element, x, y, z, a.charge])

        return todos

    def get_density(self, corner : bool = False) -> float:
        """ Calculate the collection's density

        The method will compute the total mass and volume of
        the collection and divide them to obtain the density.
        Consider that several units have to be adjusted!

        Parameters
        ----------
        corner : bool
            If True, the method will first compute the bounding
            box of the collection, shift that box to the origin
            and then compute the density.

        Returns
        -------
        density : float
            The density of the collection in g/cm^3
        """
        # Trivial case
        if len(self.molecules.keys()) == 0:
            return 0

        if corner:
            self.corner_box()

        # Molecules per mol
        avogadro = 6.022E23

        # Get the mass of all molecules in g/mol
        mass = 0
        for mol_v in self.molecules.values():
            mass += mol_v.mol_weight

        # Gram per mol to kilogram
        mass /= (1000 * avogadro)

        # Get side lengths and compute the volume in Angstrom
        lims = self.get_limits()
        volume = lims['X'][2] * lims['Y'][2] * lims['Z'][2]

        # Cubic Angstrom to cubic meter
        volume *= (1E-10)**3

        # Density in kg/m^3
        density = mass / volume

        # Density in g/cm^3 (g/mL)
        density /= 1000

        return density
    
    def get_total_mass(self) -> float:
        """ Method to get the mass of the whole collection

        Compute the mass of all the atoms in the collection.

        Returns
        -------
        mass : float
            The total mass of the collection in uma
        """
        # Get the mass of all molecules in uma
        mass = 0
        for mol_v in self.molecules.values():
            mass += mol_v.mol_weight

        return mass
                
    def get_center(self) -> np.ndarray:
        """ Method to get the geometric center of the collection

        Compute the center of the collection solely as an average
        of the coordinates of its atoms.

        Returns
        -------
        collection_center : ndarray
            A NumPy array with the X, Y, Z coordinates of the
            geometric center of the molecule.
        """
        # Start assuming that the center is at 0, 0, 0
        collection_center = np.array([0,0,0], dtype=np.float64)

        # Iterate over all molecules and atoms
        for mol in self.molecules.values():
            for atom in mol.atoms:

                # Take the coordinates of each atom and add them to the center
                collection_center += atom.coordinates

        # Scaling it down by the number of atoms
        collection_center /= self.__natoms

        return collection_center
    
    def get_center_of_mass(self) -> np.ndarray:
        """ Method to get the center of mass of the collection

        Returns
        -------
        collection_com : ndarray
            A NumPy array with the X, Y, Z coordinates of the
            geometric center of the molecule.
        """
        # Start assuming that the center is at 0, 0, 0
        collection_com = np.array([0,0,0], dtype=np.float64)

        # Iterate over all molecules and atoms
        for mol in self.molecules.values():
            for atom in mol.atoms:

                # Take the coordinates of each atom and add them to the center
                collection_com += atom.coordinates * PTE[atom.element].mass

        # Scaling it down by the number of atoms
        collection_com /= self.get_total_mass()

        return collection_com

    def get_limits(self,
                   option : str = "edges",
                   factor : float = 2.5
                   ) -> dict:
        """ Get the limits of the molecular collection

        The function finds the maximum and minimum values for
        each coordinate: X, Y, Z. It returns that and the
        distance of the collection in each axis.

        This function has a caching mechanism to avoid
        re-computing the limits of the collection.

        Parameters
        ----------
        option : str
            The option for the limits: "edges", "factor", "scan".
        factor : float
            A padding factor for each of the limits.

        Returns
        -------
        lims : dict
            The lowest and highest values for the coordinates of
            the atoms in the collection, in each axis, and the size
            of the collection in each axis.
        """
        # Trivial case
        if len(self.molecules) == 0:
            return { "x" : [0, 0, 0], "y" : [0, 0, 0], "z" : [0, 0, 0] }

        # Change the representation of the coordinates to
        # lists in each dimension
        q_trsp = { q : [] for q in "exyz" }

        for idm, mol in self.molecules.items():
            for a in mol.get_coords():
                q_trsp["e"].append(a[0])
                q_trsp["x"].append(a[1])
                q_trsp["y"].append(a[2])
                q_trsp["z"].append(a[3])
        
        # Build a new dictionary to hold the limits
        lims = {}

        for q in "xyz":

            # Compute the minimum and maximum values
            low = min(q_trsp[q])
            high = max(q_trsp[q])

            # Find those values in the list of atoms
            id_l = q_trsp[q].index(low)
            id_h = q_trsp[q].index(high)

            # Get the atoms' atomic radius to pad the molecule
            pad_i = PTE[q_trsp['e'][id_l]].vdw_radius
            pad_a = PTE[q_trsp['e'][id_h]].vdw_radius

            # Compute the limits
            lims[q] = [low - pad_i,
                       high + pad_a,
                       high + pad_a - (low - pad_i)]
        
        # If only edges are needed
        if option == "edges":
            return lims
        
        # If a factor is to be used
        if option == "factor":

            # Pad the limits
            for q in "xyz":
                lims[q][0] += factor
                lims[q][1] -= factor
                lims[q][2] = lims[q][1] - lims[q][0]

            return lims

        # If a scan is to be used, the mass per bin should
        # be computed over several bin widths
        if option == "scan":

            iteration = []

            for iter in range(7):
                bin_width = 0.2 + 0.05 * iter

                iteration.append({})

                # Build a new dictionary to hold the separators
                bins = {}
                bin_idx = {}
                seps = {}
                number_bins = {}

                # Create the bins and separators
                for q in "xyz":

                    # Compute the number of bins
                    number_bins[q] = abs(int(lims[q][1] / bin_width))

                    # Compute the new width of each bin
                    new_delta = (lims[q][2] + bin_width)
                    new_delta /= number_bins[q]

                    # Create the bins
                    bins[q] = [0.0] * (number_bins[q] + 1)

                    # Create the separators
                    seps[q] = np.linspace(
                        lims[q][0] - bin_width,
                        lims[q][1] + bin_width,
                        number_bins[q] + 2)

                    # Find the bins
                    bin_idx[q] = np.searchsorted(seps[q], q_trsp[q])

                # Add the mass of each atom to the bin
                for q in "xyz":
                    for i, a in enumerate(q_trsp[q]):
                        bins[q][bin_idx[q][i]] += PTE[q_trsp['e'][i]].mass

                iteration[iter]['bins'] = bins
                iteration[iter]['seps'] = seps
                iteration[iter]['number_bins'] = number_bins
            
            # Smoothening the distribution
            final_bins = {q : [] for q in "xyz"}
            for iter in iteration:
                for q in "xyz":
                    x = iter['seps'][q]
                    y = iter['bins'][q]
                    final_bins[q] += list(zip(x, y))

            # Curve fitting
            curve_params = {}
            for q in "xyz":
                final_bins[q].sort(key=lambda x: x[0])
                temp = list(zip(*final_bins[q]))

                height = self.get_total_mass() / lims[q][2]
                
                curve_params[q] = curve_fit(
                                    _subbotin,
                                    temp[0],
                                    temp[1],
                                    bounds=[
                                        [height, lims[q][1]/2, -lims[q][1]/2],
                                        [np.inf, lims[q][1], lims[q][1]/2]
                                    ])

                lims[q][0] = curve_params[q][0][2] - curve_params[q][0][1]
                lims[q][1] = curve_params[q][0][2] + curve_params[q][0][1]
                lims[q][2] = lims[q][1] - lims[q][0]

            return lims
        
        return lims
    
    def is_in_box(self, idm : str, dims : dict) -> bool:
        """ Check if a molecule is in a given box region

        It will try to find the molecule in the box specified
        by the provided dimensions. If at least one atom is
        inside the provided box, it will return True.

        Raises
        ------
        ValueError
            If the molecule is not in the collection

        Parameters
        ----------
        idm : str
            The name of the molecule to be found
        dims : dict
            The lowest and highest values of the coordinates
            in each axis, for the atoms in the sub-collection

        Returns
        -------
        bool
            True or false depending if the molecule is in the box
        """
        # Sanity check
        if idm not in self.molecules.keys():
            raise ValueError((f"Collection.is_in_box() The molecule {idm} "
                              "is not part of the collection."))

        # Initialize the atom count
        inside_atoms = 0

        # Get the collection limits
        lims = self.get_limits()

        # Check that the dimensions of the box are within the collection
        inside_min = [dims[q][0] > lims[q][0] for q in 'xyz']
        inside_max = [dims[q][1] < lims[q][1] for q in 'xyz']

        inside = inside_min + inside_max

        limx = list(dims['x'])
        limy = list(dims['y'])
        limz = list(dims['z'])

        # If any of the 6 coordinates of the box are out of the collection ...
        if sum(inside) != 6:
            if inside[0]: limx[0] = lims['x'][0]
            if inside[1]: limy[0] = lims['y'][0]
            if inside[2]: limz[0] = lims['z'][0]
            if inside[3]: limx[1] = lims['x'][1]
            if inside[4]: limy[1] = lims['y'][1]
            if inside[5]: limz[1] = lims['z'][1]

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


    def detect_collisions(self) -> bool:
        """ Method to check if two molecules collide inside the collection

        Only checks if two molecules are too close in the collection and
        classifies it as a collision.

        Returns
        -------
        bool
            True if two atoms from two different molecules are too close
            False if all molecules are sufficiently far away, or if there
            are not enough molecules for collisions.
        """

        # If there's less than 2 atoms, there's no point
        if self.__natoms < 2:
            warn("Collection.detect_collisions() Not enough "
                 "molecules in the collection to check for"
                 "collisions.")
            return False

        # Remaining molecules to be checked
        to_check = list(self.molecules.keys())

        # Iterate over the first molecule
        for idm1 in self.molecules.keys():
            # Remove the molecule from the list
            to_check.remove(idm1)
            # Iterate over all remaining molecules
            for idm2 in to_check:
                # Iterate over atoms of both molecules
                for a1 in self.molecules[idm1].atoms:
                    for a2 in self.molecules[idm2].atoms:
                        # Compute the minumum distance and the real one
                        min_dist = a1.vdw_radius + a2.vdw_radius
                        real_dist = np.linalg.norm(a2.coordinates - a1.coordinates)
                        # Check for a collision
                        if real_dist <= min_dist:
                            warn("Collection.detect_collisions() "
                                 "Collision found between molecules "
                            f"{idm1} and {idm2}.")

                            return True
        return False


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

        # Loop over molecules and atoms in the collection
        for mol in self.molecules.values():
            for a in mol.get_coords():
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
        for q in "xyz":
            temp_low = limits[q][0] - limits[q][2] * padding
            temp_high = limits[q][1] + limits[q][2] * padding
            box[q] = np.linspace(temp_low,
                                 temp_high,
                                 int((temp_high - temp_low) / mesh) + 1)
        
        # Create empty grid
        grid = []
        # Iterate over x coordinate
        for x in box['x']:
            # Iterate over y coordinate
            for y in box['y']:
                # Iterate over z coordinate
                for z in box['z']:
                    grid.append([x,y,z])
        
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
    
    def compute_inertia_tensor(self, bohr : bool = False) -> tuple:
        """ Method to compute the inertia tensor of the collection

        This method computes the inertia tensor of the collection,
        its eigenvalues and eigenvectors, and also returns the
        shifted atomic coordinates (to the center of mass).

        Parameters
        ----------
        bohr : bool
            Should the calculation be done in Bohr, instead of
            Angstrom?

        Returns
        -------
        inertia_tensor : ndarray
            The inertia tensor of the collection.
        eig_val : ndarray
            The eigenvalues of the inertia tensor.
        eig_vec : ndarray
            The eigenvectors of the inertia tensor.
        shifted_atoms : list
            A list with the shifted atomic coordinates.
        """
        # Get the coordinates of all the atoms
        atoms = self.get_coords()
        atoms = [(a[0], np.array([*a[1:4]])) for a in atoms]

        # Center of mass
        com = self.get_center_of_mass()

        # Shift all atoms to the center of mass
        if bohr:
            b = cts.physical_constants['Bohr radius'][0] * 1E10
            shifted_atoms = [[a[0], (a[1] - com) / b] for a in atoms]
        else:
            shifted_atoms = [[a[0], a[1] - com] for a in atoms]

        # Build the inertia tensor
        Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0

        for a in shifted_atoms:
            Ixx += PTE[a[0]].mass * (a[1][1]**2 + a[1][2]**2)
            Iyy += PTE[a[0]].mass * (a[1][0]**2 + a[1][2]**2)
            Izz += PTE[a[0]].mass * (a[1][0]**2 + a[1][1]**2)

            Ixy += PTE[a[0]].mass * (a[1][0] * a[1][1])
            Ixz += PTE[a[0]].mass * (a[1][0] * a[1][2])
            Iyz += PTE[a[0]].mass * (a[1][1] * a[1][2])
        
        inertia_tensor = np.array([
            [ Ixx, -Ixy, -Ixz],
            [-Ixy,  Iyy, -Iyz],
            [-Ixz, -Iyz,  Izz]
        ])

        # Diagonalize the inertia tensor
        eig_val, eig_vec = np.linalg.eigh(inertia_tensor)

        return inertia_tensor, eig_val, eig_vec, shifted_atoms

    def corner_box(self) -> None:
        """ Re-position the collection putting an edge on the origin

        The lower limits (in the x, y, z axes) of the collection will
        be re-positioned to the origin. The idea is not to have
        negative coordinates.

        Note
        ----
            This method doesn't require any parameters and will not
            return anything. The change is done to the collection object
            itself.
        """

        lims = self.get_limits()
        mins = np.array([lims['x'][0], lims['y'][0], lims['z'][0]])

        # Iterate over molecules
        for im, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.atoms:
                # Get the atom's current coordinates
                coords = a.coordinates
                # Compute the new coordinates
                new_coords = coords - mins
                # Move the atom ...
                a.coordinates = np.ndarray(new_coords[0],
                                           new_coords[1],
                                           new_coords[2])
    
    def center_box(self) -> None:
        """ Re-position the collection putting the center at the origin

        The center of the box (in the x, y, z axes) of the collection will
        be re-positioned to the origin.

        Note
        ----
            This method doesn't require any parameters and will not
            return anything. The change is done to the collection object
            itself.
        """

        center = self.get_center() * (-1)

        # Iterate over molecules
        for km, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.atoms:
                # Move the current atom
                a.move_atom(*center.tolist())
    

    def __encoord(self, dims : dict) -> list:
        """ Method to encode dimensions

        The method encodes the dimensions of a sub-collection
        relative to a super-collection, in a list of hexadecimals.

        Parameters
        ----------
        dims : dict
            A `dict` containing the x, y, z lower and upper limits
            of the sub-collection.

        Returns
        -------
        into_hex : list
            A list with 3 `str` objects (hex numbers).
        """
        lims = self.get_limits()
        # Enconde the position of the sub-collection
        ratio5 = [round(dims[q][0]/lims[q][1] * 1E6) for q in 'xyz']
        into_hex = [hex(r)[2:] for r in ratio5]
        return into_hex

    def __decoord(self) -> dict:
        """ Method to decode dimensions

        The method decodes the dimensions of the super-collection
        relative to the sub-collection.

        Returns
        -------
        coords : dict
            A `dict` with the x, y, z upper limits of the super-collection.
        """

        # If the encoding in the name was done correctly ...
        if self.name.count("|") == 2:
            # Extract the hex-coordinates
            namx, y, z = self.name.split('|')
            x = namx[-5:]
            # Get the limits of the current sub-collection
            l = self.get_limits()
            # Some structure
            qs = {'x':x, 'y':y, 'z':z}
            # Convert the hex-coordinates into relative coordinates
            dec = {i:int(q, 16) * 1E-6 for i, q in qs.items()}
            # Create the final coordinates of the upper limit of the
            # super-collection.
            coords = {q:round(l[q][0]/dec[q], 3) for q in 'xyz'}
            return coords
        else:
            raise ValueError(("Collection.__decoord() The name of the "
                              "collection is not in the expected format."))


    def sub_collection(self, dims : dict):
        """ Extract a collection box from the larger collection

        All molecules with at least one atom inside the box
        defined by the provided dimensions will be included
        and returned in a new collection. It considers periodic
        boundary conditions, so if the sub-collection box is
        slightly out of the collection's boundaries, it the
        function will replicate the molecules to fill the box
        specified by the provided dimensions.

        Note
        ----
        The new collection may have different dimensions than
        the ones defined by the provided dimensions. This
        happens because any molecule with at least one atom
        inside the box will be included. That molecule will
        change the final dimensions of the box.

        Parameters
        ----------
        dims : dict
            The lowest and highest values of the coordinates
            in each axis, for the atoms in the sub-collection.

        Returns
        -------
        sub_c : Collection
            A Collection object with all the molecules within
            the provided dimensions.
        """

        # Setting everything correctly before creating subset
        self.corner_box()

        # Get the collection limits
        lims = self.get_limits()

        #TODO: Re-write the next part, since the dimensions should be either fully
        #      inside the collection or, PBC should be specified as an option of
        #      the current function.

        # Check that the dimensions of the small box are within the collection
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

        # Building the list of molecules within x
        possible_x = [[] for i in range(len(sub_dims['x']))]
        # Building the list of molecules within y
        possible_y = [[] for j in range(len(sub_dims['y']))]
        # Building the list of molecules within z
        possible_z = [[] for k in range(len(sub_dims['z']))]

        # Iterate over all molecules ...
        for idm, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.get_coords():
                # Iterate over the new limits

                # Check over all domains
                for i, x in enumerate(sub_dims['x']):
                    # Add molecule if within x
                    if (a[1] > x[0]) and (a[1] < x[1]):
                        possible_x[i].append(idm)

                # Check over all domains
                for j, y in enumerate(sub_dims['y']):
                    # Add molecule if within y
                    if (a[2] > y[0]) and (a[2] < y[1]):
                        possible_y[j].append(idm)

                # Check over all domains
                for k, z in enumerate(sub_dims['z']):
                    # Add molecule if within z
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

        # Create a new collection with the required molecules
        sub_c = Collection()

        # Iterate over all sets
        for ids, mol_set in mol_sets.items():

            # Prepare to move the molecules
            motion = np.array([ int(ids[0]) * lims['x'][2],
                                int(ids[1]) * lims['y'][2],
                                int(ids[2]) * lims['z'][2],])

            # Iterate over all molecules
            for idm in mol_set:
                # Add the molecule to the new collection
                # If the object is not deepcopied, then the original will
                # suffer the same fate as the copy
                sub_c.add_molecule(idm, deepcopy(self.molecules[idm]))
                # Move the molecule
                sub_c.molecules[idm].move_molecule(motion)

        # Enconde the position of the sub collection
        sub_lims = sub_c.get_limits()
        codes = self.__encoord(sub_lims)
        # Name the new collection
        sub_c.name = f'{self.name}_{codes[0]}|{codes[1]}|{codes[2]}'

        return sub_c


    def save_as_pdb(self,
                    f_nam : str = "collection",
                    occupancies : list = []) -> None:
        """ Save collection as an PDB file

        This method does not return anything, nor it requires
        any parameters.

        Parameters
        ----------
        f_nam : str
            The name of the file *without the extension*!
        occupancies : list
            List of occupancies
        """
        # Check that there are molecules in the collection
        if len(self.molecules) == 0:
            raise ValueError("Collection.save_as_pdb() The collection is empty. "
                             "There are no molecules to save.")

        # Check that there are occupancies
        if len(occupancies) == 0:
            # If not, set all occupancies to 1.0
            occupancies = [1.0] * self.__natoms

        # Check that there is the same number of occupancies as atoms
        if len(occupancies) != self.__natoms:
            raise ValueError("Collection.save_as_pdb() The collection has "
                             f"{self.__natoms} atoms, but there are "
                             f"{len(occupancies)} occupancies.")

        # Initialize the PDB file content
        content = ("CRYST1    0.000    0.000    0.000  "
                    "90.00  90.00  90.00 P 1           1\n")

        # Create a template for the PDB coordinates
        # https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
        pdb_template = ("ATOM "        #                  1 -  4 + space
                        "{num:>6} "    # Atom number:     6 - 11 + space
                        "{ana:<4} "    # Atom name:      13 - 16 + space
                        "{nam:>3} "    # Residue name:   18 - 20 + space
                        "X"            # Chain ID:       22 - 22
                        "{molnum:>4} " # Residue number: 23 - 26 + space
                        "   "          # Whitespace:     28 - 30
                        "{x:8.3f}"     # X coordinate:   31 - 38
                        "{y:8.3f}"     # Y coordinate:   39 - 46
                        "{z:8.3f}"     # Z coordinate:   47 - 54
                        "{occ:>6.2f}"  # Occupancy:      55 - 60
                        "  0.00"       # Temperature f:  61 - 66
                        "      "       # Whitespace:     67 - 72
                        "    "         # Segment ID:     73 - 76
                        "{s:>2}"       # Element symbol: 77 - 78
                        "  \n"         # Charge:         79 - 80
                        )

        atom_counter = 0

        # Iterate over molecules
        for idm, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.atoms:
                # Get the atom's current coordinates
                coords = a.coordinates

                # Increment the number of atoms
                atom_counter += 1

                # Check if there's an Amber name
                try:
                    anam = a.amber_name
                except AttributeError:
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
                    occ=occupancies[atom_counter - 1],
                    s=a.element)

        content += "END\n"

        with open(f"{f_nam}.pdb", "w") as xyz:
            xyz.write(content)


    def save_as_xyz(self, f_nam : str = "") -> None:
        """ Save collection as an XYZ file

        This method does not return anything, nor it requires
        any parameters.

        Parameters
        ----------
        f_nam : str
            The name of the file *without the extension*!
        """
        # Check that the name is not empty
        if f_nam == "":
            f_nam = self.name

        # Create a template for the XYZ coordinates
        template = " {s} {x:16.8f} {y:16.8f} {z:16.8f}\n"

        content = f"""{self.__natoms}
XYZ file of collection: {self.name} - created by InformalFF
"""

        # Iterate over molecules
        for idm, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.get_coords():
                content += template.format(s=a[0], x=a[1], y=a[2], z=a[3])

        with open(f"{f_nam}.xyz", "w") as xyz:
            xyz.write(content)
    
    def save_selection_as_xyz(self, f_nam : str ="collection") -> None:
        """ Save collection as an XYZ file

        This method does not return anything, nor it requires
        any parameters.

        Parameters
        ----------
        f_nam : str
            The name of the file *without the extension*!
        """

        # Create a template for the XYZ coordinates
        template = " {s} {x:16.8f} {y:16.8f} {z:16.8f}\n"

        num_atoms = 0
        selected_coords = ""

        # Iterate over molecules
        for idm, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.atoms:
                if a.flag:
                    num_atoms += 1
                    selected_coords += template.format(
                                                s=a.element,
                                                x=a.coordinates[0],
                                                y=a.coordinates[1],
                                                z=a.coordinates[2])

        header = f"""{num_atoms}
XYZ file of atom selection from collection: {self.name} - created by InformalFF
"""

        with open(f"{f_nam}.xyz", "w") as xyz:
            xyz.write(header + selected_coords)


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("This library was not intended as a standalone program.")