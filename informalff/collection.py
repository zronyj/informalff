import copy              # To copy objects
import numpy as np       # To do basic scientific computing
from multiprocessing import Pool # To parallelize jobs
import warnings          # To throw warnings instead of raising errors
from functools import lru_cache # To cache functions
from scipy.special import gamma # To compute the gamma function
from scipy.optimize import curve_fit # To fit the Subbotin function


from .atom import PERIODIC_TABLE
from .molecule import Molecule, BOHR

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
    
    def __repr__(self):
        """ Method to represent a Collection

        This method builds a string with the information
        of the Collection object. Said string will be displayed
        as a ticket whenever someone prints this object.

        Returns
        -------
            text : str
                A general description of the Collection, its molecules,
                atoms and dimensions.
        """
        content = "\n==============================\n"
        content += f"     Molecular Collection\n{self.name:^30}\n"
        content += "------------------------------\n"
        content += f" Total molecules: {self.__nmols}\n"
        content += f" Total atoms: {self.__natoms}\n\n"
        content += "        Molecules\n------------------------------\n"
        if self.__nmols > 0:
            frags = [m[:3] for m in self.molecules.keys()]
            ufrags = set(frags)
            for f in ufrags:
                content += f"{f}:\n"
                content += f"Number of molecules: {frags.count(f)}\n"
                for m in self.molecules.keys():
                    if f in m:
                        temp = m
                        break
                content += ("Atoms per molecule: "
                            f"{self.molecules[temp].get_atoms()}\n")
            content += "\n         Limits\n------------------------------\n"
            lims = self.get_limits()
            content += "     Lower    Upper  Side\n"
            content += (f"X:{lims['x'][0]:8.3f} {lims['x'][1]:8.3f}"
                        f" {lims['x'][2]:5.2f}\n")
            content += (f"Y:{lims['y'][0]:8.3f} {lims['y'][1]:8.3f}"
                        f" {lims['y'][2]:5.2f}\n")
            content += (f"Z:{lims['z'][0]:8.3f} {lims['z'][1]:8.3f}"
                        f" {lims['z'][2]:5.2f}\n")
        content += "\n            Density           \n"
        content += "------------------------------\n"
        content += f"    {self.get_density():>8.4f} g/cm^3      \n"
        content += "==============================\n"
        return content
    
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
        self.__natoms += mol.get_num_atoms()

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
            self.__natoms -= self.molecules[idm].get_num_atoms()
            self.__nmols -= 1
            # Remove the molecule from the collection
            del self.molecules[idm]
            return True
        else:
            warnings.warn((f"Collection.remove_molecule() No molecule {idm} "
                           "in the collection; no molecule deleted."))
            return False
    
    def get_density(self) -> float:
        """ Calculate the collection's density

        The method will compute the total mass and volume of
        the collection and divide them to obtain the density.
        Consider that several units have to be adjusted!

        Returns
        -------
        density : float
            The density of the collection in g/cm^3
        """
        # Trivial case
        if len(self.molecules.keys()) == 0:
            return 0

        self.corner_box()

        # Molecules per mol
        avogadro = 6.022E23

        # Get the mass of all molecules in g/mol
        mass = 0
        for mol in self.molecules.keys():
            self.molecules[mol].get_mol_weight()
            mass += self.molecules[mol].mol_weight

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
    
    @lru_cache(maxsize=1)
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
        for mol in self.molecules.keys():
            self.molecules[mol].get_mol_weight()
            mass += self.molecules[mol].mol_weight

        return mass
                
    @lru_cache(maxsize=1)
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
                collection_center += atom.coords

        # Scaling it down by the number of atoms
        collection_center /= self.__natoms

        return collection_center

    @lru_cache(maxsize=3)
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
            return { "X" : [0, 0, 0], "Y" : [0, 0, 0], "Z" : [0, 0, 0] }

        # Change the representation of the coordinates to
        # lists in each dimension
        q_trsp = { q : [] for q in "eXYZ" }

        for idm, mol in self.molecules.items():
            for a in mol.get_coords():
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

            # From pm to Angstrom
            pad_i /= 100
            pad_a /= 100

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
            for q in "XYZ":
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
                for q in "XYZ":

                    # Compute the number of bins
                    number_bins[q] = int(lims[q][1] / bin_width)

                    # Compute the new width of each bin
                    new_delta = (lims[q][1] - lims[q][0] + bin_width)
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
                for q in "XYZ":
                    for i, a in enumerate(q_trsp[q]):
                        mass = PERIODIC_TABLE.loc[q_trsp['e'][i], "AtomicMass"]
                        bins[q][bin_idx[q][i]] += mass

                iteration[iter]['bins'] = bins
                iteration[iter]['seps'] = seps
                iteration[iter]['number_bins'] = number_bins
            
            # Smoothening the distribution
            final_bins = {q : [] for q in "XYZ"}
            for iter in iteration:
                for q in "XYZ":
                    x = iter['seps'][q]
                    y = iter['bins'][q]
                    final_bins[q] += list(zip(x, y))

            # Curve fitting
            curve_params = {}
            for q in "XYZ":
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
                              "is part of the collection."))

        # Initialize the atom count
        inside_atoms = 0

        # Get the collection limits
        lims = self.get_limits()

        # Check that the dimensions of the box are within the collection
        inside_min = [dims[q][0] > lims[q][0] for q in 'XYZ']
        inside_max = [dims[q][1] < lims[q][1] for q in 'XYZ']

        inside = inside_min + inside_max

        limx = list(dims['X'])
        limy = list(dims['Y'])
        limz = list(dims['Z'])

        # If any of the 6 coordinates of the box are out of the collection ...
        if sum(inside) != 6:
            if inside[0]: limx[0] = lims['X'][0]
            if inside[1]: limy[0] = lims['Y'][0]
            if inside[2]: limz[0] = lims['Z'][0]
            if inside[3]: limx[1] = lims['X'][1]
            if inside[4]: limy[1] = lims['Y'][1]
            if inside[5]: limz[1] = lims['Z'][1]

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
            warnings.warn("Collection.detect_collisions() Not enough "
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
                for a1 in self.molecules[idm1].get_coords():
                    # Find the VdW radius of first atom and its position
                    vdw_r1 = PERIODIC_TABLE.loc[a1[0], "AtomicRadius"] / BOHR
                    v1 = np.array(a1[1:4])
                    for a2 in self.molecules[idm2].get_coords():
                        # Find the VdW radius of second atom and its position
                        vdw_r2 = PERIODIC_TABLE.loc[a2[0], "AtomicRadius"] / BOHR
                        v2 = np.array(a2[1:4])
                        # Compute the minumum distance and the real one
                        min_dist = (vdw_r1 + vdw_r2) / 100
                        real_dist = np.linalg.norm(v2 - v1)
                        # Check for a collision
                        if real_dist <= min_dist:
                            warnings.warn("Collection.detect_collisions() "
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

    def corner_box(self) -> None:
        """ Re-position the collection putting an edge on the origin

        The lower limits (in the X, Y, Z axes) of the collection will
        be re-positioned to the origin. The idea is not to have
        negative coordinates.

        Note
        ----
            This method doesn't require any parameters and will not
            return anything. The change is done to the collection object
            itself.
        """

        lims = self.get_limits()
        mins = np.array([lims['X'][0], lims['Y'][0], lims['Z'][0]])

        # Iterate over molecules
        for mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.atoms:
                # Get the atom's current coordinates
                coords = a.get_coordinates()
                # Compute the new coordinates
                new_coords = coords - mins
                # Move the atom ...
                a.set_coordinates(new_coords[0],new_coords[1],new_coords[2])
    
    def center_box(self) -> None:
        """ Re-position the collection putting the center at the origin

        The center of the box (in the X, Y, Z axes) of the collection will
        be re-positioned to the origin.

        Note
        ----
            This method doesn't require any parameters and will not
            return anything. The change is done to the collection object
            itself.
        """

        center = self.get_center() * (-1)

        # Iterate over molecules
        for mol in self.molecules.items():
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
            A `dict` containing the X, Y, Z lower and upper limits
            of the sub-collection.

        Returns
        -------
        into_hex : list
            A list with 3 `str` objects (hex numbers).
        """
        lims = self.get_limits()
        # Enconde the position of the sub-collection
        ratio5 = [round(dims[q][0]/lims[q][1] * 1E6) for q in 'XYZ']
        into_hex = [hex(r)[2:] for r in ratio5]
        return into_hex

    def __decoord(self) -> dict:
        """ Method to decode dimensions

        The method decodes the dimensions of the super-collection
        relative to the sub-collection.

        Returns
        -------
        coords : dict
            A `dict` with the X, Y, Z upper limits of the super-collection.
        """

        # If the encoding in the name was done correctly ...
        if self.name.count(".") == 2:
            # Extract the hex-coordinates
            namx, y, z = self.name.split('.')
            x = namx[-5:]
            # Get the limits of the current sub-collection
            l = self.get_limits()
            # Some structure
            qs = {'X':x, 'Y':y, 'Z':z}
            # Convert the hex-coordinates into relative coordinates
            dec = {i:int(q, 16) * 1E-6 for i, q in qs.items()}
            # Create the final coordinates of the upper limit of the
            # super-collection.
            coords = {q:round(l[q][0]/dec[q], 3) for q in 'XYZ'}
            return coords


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
        outside = [dims[q][1] > lims[q][1] for q in 'XYZ']

        # Initialize dimensions of the sub-box
        sub_dims = {}

        # Dimensions of the sub-box
        for i, q in enumerate('XYZ'):
            # If this dimension is outside, establish new limits (PBC)
            if outside[i]:
                sub_dims[q] = [[dims[q][0], lims[q][1]],
                        [lims[q][0], lims[q][0] + dims[q][1] - lims[q][1]]]
            # Else, just use the current limits
            else:
                sub_dims[q] = [[dims[q][0], dims[q][1]]]

        # Building the list of molecules within X
        possible_x = [[] for i in range(len(sub_dims['X']))]
        # Building the list of molecules within Y
        possible_y = [[] for j in range(len(sub_dims['Y']))]
        # Building the list of molecules within Z
        possible_z = [[] for k in range(len(sub_dims['Z']))]

        # Iterate over all molecules ...
        for idm, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.get_coords():
                # Iterate over the new limits

                # Check over all domains
                for i, x in enumerate(sub_dims['X']):
                    # Add molecule if within X
                    if (a[1] > x[0]) and (a[1] < x[1]):
                        possible_x[i].append(idm)

                # Check over all domains
                for j, y in enumerate(sub_dims['Y']):
                    # Add molecule if within Y
                    if (a[2] > y[0]) and (a[2] < y[1]):
                        possible_y[j].append(idm)

                # Check over all domains
                for k, z in enumerate(sub_dims['Z']):
                    # Add molecule if within Z
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
            motion = np.array([ int(ids[0]) * lims['X'][2],
                                int(ids[1]) * lims['Y'][2],
                                int(ids[2]) * lims['Z'][2],])

            # Iterate over all molecules
            for idm in mol_set:
                # Add the molecule to the new collection
                # If the object is not deepcopied, then the original will
                # suffer the same fate as the copy
                sub_c.add_molecule(idm, copy.deepcopy(self.molecules[idm]))
                # Move the molecule
                sub_c.molecules[idm].move_molecule(motion)

        # Enconde the position of the sub collection
        sub_lims = sub_c.get_limits()
        codes = self.__encoord(sub_lims)
        # Name the new collection
        sub_c.name = f'{self.name}_{codes[0]}.{codes[1]}.{codes[2]}'

        return sub_c


    def save_as_pdb(self, f_nam : str = "collection") -> None:
        """ Save collection as an PDB file

        This method does not return anything, nor it requires
        any parameters.

        Parameters
        ----------
        f_nam : str
            The name of the file *without the extension*!
        """

        # Initialize the PDB file content
        content = ("CRYST1    0.000    0.000    0.000  "
                    "90.00  90.00  90.00 P 1           1\n")

        # Create a template for the PDB coordinates
        pdb_template = ("ATOM {num:>6} {s:>2}   {nam} X{molnum:>4}     "
                        "{x:7.3f} {y:7.3f} {z:7.3f}  1.00  0.00          "
                        "{s:>2}\n")

        atom_counter = 0

        # Iterate over molecules
        for idm, mol in self.molecules.items():
            # Iterate over atoms
            for a in mol.atoms:
                # Get the atom's current coordinates
                coords = a.get_coordinates()

                # Increment the number of atoms
                atom_counter += 1

                # Check if there's an Amber name
                try:
                    anam = a.amber_name
                except AttributeError as e:
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
                    s=a.element)

        content += "END\n"

        with open(f"{f_nam}.pdb", "w") as xyz:
            xyz.write(content)


    def save_as_xyz(self, f_nam : str ="collection") -> None:
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
                num_atoms += 1
                selected_coords += template.format(
                                            s=a.element,
                                            x=a.coords[1],
                                            y=a.coords[2],
                                            z=a.coords[3])

        header = f"""{self.__natoms}
XYZ file of atom selection from collection: {self.name} - created by InformalFF
"""

        with open(f"{f_nam}.xyz", "w") as xyz:
            xyz.write(header + selected_coords)


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("This library was not intended as a standalone program.")