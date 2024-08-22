import numpy as np       # To do basic scientific computing
from multiprocessing import Pool # To parallelize jobs
import warnings          # To throw warnings instead of raising errors

from .atom import PERIODIC_TABLE
from .molecule import Molecule, BOHR

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
            A name for the cluster (can be anything you choose)
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
        content = "\n=========================\n"
        content += f"  Molecular  Collection\n{self.name:^25}\n"
        content += "-------------------------\n"
        content += f" Total molecules: {self.__nmols}\n"
        content += f" Total atoms: {self.__natoms}\n\n"
        content += "        Molecules\n-------------------------\n"
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
            content += "\n         Limits\n-------------------------\n"
            lims = self.get_limits()
            content += "     Lower    Upper  Side\n"
            content += (f"X:{lims['x'][0]:8.3f} {lims['x'][1]:8.3f}"
                        f" {lims['x'][2]:5.2f}\n")
            content += (f"Y:{lims['y'][0]:8.3f} {lims['y'][1]:8.3f}"
                        f" {lims['y'][2]:5.2f}\n")
            content += (f"Z:{lims['z'][0]:8.3f} {lims['z'][1]:8.3f}"
                        f" {lims['z'][2]:5.2f}\n")
        content += "\n         Density         \n"
        content += "-------------------------\n"
        content += f"    {self.get_density():>8.4f} g/cm^3      \n"
        content += "=========================\n"
        return content
    
    def add_molecule(self, idm, mol):
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
            # Remove the molecule from the cluster
            del self.molecules[idm]
            return True
        else:
            warnings.warn((f"Collection.remove_molecule() No molecule {idm} "
                           "in the collection; no molecule deleted."))
            return False
    
    def clash_detector(self) -> bool:
        """ Method to check if two molecules clash inside the collection

        Only checks if two molecules are two close in the collection and
        classifies it as a clash.

        Returns
        -------
        bool
            True if two atoms from two different molecules are too close
            False if all molecules are sufficiently far away, or if there
            are not enough molecules for clashes.
        """

        # If there's less than 2 atoms, there's no point
        if self.__natoms < 2:
            warnings.warn("Collection.clash_detector() Not enough "
                          "molecules in the collection to check for"
                          "clashes.")
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
                        # Check for a clash
                        if real_dist <= min_dist:
                            warnings.warn("Collection.clash_detector() Clash "
                            f"found between molecules {idm1} and {idm2}.")

                            return True
        return False
                
    
    def get_center(self) -> np.ndarray:
        """ Method to get the geometric center of the collection

        Compute the center of the collection solely as an average
        of the coordinates of its atoms.

        Returns
        -------
        centro : ndarray
            A NumPy array with the X, Y, Z coordinates of the
            geometric center of the molecule.
        """
        # Start assuming that the center is at 0, 0, 0
        centro = np.array([0,0,0], dtype=np.float64)

        # Iterate over all molecules and atoms
        for mol in self.molecules.values():
            for atom in mol.atoms:

                # Take the coordinates of each atom and add them to the center
                centro += atom.coords

        # Scaling it down by the number of atoms
        centro *= (1.0/self.__natoms)

        return centro

    def get_limits(self) -> dict:
        """ Get the limits of the molecular collection

        The function finds the maximum and minimum values for
        each coordinate: X, Y, Z. It returns that and the
        distance of the collection in each axis.

        Returns
        -------
        lims : dict
            The lowest and highest values for the coordinates of
            the atoms in the collection, in each axis, and the size
            of the collection in each axis.
        """     
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
        
        return lims

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

    def corner_box(self):
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
        mins = np.array([lims['x'][0], lims['y'][0], lims['z'][0]])

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
    
    def center_box(self):
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


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("This library was not intended as a standalone program.")