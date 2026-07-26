from __future__ import annotations

import os                           # To navigate the file system
import json                         # To handle JSON files
import numpy as np                  # To do basic scientific computing
import spglib as spg                # To handle space groups
from functools import lru_cache     # To cache functions
from copy import deepcopy           # To copy objects completely
from pathlib import Path            # To locate files in the file system
from string import ascii_uppercase  # To generate labels
from decimal import Decimal         # To handle decimal numbers
from fractions import Fraction      # To handle fractions
from itertools import product       # To generate combinations of indices
from typing import Self             # Dirty way to return an object of the enclosing class

# To warn the user about certain circumstances
from warnings import filterwarnings, warn, catch_warnings

from .elements import PTE           # To handle elements and their properties
from .atom import Atom              # To handle atoms
from .molecule import Molecule      # To handle molecules
from .structure import Structure    # To handle structures
from .collection import Collection  # To handle collections of molecules

# Allows functions from spglib to raise native Python exceptions
# instead of using legacy error flags
try:
    spg.error.OLD_ERROR_HANDLING = False
except AttributeError:
    pass

# ------------------------------------------------------- #
#                   The Cell Type Class                   #
# ------------------------------------------------------- #

class CellType:
    """ Class to handle the type of a crystal cell.

    This class is used to represent a crystal cell type and
    offers methods to determine the type of a given cell,
    filter out the non-active cell parameters, leaving only
    the active ones, and to check if a given cell type is valid.

    Attributes
    ----------
    _type : str
        The type of the crystal cell.

    Methods
    -------
    __init__(cell_type) : None
        Initializes the CellType object with a given cell type.
    __repr__() : str
        Returns a string representation of the CellType object.
    __str__() : str
        Returns a string representation of the CellType object.
    type : str
        Property to get or set the cell type.
    
    Raises
    ------
    ValueError
        If an invalid cell type is provided.
    """
    def __init__(self, cell_type : str):
        """ Initialize the CellType object with a given cell type.

        Parameters
        ----------
        cell_type : str
            The type of the crystal cell.

        Raises
        ------
        ValueError
            If an invalid cell type is provided.
        """
        self.__retrieve_types()
        self.__cell_type_names = list(self.__cell_type_data.keys())
        self._type = None
        self.type = cell_type  # Use the setter to validate the input

    def find_type(self, params : dict, set_type : bool = True, tolerance : float = 1e-2) -> str:
        """ Determine the type of a crystal cell based on its parameters.

        Determine the most likely crystal type(s) for a full set of cell
        parameters (a, b, c, alpha, beta, gamma), ranked from most to least
        symmetric (fewest to most active/independent parameters).

        Because higher-symmetry systems are special cases of lower-symmetry
        ones (cubic subset of orthorhombic subset of monoclinic subset of
        triclinic), several systems can match simultaneously. The
        result is a ranked list so the caller can pick the most specific
        (highest-symmetry) match; usually ranked[0].

        Parameters
        ----------
        params : dict
            A dictionary containing the cell parameters with keys:
            "a", "b", "c", "alpha", "beta", "gamma".
        set_type : bool, optional
            If True, set the type of the cell to the determined type.
        tolerance : float, optional
            The tolerance for comparing equal parameters.

        Returns
        -------
        str
            The determined type of the crystal cell.

        Raises
        ------
        ValueError
            If the provided parameters do not match any known cell type.
        """
        if not all(key in params for key in ("a", "b", "c", "alpha", "beta", "gamma")):
            raise ValueError("CellType.find_type() : Invalid cell parameters.")

        for cell_type, cell_data in self.__cell_type_data.items():
            active_params = cell_data["active"]
            fix_params = cell_data["fix"]
            equal_params = cell_data["equal"]

            # Check if all fixed parameters have the correct value
            if not all(params[param] == fix_params[param] for param in fix_params):
                continue

            # Check if all equal parameters have the same value
            if any(abs(params[param] - params[equal_params[param]]) >= tolerance for param in equal_params):
                continue

            if set_type:
                self.type = cell_type

            return cell_type

        raise ValueError("Invalid cell parameters.")

# ------------------------------------------------------- #
#                     The Cell Class                      #
# ------------------------------------------------------- #

class Cell:
    """
    Class representing a crystal cell with methods to calculate
    cell parameters and vectors, as well as the reciprocal cell.

    Notes
    -----
    Formulas extracted from:
    https://gisaxs.com/index.php/Unit_cell
    
    Attributes
    ----------
    _a : float
        The a parameter of the cell.
    _b : float
        The b parameter of the cell.
    _c : float
        The c parameter of the cell.
    _alpha : float
        The alpha parameter of the cell.
    _beta : float
        The beta parameter of the cell.
    _gamma : float
        The gamma parameter of the cell.
    _vector : np.ndarray
        The cell parameters as 3 3D vectors in matrix form.
    
    Methods
    -------
    __init__(*args) : None
        Initializes the Cell object with either a cell vector or cell parameters.
    __repr__() : str
        Returns a string representation of the Cell object.
    vector : numpy.ndarray
        Returns the cell vector as a 3x3 numpy array.
    params : dict
        Returns the cell parameters as a dictionary with keys:
        "a", "b", "c", "alpha", "beta", "gamma".
    volume : float
        Calculates and returns the volume of the cell.
    reciprocal() : dict
        Calculates and returns the reciprocal cell vector as a dictionary
        with keys "u", "v", "w".
    __cell_params() : None
        Calculates the cell parameters from the cell vector.
    __cell_vector() : None
        Calculates the cell vector from the cell parameters.
    """
    def __init__(self, *args):
        """
        Initialize the Cell object with either a cell vector or cell parameters.

        Parameters
        ----------
        args : numpy.ndarray, list, or individual parameters
            - If a numpy array, it should be a 3x3 array representing the
              cell vector.
            - If a list, it can be of length 3 (for cell vector) or 6 (for
              cell parameters).
            - If individual parameters, they should be in the order:
              a, b, c, alpha, beta, gamma.

        Raises
        ------
        ValueError
            If the provided arguments do not match the expected formats.
        TypeError
            If the provided arguments are not of the expected types.
        """
        # If only one object was provided ...
        if len(args) == 1:
            # ... check if it's a NumPy array (It might be a 3x3 matrix)
            if isinstance(args[0], np.ndarray):
                if args[0].shape != (3, 3):
                    raise ValueError(
                        "Cell.__init__() The vector must be a "
                        "3x3 numpy array."
                    )
                else:
                    self._vector = args[0]
                    self.__cell_params()
            # ... check if it's a list. In this case it might be:
            #     - A list of three 3D NumPy arrays
            #     - A list of all 6 cell parameters
            #     - A list with all 9 entries of the transformation matrix
            elif isinstance(args[0], list):
                if (len(args[0]) == 3) and \
                    all([isinstance(i, (list, np.ndarray)) for i in args[0]]):
                    temp_vector = np.array(args[0], dtype=float)
                    if temp_vector.shape != (3, 3):
                        raise ValueError(
                            "Cell.__init__() The vector must be a "
                            "3x3 numpy array."
                        )
                    else:
                        self._vector = temp_vector
                        self.__cell_params()
                elif len(args[0]) == 6 and \
                    all([isinstance(i, (int, float)) for i in args[0]]):
                    self._a  = args[0][0]
                    self._b  = args[0][1]
                    self._c  = args[0][2]
                    self._alpha = args[0][3]
                    self._beta  = args[0][4]
                    self._gamma = args[0][5]
                    self.__cell_vector()
                elif (len(args[0]) == 9) and \
                    all([isinstance(i, (int, float)) for i in args[0]]):
                    temp_vector = np.array(args[0], dtype=float).reshape((3, 3))
                    if temp_vector.shape != (3, 3):
                        raise ValueError(
                            "Cell.__init__() The vector must be a "
                            "list with 9 entries."
                        )
                    else:
                        self._vector = temp_vector
                        self.__cell_params()
                else:
                    raise ValueError(
                        "Cell.__init__() The cell must be initialized "
                        "with either a 3x3 numpy array, a list of 3 lists, "
                        "a list of 6 values, or a list of 9 values."
                    )
            # ... check if it's a dictionary. In this case it should contain
            # the 6 cell parameters
            elif isinstance(args[0], dict):
                # Check if the dictionary has the correct keys and
                # the correct entries
                if len(args[0]) != 6:
                    raise ValueError(
                        "Cell.__init__() The dictionary must contain exactly "
                        "6 entries for the cell parameters."
                    )
                params = set(["a", "b", "c", "alpha", "beta", "gamma"])
                if not all(key in params for key in args[0]):
                    raise ValueError(
                        "Cell.__init__() The dictionary must contain only the "
                        "following keys: a, b, c, alpha, beta, gamma."
                    )
                # Extract the cell parameters from the dictionary and calculate
                # the cell vector
                self._a = args[0]["a"]
                self._b = args[0]["b"]
                self._c = args[0]["c"]
                self._alpha = args[0]["alpha"]
                self._beta = args[0]["beta"]
                self._gamma = args[0]["gamma"]
                self.__cell_vector()
            else:
                raise TypeError(
                    "Cell.__init__() The cell vector must be a "
                    "numpy array."
                )
        # If 3 objects were provided, and ll of them are NumPy arrays ...
        # ... those might be the cell vectors
        elif (len(args) == 3) and \
            all([isinstance(i, (list, np.ndarray)) for i in args]):
            temp_vector = np.array(args, dtype=float)
            if temp_vector.shape != (3, 3):
                raise ValueError(
                    "Cell.__init__() The cell vector must be a 3x3 "
                    "numpy array."
                )
            else:
                self._vector = temp_vector
                self.__cell_params()
        # If 6 objects were provided, and all of them are ints or floats ...
        # ... those might be the individual cell parameters
        elif (len(args) == 6) and \
            all([isinstance(i, (int, float)) for i in args]):
            self._a  = args[0]
            self._b  = args[1]
            self._c  = args[2]
            self._alpha = args[3]
            self._beta  = args[4]
            self._gamma = args[5]
            self.__cell_vector()
        # If 9 objects were provided, and all of them are ints or floats ...
        # ... those might be the entries of the cell vectors
        elif (len(args) == 9) and \
            all([isinstance(i, (int, float)) for i in args]):
            self._vector = np.array(args, dtype=float).reshape((3, 3))
            self.__cell_params()
        else:
            raise ValueError(
                "Cell.__init__() The cell must be initialized with "
                "either a 3x3 numpy array, a list of 3 or 6 values, or 6 "
                "individual parameters.")

        # Reset the cell type
        self._type = None

        # Retrieve the valid cell types
        self.__retrieve_types()

        # Try to find the cell type
        try:
            self.find_type(set_type=True)
        except ValueError:
            warn("Cell.__init__() Could not find a matching cell type "
                 "for the provided parameters. The cell type will be "
                 "set to None.")
    
    def __repr__(self):
        """
        Return a string representation of the Cell object.
        
        Returns
        -------
        str
            A string representation of the Cell object, showing the cell parameters.
        """
        return f"Cell(params={self.params})"

    def __retrieve_types(self):
        """ Retrieve the valid cell types and their active parameters

        Returns
        -------
        dict
            A dictionary containing the valid cell types and
            their active parameters
        """
        here = Path(globals().get("__file__", "./_")).absolute().parent
        ct_file_location = os.path.join(here, "data", "crystal_systems.json")

        with open(ct_file_location, "r") as f:
            cell_types = json.load(f)

        self.__cell_type_data = cell_types
        self.__cell_type_names = list(self.__cell_type_data.keys())

    def __check_fix_params(self,
                           cell_type : str,
                           params : dict,
                           tolerance : float = 1e-2) -> bool:
        """Check if the fixed parameters of a cell type match the
        provided parameters

        Parameters
        ----------
        cell_type : str
            The type of the crystal cell.
        params : dict
            A dictionary containing the cell parameters with keys:
            "a", "b", "c", "alpha", "beta", "gamma".
        tolerance : float, optional
            The tolerance for comparing fixed parameters.

        Returns
        -------
        bool
            True if the fixed parameters match, False otherwise.
        """
        fix_params = self.__cell_type_data[cell_type]["fix"]
        return all(abs(params[p] - v) < tolerance for p, v in fix_params.items())

    def __check_equal_params(self,
                             cell_type : str,
                             params : dict,
                             tolerance : float = 1e-2) -> bool:
        """ Check if the equal parameters of a cell type match the
        provided parameters

        Parameters
        ----------
        cell_type : str
            The type of the crystal cell.
        params : dict
            A dictionary containing the cell parameters with keys:
            "a", "b", "c", "alpha", "beta", "gamma".
        tolerance : float, optional
            The tolerance for comparing equal parameters.

        Returns
        -------
        bool
            True if the equal parameters match, False otherwise.
        """
        equal_params = self.__cell_type_data[cell_type]["equal"]
        are_equal = []
        for pair in equal_params:
            are_equal.append(abs(params[pair[0]] - params[pair[1]]) < tolerance)
        return all(are_equal)

    @property
    def vector(self):
        """Return the cell vector as a 3x3 numpy array.
        
        Returns
        -------
        numpy.ndarray
            A 3x3 numpy array representing the cell vector."""
        return self._vector
    
    @vector.setter
    def vector(self, new_vector : np.ndarray):
        """Set the cell vector and recalculate cell parameters.
        
        Parameters
        ----------
        new_vector : numpy.ndarray
            A 3x3 numpy array representing the new cell vector.
        
        Raises
        ------
        ValueError
            If the new vector is not a 3x3 numpy array."""
        if not isinstance(new_vector, np.ndarray) or new_vector.shape != (3, 3):
            raise ValueError("Cell vector must be a 3x3 numpy array.")
        self._vector = new_vector
        self.__cell_params()
    
    @property
    def params(self):
        """Return the cell parameters as a dictionary.
        
        Returns
        -------
        dict
            A dictionary containing the cell parameters with keys:
            "a", "b", "c", "alpha", "beta", "gamma".
        """
        return {
            "a": self._a,
            "b": self._b,
            "c": self._c,
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma
        }
    
    @params.setter
    def params(self, new_params : dict | list):
        """Set the cell parameters and recalculate the cell vector.
        
        Parameters
        ----------
        new_params : dict or list
            - If a dictionary, it should contain keys:
              "a", "b", "c", "alpha", "beta", "gamma".
            - If a list, it should contain 6 values in the order:
              a, b, c, alpha, beta, gamma.
        Raises
        ------
        ValueError
            If the new parameters are not in the expected format.
        """
        if isinstance(new_params, dict) and len(new_params) == 6:
            required_keys = {"a", "b", "c", "alpha", "beta", "gamma"}
            if not required_keys.issubset(new_params.keys()):
                raise ValueError(f"Cell parameters must contain keys: {required_keys}")
            self._a = new_params["a"]
            self._b = new_params["b"]
            self._c = new_params["c"]
            self._alpha = new_params["alpha"]
            self._beta = new_params["beta"]
            self._gamma = new_params["gamma"]
            self.__cell_vector()
        
        elif isinstance(new_params, list) and len(new_params) == 6:
            self._a  = new_params[0]
            self._b  = new_params[1]
            self._c  = new_params[2]
            self._alpha = new_params[3]
            self._beta  = new_params[4]
            self._gamma = new_params[5]
            self.__cell_vector()
        else:
            raise ValueError("Cell parameters must be a dictionary or a list of 6 values.")

    @property
    def active(self):
        """Return the active cell parameters as a dictionary, according to
        the cell type
        
        Returns
        -------
        dict
            A dictionary containing the active cell parameters.
        """
        active_params = self.__cell_type_data[self._type]["active"]
        return {p: self.params[p] for p in active_params}
    
    @property
    def volume(self):
        """Calculate and return the volume of the cell.

        Returns
        -------
        float
            The volume of the cell calculated using the formula:
            V = |a . (b x c)|, where a, b, c are the cell vectors.
        """
        return np.abs(
            np.dot(self._vector[:, 0],
                   np.cross(self._vector[:, 1], self._vector[:, 2]))
        )

    @property
    def type(self):
        """ Get the type of the crystal cell.

        Returns
        -------
        str
            The type of the crystal cell.
        """
        return self._type

    @type.setter
    def type(self, value : str):
        """ Set the type of the crystal cell.

        Parameters
        ----------
        value : str
            The new type of the crystal cell.

        Raises
        ------
        ValueError
            If an invalid cell type is provided.
        """
        if value not in self.__cell_type_names:
            raise ValueError(
                f"Invalid cell type '{value}'. "
                f"Valid types are: {', '.join(self.__cell_type_names)}."
            )
        self._type = value

    def find_type(self,
                  set_type : bool = True,
                  tolerance : float = 1e-2) -> list[dict]:
        """ Determine the type of a crystal cell based on its parameters.

        Determine the most likely crystal type(s) for a full set of cell
        parameters (a, b, c, alpha, beta, gamma), ranked from most to least
        symmetric (fewest to most active/independent parameters).

        Because higher-symmetry systems are special cases of lower-symmetry
        ones (cubic subset of orthorhombic subset of monoclinic subset of
        triclinic), several systems can match simultaneously. The
        result is a ranked list so the caller can pick the most specific
        (highest-symmetry) match; usually ranked[0].

        Parameters
        ----------
        set_type : bool, optional
            If True, set the type of the cell to the determined type.
        tolerance : float, optional
            The tolerance for comparing equal parameters.

        Returns
        -------
        list of dict
            A list of dictionaries, where each dictionary contains:
            - type : The type of the cell.
            - num_active : The number of active parameters.

        Raises
        ------
        ValueError
            If the provided parameters do not match any known cell type
        """
        matches = []
        for cell_type, cell_data in self.__cell_type_data.items():
            active_params = cell_data["active"]
            fix_params = cell_data["fix"]
            equal_params = cell_data["equal"]

            # Check if all fixed/equal parameters have the correct values
            if self.__check_fix_params(cell_type, self.params, tolerance) and \
               self.__check_equal_params(cell_type, self.params, tolerance):
                matches.append({
                    "type": cell_type,
                    "num_active": len(active_params)
                })

        # Sort matches by number of active parameters (ascending)
        matches.sort(key=lambda x: x["num_active"])

        if not matches:
            raise ValueError("Cell.find_type(): Could not find a matching "
                             "cell type")

        # Set the type of the cell to the most specific match
        if set_type:
            self.type = matches[0]["type"]

        return matches
    
    def reciprocal(self):
        """Calculate and return the reciprocal cell vector.
        
        Returns
        -------
        dict
            A dictionary containing the reciprocal cell vectors with keys:
            "u", "v", "w".
        
        Raises
        ------
        ValueError
            If the cell vectors are not defined.

        Notes
        -----
        The reciprocal cell vectors are calculated using the formula:
        u = (b x c) / V, v = (c x a) / V, w = (a x b) / V,
        where V is the volume of the cell.
        """
        if not hasattr(self, "_vector"):
            raise ValueError("Cell vectors are not defined.")
        
        a1 = self._vector[:, 1]
        a2 = self._vector[:, 2]
        a3 = self._vector[:, 0]

        volume = self.volume

        reci = {}
        reci['u'] = np.cross(a2, a3) / volume
        reci['v'] = np.cross(a3, a1) / volume
        reci['w'] = np.cross(a1, a2) / volume

        return reci

    def __cell_params(self):
        """Calculate the cell parameters from the cell vector.
        
        This method calculates the lengths of the cell vectors (a, b, c)
        and the angles between them (alpha, beta, gamma).
        
        Notes
        -----
        The angles are calculated using the dot product and the cosine
        formula:
        cos(theta) = (b . c) / (|b| * |c|), where theta is the angle
        between vectors b and c, and |b|, |c| are their magnitudes."""

        self._a = np.linalg.norm(self._vector[:, 0])
        self._b = np.linalg.norm(self._vector[:, 1])
        self._c = np.linalg.norm(self._vector[:, 2])

        self._alpha = np.degrees(np.arccos(
            np.dot(self._vector[:, 1], self._vector[:, 2]) /
            (self._b * self._c)
        ))
        self._beta = np.degrees(np.arccos(
            np.dot(self._vector[:, 0], self._vector[:, 2]) /
            (self._a * self._c)
        ))
        self._gamma = np.degrees(np.arccos(
            np.dot(self._vector[:, 0], self._vector[:, 1]) /
            (self._a * self._b)
        ))

    def __cell_vector(self):
        """
        Calculate the cell vector from the cell parameters.

        This method calculates the cell vector based on the lengths
        (a, b, c) and angles (alpha, beta, gamma) of the cell.

        Notes
        -----
        The cell vector is calculated using the following formulas:
        a = [a, 0, 0]
        b = [b * cos(gamma), b * sin(gamma), 0]
        c = [
            c * cos(beta),
            c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma),
            c * np.sqrt(
                    1 -
                    cos(beta)**2 -
                    ((cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma))**2
                )
        ]
        where a, b, c are the lengths of the cell vectors and alpha,
        beta, gamma are the angles between them in degrees.
        The angles are converted to radians for the calculations.
        """
        sg = np.sin(np.radians(self._gamma))

        ca = np.cos(np.radians(self._alpha))
        cb = np.cos(np.radians(self._beta))
        cg = np.cos(np.radians(self._gamma))

        va = self._a * \
            np.array([
                1,
                0,
                0
            ])
        
        vb = self._b * \
            np.array([
                cg,
                sg,
                0
            ])
        
        vc = self._c * \
            np.array([
                cb,
                (ca - cb * cg) / sg,
                np.sqrt(1 - cb**2 - ((ca - cb * cg) / sg)**2)
            ])

        self._vector = np.array([va, vb, vc]).T

# ------------------------------------------------------- #
#              The Symmetry Operations Class              #
# ------------------------------------------------------- #

class SymmetryOperations:
    """ Class to handle symmetry operations in crystals.
    
    This class allows adding, retrieving, and managing symmetry operations
    for crystal structures. It can also identify the space group based on
    the provided symmetry operations.
    
    Attributes
    ----------
    SPACE_GROUPS : list
        A list of dictionaries containing space group information.
    _ops : list
        A list of symmetry operations, each represented as a dictionary
        with keys "matrix" and "vector".
    
    Methods
    -------
    __init__() : None
        Initializes the SymmetryOperations object and loads space group data.
    __getitem__(given) : dict or list
        Retrieves a symmetry operation or a list of operations based on
        the provided index or slice.
    __setitem__(given, data) : None
        Sets a symmetry operation or a list of operations based on
        the provided index or slice.
    __len__() : int
        Returns the number of symmetry operations.
    __repr__() : str
        Returns a string representation of the symmetry operations.
    __str__() : str
        Returns a string representation of the symmetry operations.
    __as_cif() : str
        Returns a string representation of the symmetry operations
        in CIF format.
    operations : list
        Property to get the list of symmetry operations.
    add(op) : None
        Adds a new symmetry operation to the list.
    __parse_op(op) : dict
        Parses a symmetry operation string and converts it to a dictionary
        with keys "matrix" and "vector".
    __load_space_groups(file_path) : list
        Loads space group data from a JSON file.
    _symop_as_txt(symop) : str
        Converts a symmetry operation dictionary to its string representation.
    get_sg_from_symops() : dict
        Identifies and returns the space group based on the current
        symmetry operations.
    symops_to_sg(symops) : int
        Class method to identify the space group based on a list of
        symmetry operations.
    
    Raises
    ------
    ValueError
        If the provided symmetry operation string is invalid or if the
        provided symmetry operations do not match any standard space group.
    TypeError
        If the provided arguments are not of the expected types.
    IndexError
        If the number of items being set does not match the number of
        items being retrieved.
    """

    SPACE_GROUPS = None

    def __init__(self):
        """ Initialize the SymmetryOperations object.
        
        This method initializes the SymmetryOperations object and loads
        space group data from a JSON file.
        
        Raises
        ------
        FileNotFoundError
            If the space group data file is not found or cannot be loaded."""
        here = Path(globals().get("__file__", "./_")).absolute().parent
        sg_file_location = os.path.join(here, "data", "space_groups.json")

        self._ops = []

        if self.SPACE_GROUPS is None:
            SymmetryOperations.SPACE_GROUPS = self.__load_space_groups(
                                                        sg_file_location
                                                )
    
    def __getitem__(self, given : int | slice) -> dict | list:
        """ Retrieve a symmetry operation or a list of operations.

        Parameters
        ----------
        given : int or slice
            - If an integer, it retrieves the symmetry operation at that index.
            - If a slice, it retrieves a list of symmetry operations
              defined by the slice.

        Returns
        -------
        dict or list
            - If an integer is provided, it returns a dictionary representing
            the symmetry operation at that index.
            - If a slice is provided, it returns a list of dictionaries
            representing the symmetry operations defined by the slice.
        """
        if isinstance(given, int):
            return self._ops[given]
        elif isinstance(given, slice):
            return self._ops[ given.start : given.stop : given.step ]
        else:
            raise TypeError(
                    f"SymmetryOperations.__getitem__() The argument {given} "
                    "is not an integer or a slice object."
            )
    
    def __setitem__(self, given : int | slice, data : str | list | tuple):
        """ Set a symmetry operation or a list of operations.

        Parameters
        ----------
        given : int or slice
            - If an integer, it sets the symmetry operation at that index.
            - If a slice, it sets a list of symmetry operations defined by
              the slice.
        data : str or list or tuple
            - If an integer is provided, data should be a string representing
              the symmetry operation to be set at that index.
            - If a slice is provided, data should be a list or tuple of
              strings representing the symmetry operations to be set
              defined by the slice.
        
        Raises
        ------
        ValueError
            If the provided symmetry operation string is invalid or if the
            number of items being set does not match the number of items
            being retrieved.
        TypeError
            If the provided arguments are not of the expected types.
        IndexError
            If the number of items being set does not match the number of
            items being retrieved.
        """
        if isinstance(given, int):
            if isinstance(data, str):
                self._ops[given] = self.__parse_op(data)
            else:
                raise ValueError(
                        "SymmetryOperations.__setitem__() Can not assign "
                        "values from a list or tuple to an individual entry."
                )
        elif isinstance(given, slice):
            if isinstance(data, list) or isinstance(data, tuple):
                i = given.start if given.start != None else 0
                o = given.stop
                e = given.step if given.step != None else 1
                if len(data) == len(self._ops[i : o : e]):
                    for idx, new_op in enumerate(range(i, o, e)):
                        if isinstance(data[idx], str):
                            self._ops[new_op] = self.__parse_op(data[idx])
                        else:
                            raise ValueError(
                                    "SymmetryOperations.__setitem__() Can "
                                    "not assign values from a list or tuple "
                                    "to an individual entry."
                            )
                else:
                        raise IndexError(
                                "SymmetryOperations.__setitem__() The number "
                                "of items being set does not match the number"
                                " items being retreived."
                        )
        else:
            raise TypeError(
                    f"SymmetryOperations.__setitem__() The argument {given} "
                    "is not an integer or a slice object."
            )
    
    def __len__(self):
        """ Return the number of symmetry operations.

        Returns
        -------
        int
            The number of symmetry operations currently stored in the object.
        """
        return len(self._ops)

    def __repr__(self):
        """ Return a string representation of the symmetry operations.
        
        Returns
        -------
        str
            A string representation of the symmetry operations in CIF format.
        """
        return self.__as_cif()
    
    def __str__(self):
        """ Return a string representation of the symmetry operations.
        
        Returns
        -------
        str
            A string representation of the symmetry operations in CIF format.
        """
        return self.__as_cif()
    
    def __as_cif(self):
        """ Return a string representation of the symmetry operations in CIF format.

        Returns
        -------
        str
            A string representation of the symmetry operations in CIF format.
        """
        res = ""
        dim = "xyz"
        for i, o in enumerate(self._ops):
            so = SymmetryOperations._symop_as_txt(o)
            res += f"{i + 1} {so}\n"
        return res

    @property
    def operations(self):
        """ Return the list of symmetry operations.

        Returns
        -------
        list
            A list of dictionaries representing the symmetry operations,
            each with keys "matrix" and "vector".
        """
        return self._ops
    
    def add(self, op : str):
        """ Add a new symmetry operation to the list.

        Parameters
        ----------
        op : str
            A string representing the symmetry operation to be added.
        
        Raises
        ------
        ValueError
            If the provided symmetry operation string is invalid.
        """
        self._ops.append(self.__parse_op(op))
    
    def __parse_op(self, op : str):
        """ Parse a symmetry operation string and convert it to a dictionary.

        Parameters
        ----------
        op : str
            A string representing the symmetry operation to be parsed.

        Returns
        -------
        dict
            A dictionary representing the symmetry operation with keys
            "matrix" and "vector".

        Raises
        ------
        ValueError
            If the provided symmetry operation string is invalid.
        """
        if not isinstance(op, str):
            raise TypeError(
                "SymmetryOperations.add_operation() Expected a "
                f"string, but found a {type(op)}."
            )

        if len(op) < 7:
            raise ValueError(
                    f"SymmetryOperations.add_operation() {op} "
                    "is too small to be a symmetry operation."
            )
        
        # Convert to lower case ensuring consistency
        op = op.lower()
        
        try:
            first_level = op.split()
        except ValueError as e:
            raise ValueError(
                    "SymmetryOperations.add_operation() The format of "
                    f"{op} is invalid."
            )
        if len(first_level) == 1:
            individuals = first_level.split(',')
        else:
            individuals = first_level[1].split(',')
        if len(individuals) != 3:
            raise ValueError(
                    "SymmetryOperations.add_operation() The format of "
                    f"{op} is invalid."
            )
        dim = "xyz"
        matrix = np.zeros((3,3))
        vector = [0.0, 0.0, 0.0]
        for i, o in enumerate(individuals):
            rem = o
            for j, q in enumerate(dim):
                if q in rem:
                    pos = rem.index(q)
                    if pos != 0:
                        matrix[i][j] = int(rem[pos - 1] + "1")
                        rem = rem.replace(rem[pos - 1:pos + 1], "")
                    else:
                        matrix[i][j] = 1.0
                        rem = rem.replace(q, "")
            frac = rem.split("/")
            if len(frac) > 2:
                raise ValueError(
                        f"SymmetryOperations.add_operation() {o} "
                        "is not a symmetry operation."
                )
            elif len(frac) == 2:
                vector[i] = float(frac[0]) / float(frac[1])
                if vector[i] < 0:
                    vector[i] += 1.0
            elif len(frac) == 1:
                if len(frac[0]) >= 1:
                    vector[i] = int(frac[0])
        
        return {"matrix" : matrix, "vector" : np.array(vector)}
    
    def __load_space_groups(self, file_path : str) -> list:
        """
        Load space group data from a JSON file.

        Parameters
        ----------
        file_path : str
            Path to the JSON file containing space group data.

        Returns
        -------
        list
            A list of dictionaries containing space group information.
        
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                    f"SymmetryOperations.__load_space_groups() The file "
                    f"{file_path} was not found."
            )
        with open(file_path, 'r') as file:
            space_groups = json.load(file)
        
        tot = len(space_groups)

        # Convert symmetry operations to numpy arrays and clean the identifiers
        for sg in range(tot):
            space_groups[sg]['ITC'] = sg
            if 'identifier' in space_groups[sg]:
                id = space_groups[sg]['identifier']
                id_size = len(id)
                if id_size > 2:
                    if id[1] == '1' and id[-1] == '1':
                        space_groups[sg]['identifier'] = id[0] + id[2:-1]
            if 'symops' in space_groups[sg]:
                space_groups[sg]['text'] = set([])
                for i, op in enumerate(space_groups[sg]['symops']):
                    space_groups[sg]['symops'][i] = {
                                    "matrix" : np.array(op[0]),
                                    "vector" : np.array(op[1])
                    }
                    so = SymmetryOperations._symop_as_txt(
                                                space_groups[sg]['symops'][i]
                    )
                    space_groups[sg]['symops'][i]["txt"] = so
                    space_groups[sg]['text'].add(so)

        return space_groups
    
    def get_sg_from_symops(self) -> dict:
        """ Identify and return the space group based on the current symops.
        
        Returns
        -------
        dict
            A dictionary containing the space group information if a match
            is found.
        
        Raises
        ------
        ValueError
            If the current symmetry operations do not match any standard
            space group.
        """
        idx = SymmetryOperations.symops_to_sg(self._ops)
        return SymmetryOperations.SPACE_GROUPS[idx]
    
    @classmethod
    def _symop_as_txt(cls, symop : dict) -> str:
        """ Convert a symmetry operation dictionary to its string representation.

        Parameters
        ----------
        symop : dict
            A dictionary representing a symmetry operation with keys
            "matrix" and "vector".
        
        Returns
        -------
        str
            A string representation of the symmetry operation.
        """
        so = {}
        dim = "xyz"
        for i, d in enumerate(dim):
            so[d] = ""
            for j, e in enumerate(dim):
                m = symop['matrix'][i,j]
                if m != 0:
                    if m > 0 and len(so[d]) != 0:
                        so[d] += '+'
                    so[d] += str(int(m)).replace('1', e)
            v = symop['vector'][i]
            if v != 0:
                temp = str(Fraction(Decimal(v)).limit_denominator())
                if len(so[d]) != 0 and so[d][0] != "-":
                    so[d] = temp + "+" + so[d]
                else:
                    so[d] = temp + so[d]
        return f"{so['x']},{so['y']},{so['z']}"
    
    @classmethod
    def symops_to_sg(cls, symops : list = []) -> int:
        """ Class method to identify the space group based on a list of symops.
        
        Parameters
        ----------
        symops : list
            A list of symmetry operations, each represented as a dictionary
            with keys "matrix" and "vector".
        
        Returns
        -------
        int
            The index of the matching space group in the SPACE_GROUPS list.
        
        Raises
        ------
        ValueError
            If no symmetry operations are provided or if the provided
            symmetry operations do not match any standard space group.
        """
        if len(symops) == 0:
            raise ValueError(
                    "SymmetryOperations.symops_to_sg() No symmetry "
                    "operations were provided!"
            )
        
        sos_txt = set([])
        for so in symops:
            sos_txt.add(cls._symop_as_txt(so))

        for i, sg in enumerate(cls.SPACE_GROUPS):
            if 'symops' in sg:
                if len(sg['symops']) == len(sos_txt):
                    if len(sg["text"].difference(sos_txt)) == 0:
                        return i

        raise ValueError(
                "SymmetryOperations.symops_to_sg() The provided symmetry "
                "operations do not match any standard space group."
        )

    @classmethod
    def get_sg_from_cell(
            cls,
            cell : Cell,
            positions : list,
            symbols : list,
            tolerance : float = 1E-5,
            full_output : bool = False) -> dict:
        """ Identify and return the space group based on the cell and atoms.

        Parameters
        ----------
        cell : Cell
            A Cell object representing the unit cell of the crystal structure.
        positions : list
            A list of numpy arrays of shape (1, 3) representing the fractional
            coordinates of N atoms in the RESOLVED unit cell.
        symbols : list
            A list of length N containing the chemical symbols of the atoms
            in the RESOLVED unit cell.
        tolerance : float, optional
            A float representing the tolerance for symmetry detection.
            Default is 1E-5.
        full_output : bool, optional
            A boolean stating if the complete output of the space group search
            should be included as a return value

        Returns
        -------
        dict
            A dictionary containing the space group information if a match
            is found.

        Raises
        ------
        ValueError
            If the provided cell, positions, or symbols are invalid or if
            no matching space group is found.

        Notes
        -----
        This method uses the spglib library to identify the space group based
        on the provided cell and atomic positions.
        """
        if not isinstance(cell, Cell):
            raise ValueError(
                    "SymmetryOperations.get_sg_from_cell() The provided "
                    "cell is not a Cell instance."
            )
        
        if not isinstance(positions, list) or len(positions) == 0:
            raise ValueError(
                    "SymmetryOperations.get_sg_from_cell() The provided "
                    "positions are not a valid list of positions or is empty."
            )
    
        atom_numbers = []
        for s in symbols:
            try:
                atom_nb = PTE[s].number
                atom_numbers.append(atom_nb)
            except KeyError:
                raise ValueError(
                        "SymmetryOperations.get_sg_from_cell() The provided "
                        f"symbol {s} is not a valid chemical symbols."
                )
        
        spg_cell = (
            cell.vector.T,
            np.array(positions),
            np.array(atom_numbers)
        )

        try:
            spg_info = spg.get_symmetry_dataset(spg_cell, symprec=tolerance)
            if spg_info is None:
                raise ValueError(
                        "SymmetryOperations.get_sg_from_cell() No space "
                        "group could be identified with the provided cell "
                        "and atoms."
                )
            found_sg = deepcopy(cls.SPACE_GROUPS[spg_info.number])

            if found_sg['identifier'] == spg_info.international:

                if np.linalg.norm(spg_info.transformation_matrix) != 1 or \
                    np.linalg.norm(spg_info.origin_shift) != 0:

                    if full_output:
                        warn("SymmetryOperations.get_sg_from_cell() "
                             "The space group in this crystal is transformed "
                             "(has a change of basis) and/or a shift in the "
                             "origin.\n"
                             "To perform the backwards transformation to the "
                             "original symmetry operations, consider the "
                             "following expression:\n\n"
                             "(W', w') = (P^{-1} * W * P,   "
                             "P^{-1} * [w + (W - I) * t])\n\n"
                             "Where:\n"
                             "I ... the identity matrix\n"
                             "W'... transformed symmetry rotation\n"
                             "w'... transformed symmetry translation\n"
                             "W ... standard symmetry rotation\n"
                             "w ... standard symmetry translation\n"
                             "P ... change of basis (transformation)\n"
                             "t ... shift of the origin\n\n"
                             "Finally, remember to shift translations so "
                             "that they lie between 0 and +1.\n\n"
                             "For more information, check out:\n"
                             "- https://www.cryst.ehu.es/cryst/help/"
                             "trmatrix_identify.html")

                        found_sg["transformation"] = \
                            spg_info.transformation_matrix
                        
                        found_sg["origin_shift"] = spg_info.origin_shift

                        found_sg["unique_atoms"] = \
                            set(spg_info.equivalent_atoms.tolist())
                    else:
                        warn("SymmetryOperations.get_sg_from_cell() "
                             "The space group in this crystal is transformed "
                             "(has a change of basis) and/or a shift in "
                             "origin.\nUse `full_output = True` to include "
                             "these transformations in the return dictionary.")
                    found_sg['symops'] = []
                    found_sg['text'] = set([])

                    nb_symops = spg_info.rotations.shape[0]
                    for i in range(nb_symops):
                        temp_symop = {
                            'matrix' : spg_info.rotations[i],
                            'vector' : spg_info.translations[i]
                        }
                        op = SymmetryOperations._symop_as_txt(temp_symop)
                        temp_symop['txt'] = op
                        found_sg['symops'].append(temp_symop)
                        found_sg['text'].add(op)

                return found_sg
            
            raise ValueError(
                    "SymmetryOperations.get_sg_from_cell() No space group "
                    "could be identified with the provided cell and atoms."
            )
        except Exception as e:
            raise ValueError(
                    "SymmetryOperations.get_sg_from_cell() An error "
                    "occurred while identifying the space group: "
                    f"{str(e)}"
            )

# ------------------------------------------------------- #
#                   The Crystal Class                     #
# ------------------------------------------------------- #

class Crystal:
    """ Class representing a crystal structure.
    
    This class allows the definition and manipulation of crystal structures,
    including setting the unit cell, adding atoms, and converting between
    fractional and Cartesian coordinates.
    
    Attributes
    ----------
    name : str
        The name of the crystal structure.
    cell : Cell
        The unit cell of the crystal structure.
    _labels : list
        List of atom labels in the crystal.
    _symbols : list
        List of atom symbols in the crystal.
    _frac : list
        List of fractional coordinates of atoms in the crystal.
    _cartesian : list
        List of Cartesian coordinates of atoms in the crystal.
    symops : SymmetryOperations
        Symmetry operations associated with the crystal structure.
    
    Methods
    -------
    __init__(nam='crystal') : None
        Initializes the Crystal object with a name and default unit cell.
    cell : Cell
        Property to get/set the unit cell of the crystal structure.
    fill_atoms(atoms, relative=False) : None
        Adds atoms to the crystal structure.
    __getitem__(idx) : list
        Retrieves an atom's information based on index or label.
    __setitem__(idx, data) : None
        Sets an atom's information based on index or label.
    __len__() : int
        Returns the number of atoms in the crystal structure.
    __str__() : str
        Returns a string representation of the crystal structure.
    labels : list
        Property to get the list of atom labels.
    symbols : list
        Property to get the list of atom symbols.
    cart_coords : list
        Property to get the list of Cartesian coordinates of atoms.
    rel_coords : list
        Property to get the list of fractional coordinates of atoms.
    rel_coords : list
        Property to get the list of fractional coordinates of atoms.
    __frac_to_cart() : None
        Converts fractional coordinates to Cartesian coordinates.
    __cart_to_frac() : None
        Converts Cartesian coordinates to fractional coordinates.
    find_symmetry(threshold) : dict
        Finds the crystal's space group and symmetry operations through
        spglib.
    read_cif(file_path) : None
        Reads a CIF file and populates the crystal structure.
    write_cif(file_path) : None
        Writes the crystal structure to a CIF file.
    as_cif_coords() : str
        Return the atom information in CIF format
    apply_symops(symops, wrap_in_cell) : None
        Applies symmetry operations to generate the full crystal structure.
    create_supercell(indices) : Crystal
        Applies the crystal's symmetry operations, creates a supercell
        based on the provided indices, and attempts to find the space group and
        symmetry operations of the new crystal.
    """

    FIELDS = [
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z"
    ]

    def __init__(self, nam : str = 'crystal'):
        """ Initialize the Crystal object.

        Parameters
        ----------
        nam : str, optional
            The name of the crystal structure. Default is 'crystal'.
        """
        self.name = nam
        self._cell = Cell(1, 1, 1, 90, 90, 90)
        self._labels = []
        self._symbols = []
        self._frac = []
        self._cartesian = []
        self.symops = SymmetryOperations()
        self.symops.add("1 x,y,z")
    
    @property
    def cell(self):
        """ Retrieves the cell object with its parameters
        
        Returns
        -------
        Cell
            A Cell object representing the unit cell of the crystal structure.
        """
        return self._cell

    @cell.setter
    def cell(self, cell : Cell) -> None:
        """ Set the unit cell of the crystal structure.

        Parameters
        ----------
        cell : Cell
            A Cell object representing the unit cell of the crystal structure.
        
        Raises
        ------
        TypeError
            If the provided object is not a Cell instance.
        """
        if not isinstance(cell, Cell):
            raise TypeError(
                    "Crystal.cell() The provided object is not a "
                    "cell.")
        
        if len(self._frac) == 0 and len(self._cartesian) == 0:
            self._cell = cell
        else:
            if len(self._frac) != 0:
                self.__frac_to_cart()
            self._cell = cell
            self.__cart_to_frac()
            
    def fill_atoms(self, atoms : list, relative : bool = False) -> None:
        """ Add atoms to the crystal structure.

        Parameters
        ----------
        atoms : list
            A list of atoms to be added to the crystal structure. Each atom
            should be represented as a list or tuple containing:
            - label (str): A unique identifier for the atom (e.g., "C1").
            - symbol (str): The chemical symbol of the atom (e.g., "C").
            - coordinates (numpy.ndarray): A 3D numpy array representing the
              atom's coordinates. If `relative` is True, these should be
              fractional coordinates; otherwise, they should be Cartesian
              coordinates.
        relative : bool, optional
            A boolean indicating whether the provided coordinates are
            fractional (True) or Cartesian (False). Default is False.
        
        Raises
        ------
        TypeError
            If the provided atoms are not in the expected format.
        ValueError
            If the provided atoms do not meet the required conditions.
        """
        if not isinstance(atoms, list):
            raise TypeError("Crystal.fill_atoms() atoms should be a list.")
        if len(atoms) == 0:
            raise ValueError(
                    "Crystal.fill_atoms() The provided list of "
                    "atoms is empty."
            )
        pre_labels = []
        for i, at in enumerate(atoms):
            if len(at) != 3:
                raise ValueError(
                        "Crystal.fill_atoms() Each entry in atoms should "
                        "contain 3 items"
                )
            if not isinstance(at[0], str) or \
                not isinstance(at[1], str) or \
                not isinstance(at[2], np.ndarray):
                raise ValueError(
                        "Crystal.fill_atoms() Each entry in atoms should "
                        "contain 3 items: a label, a symbol and the atom's "
                        f"relative coordinates. Entry {i+1} is {at}"
                )
            if at[1] not in at[0]:
                raise ValueError(
                        "Crystal.fill_atoms() Inconsistent label found for "
                        f"entry {i+1}. Label '{at[0]}' does not contain "
                        f"symbol {at[1]}"
                )
            pre_labels.append(at[0])
        label_diff = len(atoms) - len(set(pre_labels))
        if label_diff != 0:
            raise ValueError(
                    "Crystal.fill_atoms() Each atom requires a unique label. "
                    f"There are {label_diff} entries with equal labels."
            )
        for at in atoms:
            self._labels.append(at[0])
            self._symbols.append(at[1])
            if relative:
                self._frac.append(at[2])
            else:
                self._cartesian.append(at[2])
            
        if relative:
            self.__frac_to_cart()
        else:
            self.__cart_to_frac()
    
    def __getitem__(self, idx : int | slice | str):
        """ Retrieve an atom's information based on index or label.

        Parameters
        ----------
        idx : int, slice, or str
            - If an integer, it retrieves the atom at that index.
            - If a slice, it retrieves a list of atoms defined by the slice.
            - If a string, it retrieves the atom with the matching label.
        
        Returns
        -------
        list or list of lists
            - If an integer or string is provided, it returns a list
              containing the atom's label, symbol, and Cartesian coordinates.
            - If a slice is provided, it returns a list of lists, each
              containing the label, symbol, and Cartesian coordinates of
              the atoms defined by the slice.
        
        Raises
        ------
        TypeError
            If the provided argument is not an integer, slice, or string.
        ValueError
            If the provided label does not match any atom in the crystal.
        """
        if isinstance(idx, int):
            return [
                    self._labels[idx],
                    self._symbols[idx],
                    self._cartesian[idx]
            ]
        elif isinstance(idx, str):
            if idx not in self._labels:
                raise ValueError(
                        f"Crystal.__setitem__() The label {idx} is not within "
                        "the crystal labels."
                )
            ridx = self._labels.index(idx)
            return [
                    self._labels[ridx],
                    self._symbols[ridx],
                    self._cartesian[ridx]
            ]
        elif isinstance(idx, slice):
            piece = []
            i = idx.start if idx.start != None else 0
            o = idx.stop if idx.stop != None else len(self._labels)
            e = idx.step if idx.step != None else 1
            for j in range(i, o, e):
                piece.append(
                        [
                            self._labels[j],
                            self._symbols[j],
                            self._cartesian[j]
                        ]
                )
            return piece
        else:
            raise TypeError(
                    f"Crystal.__getitem__() The argument {idx} is not an "
                    "integer or a slice object."
            )

    def __setitem__(self, idx : int | slice | str, data : list):
        """ Set an atom's information based on index or label.

        Parameters
        ----------
        idx : int, slice, or str
            - If an integer, it sets the atom at that index.
            - If a slice, it sets a list of atoms defined by the slice.
            - If a string, it sets the atom with the matching label.
        data : list
            - If an integer or string is provided, data should be a list
              containing the atom's label, symbol, and Cartesian coordinates.
            - If a slice is provided, data should be a list of lists, each
              containing the label, symbol, and Cartesian coordinates of
              the atoms to be set defined by the slice.
        
        Raises
        ------
        TypeError
            If the provided arguments are not of the expected types.
        ValueError
            If the provided label does not match any atom in the crystal,
            or if the provided data does not meet the required conditions.
        """
        if not isinstance(data, list):
            raise TypeError(
                    f"Crystal.__setitem__() Expected a list, got {type(data)}"
            )
        if len(data) < 3:
            raise TypeError(
                    "Crystal.__setitem__() Wrong number of entries in the "
                    "list; should be 3"
            )
        if isinstance(idx, int):
            if isinstance(data[0], str) and \
                isinstance(data[1], str) and \
                isinstance(data[2], np.ndarray):
                if len(data[2]) != 3:
                    raise ValueError(
                            "Crystal.__setitem__() Expected a 3D vector, got "
                            f"an array with shape {data[2].shape}"
                    )
                self._labels[idx] = self._labels[idx].replace(
                                                self._symbols[idx],
                                                data[1]
                                    )
                self._symbols[idx] = data[1]
                self._cartesian[idx] = data[2]
                self.__cart_to_frac()
            else:
                raise TypeError(
                        "Crystal.__setitem__() The coordinate is not shaped "
                        "as a list of str, str, and np.ndarray."
                )
        elif isinstance(idx, str):
            if idx not in self._labels:
                raise ValueError(
                        f"Crystal.__setitem__() The label {idx} is not within "
                        "the crystal labels."
                )
            ridx = self._labels.index(idx)
            self[ridx] = data
        elif isinstance(idx, slice):
            i = idx.start if idx.start != None else 0
            o = idx.stop if idx.stop != None else len(data)
            e = idx.step if idx.step != None else 1

            for jdx, op in enumerate(range(i, o, e)):
                if not isinstance(data[jdx], list):
                    raise TypeError(
                            "Crystal.__setitem__() Expected a list, got "
                            f"{type(data[jdx])}"
                    )
                if not isinstance(data[jdx][0], str) or \
                    not isinstance(data[jdx][1], str) or \
                    not isinstance(data[jdx][2], np.ndarray):
                    raise TypeError(
                            f"Crystal.__setitem__() The coordinate {jdx}: "
                            f"{data[jdx]} is not shaped as a list of str, "
                            "str, and np.ndarray."
                    )
                if len(data[jdx][2]) != 3:
                    raise ValueError(
                            "Crystal.__setitem__() Expected a 3D vector, "
                            f"got an array with shape {data[jdx][2].shape}"
                    )
            
            for jdx, op in enumerate(range(i, o, e)):
                self._labels[op] = self._labels[op].replace(
                                            self._symbols[op],
                                            data[jdx][1]
                                    )
                self._symbols[op] = data[jdx][1]
                self._cartesian[op] = data[jdx][2]
            self.__cart_to_frac()
        else:
            raise TypeError(
                    f"Crystal.__setitem__() The argument {idx} "
                    "is not an integer or a slice object."
            )

    def __len__(self):
        """ Return the number of atoms in the crystal structure.

        Returns
        -------
        int
            The number of atoms currently stored in the crystal structure.
        """
        return len(self._cartesian)
    
    def __str__(self):
        """ Return a string representation of the crystal structure.

        Returns
        -------
        str
            A string representation of the crystal structure, including
            cell parameters, symmetry operations, and atom information.
        """
        max_lbl = max([len(lbl) for lbl in self._labels])

        try:
            sg = self.find_symmetry()
            sg_info = f"Space group: {sg['ITC']} :: {sg['identifier']}"
        except ValueError:
            sg_info = "Space group: Not identified"

        div = "-" * 120
        output = div + "\n"
        for k, v in self.cell.params.items():
            output += f"{k:<8}:{v:>14.6f}\n"
        output += div + "\n"
        output += str(self.cell.vector) + "\n"
        output += div + "\n"
        output += f"[{sg_info}]\n"
        output += str(self.symops)
        output += div + "\n"
        for i in range(len(self._cartesian)):
            output += f"{i})\t"
            output += f"{self._labels[i]:>{max_lbl}}\t"
            output += f"{self._symbols[i]}\t:: "
            output += f"{self._cartesian[i][0]:14.6f} "
            output += f"{self._cartesian[i][1]:14.6f} "
            output += f"{self._cartesian[i][2]:14.6f}\t:: "
            output += f"{self._frac[i][0]:14.6f} "
            output += f"{self._frac[i][1]:14.6f} "
            output += f"{self._frac[i][2]:14.6f}\n"
        output += div
        return output

    @property
    def labels(self):
        """ Return the list of atom labels.
        
        Returns
        -------
        list
            A list of strings representing the labels of the atoms in
            the crystal structure."""
        return self._labels
    
    @labels.setter
    def labels(self, atom_labels : list):
        """ Set the list of atom labels.
        
        Parameters
        ----------
        atom_labels : list
            A list of strings representing the labels of the atoms in
            the crystal structure.
        
        Raises
        ------
        TypeError
            If the provided atom labels are not a list or if any of
            the labels are not strings."""
        if not isinstance(atom_labels, list):
            raise TypeError(
                    "Crystal.labels() The provided atom "
                    "labels are not a list."
            )
        for i, c in enumerate(atom_labels):
            if not isinstance(c, str):
                raise TypeError(
                        f"Crystal.labels() The {i}. symbol "
                        "is not a string."
                )
        self._labels = atom_labels

    @property
    def symbols(self):
        """ Return the list of atom symbols.
        
        Returns
        -------
        list
            A list of strings representing the chemical symbols of the
            atoms in the crystal structure.
        """
        return self._symbols
    
    @symbols.setter
    def symbols(self, atom_symbols : list):
        """ Set the list of atom symbols.

        Parameters
        ----------
        atom_symbols : list
            A list of strings representing the chemical symbols of the
            atoms in the crystal structure.
        
        Raises
        ------
        TypeError
            If the provided atom symbols are not a list or if any of
            the symbols are not strings.
        """
        if not isinstance(atom_symbols, list):
            raise TypeError(
                    "Crystal.symbols() The provided atom "
                    "symbols are not a list."
            )
        for i, c in enumerate(atom_symbols):
            if not isinstance(c, str):
                raise TypeError(
                        f"Crystal.symbols() The {i}. symbol "
                        "is not a string."
                )
        self._symbols = atom_symbols
    
    @property
    def cart_coords(self):
        """ Return the list of Cartesian coordinates of atoms.
        
        Returns
        -------
        list
            A list of numpy arrays representing the Cartesian coordinates
            of the atoms in the crystal structure.
        """
        return self._cartesian
    
    @cart_coords.setter
    def cart_coords(self, cartesian_coordinates : list):
        """ Set the list of Cartesian coordinates of atoms.

        Parameters
        ----------
        cartesian_coordinates : list
            A list of numpy arrays representing the Cartesian coordinates
            of the atoms in the crystal structure.
        
        Raises
        ------
        TypeError
            If the provided Cartesian coordinates are not a list or if
            any of the coordinates are not numpy arrays.
        """
        if not isinstance(cartesian_coordinates, list):
            raise TypeError(
                    "Crystal.cart_coords() The provided cartesian "
                    "coordinates are not a list."
            )
        for i, c in enumerate(cartesian_coordinates):
            if not isinstance(c, np.ndarray):
                raise TypeError(
                        f"Crystal.cart_coords() The {i}. coordinate "
                        "is not a NumPy array."
                )
        self._cartesian = cartesian_coordinates.copy()
        self.__cart_to_frac()
    
    @property
    def rel_coords(self):
        """ Return the list of fractional coordinates of atoms.

        Returns
        -------
        list
            A list of numpy arrays representing the fractional coordinates
            of the atoms in the crystal structure.
        """
        return self._frac
    
    @rel_coords.setter
    def rel_coords(self, relative_coordinates : list):
        """ Set the list of fractional coordinates of atoms.

        Parameters
        ----------
        relative_coordinates : list
            A list of numpy arrays representing the fractional coordinates
            of the atoms in the crystal structure.
        
        Raises
        ------
        TypeError
            If the provided fractional coordinates are not a list or if
            any of the coordinates are not numpy arrays.
        """
        if not isinstance(relative_coordinates, list):
            raise TypeError(
                    "Crystal.rel_coords() The provided relative "
                    "coordinates are not a list."
            )
        for i, c in enumerate(relative_coordinates):
            if not isinstance(c, np.ndarray):
                raise TypeError(
                        f"Crystal.rel_coords() The {i}. coordinate "
                        "is not a NumPy array."
                )
        self._frac = relative_coordinates
        self.__frac_to_cart()
    
    def __frac_to_cart(self):
        """ Convert fractional coordinates to Cartesian coordinates.
        
        Raises
        ------
        ValueError
            If the unit cell has not been defined.
        """
        if self.cell == None:
            raise ValueError(
                    "Crystal.__frac_to_cart() The unit cell has not been "
                    "defined."
            )
        vec = self.cell.vector

        self._cartesian = []
        for a in self._frac:
            self._cartesian.append(vec @ a)
    
    def __cart_to_frac(self):
        """ Convert Cartesian coordinates to fractional coordinates.
        
        Raises
        ------
        ValueError
            If the unit cell has not been defined.
        """
        if self.cell == None:
            raise ValueError(
                    "Crystal.__cart_to_frac() The unit cell has not been "
                    "defined."
            )
        ivec = np.linalg.inv(self.cell.vector)

        self._frac = []
        for a in self._cartesian:
            self._frac.append(ivec @ a)
    
    def find_symmetry(self, tolerance : float = 1E-5) -> dict:
        """ Find and return the space group of the crystal structure.

        Parameters
        ----------
        tolerance : float, optional
            A float representing the tolerance for symmetry detection.
            Default is 1E-5.

        Returns
        -------
        dict
            A dictionary containing the space group information if a match
            is found.
        Raises
        ------
        ValueError
            If the unit cell has not been defined or if no matching space
            group is found.
        """
        if self.cell == None:
            raise ValueError(
                    "Crystal.find_symmetry() The unit cell has not been "
                    "defined."
            )
        
        try:
            sg = self.symops.get_sg_from_symops()
        except ValueError as e:
            positions = []
            symbols = []

            for a in self.apply_symops():
                positions.append(a[2])
                symbols.append(a[1])

            sg = SymmetryOperations.get_sg_from_cell(
                    self.cell,
                    positions,
                    symbols
            )
        return sg
    
    def change_cell_axes(self, axes : str, reverse : bool = False) -> Self:
        """ Change the axes of the unit cell

        The rotation of the cell axes is performed by applying a
        transformation matrix to the cell vectors, and to to coordinates
        in relative space. The symmetry operations are also transformed
        accordingly, and the space group of the new crystal is identified
        based on the transformed symmetry operations.
        
        If the space group cannot be identified from the transformed
        symmetry operations, a supercell is created and the space group
        is identified from the transformed supercell.

        Parameters
        ----------
        axes : str
            A string representing the axes to be changed.
                - "ab" ... rotate the a and b axes around the c axis
                - "ac" ... rotate the a and c axes around the b axis
                - "bc" ... rotate the b and c axes around the a axis
        reverse : bool, optional
            A boolean indicating whether to reverse the axes. Default is False.
        
        Returns
        -------
        Crystal
            A new Crystal object with the changed cell axes and updated symmetry
            operations.
        
        Raises
        ------
        ValueError
            If the provided axes argument is not valid.
        """
        # Rotation matrices
        matrices = {
            'ab' : {
                'matrix' : np.array([[ 0,-1, 0],
                                     [ 1, 0, 0],
                                     [ 0, 0, 1]]),

                'reverse' : np.array([-1, -1, 1])
            },
            'ac' : {
                'matrix' : np.array([[ 0, 0,-1],
                                     [ 0, 1, 0],
                                     [ 1, 0, 0]]),

                'reverse' : np.array([-1, 1, -1])
            },
            'bc' : {
                'matrix' : np.array([[ 1, 0, 0],
                                     [ 0, 0,-1],
                                     [ 0, 1, 0]]),

                'reverse' : np.array([1, -1, -1])
            }
        }
        
        # Check if the provided axes argument is valid
        if axes not in matrices:
            raise ValueError(
                    "Crystal.change_cell_axes() The provided axes argument "
                    f"'{axes}' is not valid. "
                    f"Expected one of: {list(matrices.keys())}"
            )
        
        # If reverse is True, apply the reverse transformation; otherwise, apply
        # the normal transformation
        if reverse:
            mat = np.diag(matrices[axes]['reverse']) @ matrices[axes]['matrix']
        else:
            mat = matrices[axes]['matrix']

        # Change the symmetry operations

        # Symmetry operations that don't change under the transformation:
        # identity and inversion
        no_change = ['x,y,z', '-x,-y,-z']

        # Initialize the new symmetry operations
        r_symops = SymmetryOperations()

        # Iterate over the original symmetry operations
        for idop, op in enumerate(self.symops._ops, 1):

            # Generate the text form of the original symmetry operation
            op_txt = SymmetryOperations._symop_as_txt(op)

            # Check that this symop is not one of the ones that don't change
            if op_txt in no_change:
                r_symops.add(f"{idop} {op_txt}")
                continue

            # Apply the transformation to the symmetry operation
            new_op = {
                'matrix' : mat @ op['matrix'] @ np.linalg.inv(mat),
                'vector' : mat @ op['vector']
            }

            # Generate the text form of the transformed symmetry operation
            parsed_op = SymmetryOperations._symop_as_txt(new_op)

            # Add the transformed symop to the new symmetry operations
            r_symops.add(f"{idop} {parsed_op}")
        
        try:
            # Find the space group of the transformed symmetry operations
            sg = r_symops.get_sg_from_symops()

        except ValueError:
            t_cry = self.create_supercell()

            # Change the cell axes
            t_vector = mat @ t_cry.cell.vector @ np.linalg.inv(mat)
            t_cell = Cell(t_vector)

            # Change the atom positions
            t_relative = []
            for a in t_cry.rel_coords:
                t_relative.append(mat @ a)
            
            # Find the space group of the transformed cell and positions
            sg = SymmetryOperations.get_sg_from_cell(
                    t_cell,
                    t_relative,
                    t_cry._symbols
            )

        # Create a new symmetry operations object with the
        # transformed symmetry operations
        new_symops = SymmetryOperations()
        for i, so in enumerate(sg["text"]):
            new_symops.add(f"{i + 1} {so}")

        # Change the cell axes
        r_vector = mat @ self.cell.vector @ np.linalg.inv(mat)
        new_cell = Cell(r_vector)

        # Change the atom positions
        r_relative = []
        for ida in range(len(self)):
            r_relative.append([self._labels[ida],
                               self._symbols[ida],
                               mat @ self.rel_coords[ida]])
        
        # Create a new crystal with the changed cell axes and updated symmetry
        # operations
        new_crystal = Crystal(self.name + f"_axes_{axes}")
        new_crystal.cell = new_cell
        new_crystal.symops = new_symops
        new_crystal.fill_atoms(r_relative, relative=True)

        return new_crystal

    def read_cif(self, file_name : str, accurate : bool = False) -> dict:
        """ Read a CIF file and populate the crystal structure.

        Parameters
        ----------
        file_name : str
            The path to the CIF file to be read.
        accurate : bool, optional
            Whether to use accurate parsing for decimal values, by default False

        Returns
        -------
        atom_info : dict
            A dictionary containing the atom information extracted from the
            CIF file, with keys corresponding to the atom site fields defined
            in the CIF file and values being lists of the corresponding data for
            each atom.

        Raises
        ------
        FileNotFoundError
            If the specified CIF file does not exist.
        ValueError
            If the CIF file is malformed or missing required information.
        """
        if not os.path.isfile(file_name):
            raise FileNotFoundError(
                    f"Crystal.read_cif() The file {file_name} was not found."
            )

        with open(file_name, 'r') as f:
            data = f.readlines()

        if accurate:
            clean_df = lambda t: float(t.split()[1].split("(")[0])
            clean_af = lambda t: float(t.split("(")[0])
        else:
            clean_df = lambda t: float(t.split()[1].replace(
                                                    "(",
                                                    "").replace(
                                                            ")",
                                                            ""))
            clean_af = lambda t: float(t.replace(
                                            "(",
                                            "").replace(
                                                    ")",
                                                    ""))

        positions = []
        symops = SymmetryOperations()
        atom_sequence = []
        coords_active = False
        atom_info = {atm_inf : [] for atm_inf in self.FIELDS}
        for i, l in enumerate(data):

            # If the line is "#END", stop reading the file
            if "#END" in l:
                break

            # If the line is a comment, skip it
            if l[0] == "#":
                continue

            # If the line's length is 0 or has only one character
            # (whitespace), skip it and set the flag for reading
            # coordinates to False
            if len(l) in [0, 1]:
                coords_active = False
                continue

            # If the line starts with "loop_", set the flag for reading
            # coordinates to False, and reset the atom sequence list
            if "loop_" in l:
                coords_active = False
                atom_sequence = []
                continue

            # Lattic type
            if "_symmetry_cell_setting" in l:
                setting = l.split()[1]
                continue

            # Symmetry Operations
            if "_symmetry_equiv_pos_as_xyz" in l:
                pos = i
                while True:
                    pos += 1
                    if data[pos][0] == "_" or \
                        data[pos][0] not in "1234567890" or \
                        "loop_" in data[pos]:
                        break
                    else:
                        symops.add(data[pos])
                continue

            # Cell parameters
            if "_cell_length_a" in l:
                a = clean_df(l)
                continue
            
            if "_cell_length_b" in l:
                b = clean_df(l)
                continue
            
            if "_cell_length_c" in l:
                c = clean_df(l)
                continue
            
            if "_cell_angle_alpha" in l:
                aa = clean_df(l)
                continue
            
            if "_cell_angle_beta" in l:
                ba = clean_df(l)
                continue
            
            if "_cell_angle_gamma" in l:
                ga = clean_df(l)
                continue
        
            # Atoms and coordinates
            if l.startswith("_atom_site_"):
                coords_active = True
                atom_sequence.append(l.strip())
                atom_info[l.strip()] = []
                continue
            
            if coords_active:
                if l[0] in ascii_uppercase:
                    temp = l.split()

                    if len(temp) != len(atom_sequence):
                        raise ValueError(
                                "Crystal.read_cif() The number of entries in "
                                f"the atom line '{l}' does not match  the "
                                "number of atom site fields defined in the "
                                f"CIF file: {atom_sequence}"
                        )
                    
                    for k, d in zip(atom_sequence, temp):
                        atom_info[k].append(d)
                    
                    continue

        # Clean up the data by removing any parentheses and converting
        # to the appropriate types
        for kai in atom_info.keys():
            try:
                atom_info[kai] = [clean_af(t) for t in atom_info[kai]]
            except ValueError:
                pass
        
        # Pack the atomic positions as 3D vectors
        positions = zip(atom_info["_atom_site_fract_x"],
                        atom_info["_atom_site_fract_y"],
                        atom_info["_atom_site_fract_z"])

        # Create the crystal structure
        self.cell = Cell(a, b, c, aa, ba, ga)
        self._labels = atom_info["_atom_site_label"].copy()
        self._symbols = atom_info["_atom_site_type_symbol"].copy()
        self._frac = [np.array(p) for p in positions]
        self.symops = symops
        self.__frac_to_cart()

        return atom_info
    
    def write_cif(
            self,
            file_name : str = "",
            sg_data : dict = {},
            extra_atom_fields : dict = {}
        ) -> None:
        """ Write the crystal structure to a CIF file.

        Parameters
        ----------
        file_name : str, optional
            The path to the CIF file to be written. If not provided,
            the file will be named after the crystal's name attribute
            with a ".cif" extension. Default is an empty string.
        sg_data : dict, optional
            A dictionary containing space group information with keys:
            - "ITC": International Tables for Crystallography number (int)
            - "identifier": Hermann-Mauguin symbol (str)
            - "system": Crystal system (str)
            If not provided, the space group will be determined from the
            current symmetry operations. Default is an empty dictionary.
        extra_atom_fields : dict, optional
            A dictionary containing additional atom fields to be written
            to the CIF file. Default is an empty dictionary.
        
        Raises
        ------
        ValueError
            If the provided space group data is missing required fields.
        """
        if len(file_name) == 0:
            file_name = self.name + ".cif"
        
        params = self.cell.params

        if len(sg_data) == 0:
            sg = self.find_symmetry()
        else:
            ks = sg_data.keys()
            if "ITC" not in ks or "identifier" not in ks or "system" not in ks:
                raise ValueError(
                        "Crystal.write_cif() The provided space group data "
                        "does not contain the necessary fields: ITC, "
                        "identifier, system."
                )
            sg = sg_data

        div = "#" * 71
        output = div + "\n"
        output += f"#{'InformalCrystal':^69}\n"
        output += div + "\n"

        output += f"\ndata_{self.name}\n"
        output += f"_cell_length_a {params['a']:.10f}\n"
        output += f"_cell_length_b {params['b']:.10f}\n"
        output += f"_cell_length_c {params['c']:.10f}\n"
        output += f"_cell_angle_alpha {params['alpha']:.10f}\n"
        output += f"_cell_angle_beta {params['beta']:.10f}\n"
        output += f"_cell_angle_gamma {params['gamma']:.10f}\n"
        output += f"_cell_volume {self.cell.volume:.10f}\n"
        output += f"_symmetry_cell_setting {sg['system']}\n"
        output += f"_symmetry_space_group_name_H-M '{sg['identifier']}'\n"
        output += f"_symmetry_Int_Tables_number {sg['ITC']}\n"

        output += "loop_\n"
        output += "_symmetry_equiv_pos_site_id\n"
        output += "_symmetry_equiv_pos_as_xyz\n"
        output += str(self.symops)

        output += "loop_\n"
        output += "_atom_site_label\n"
        output += "_atom_site_type_symbol\n"
        output += "_atom_site_fract_x\n"
        output += "_atom_site_fract_y\n"
        output += "_atom_site_fract_z\n"

        if len(extra_atom_fields) != 0:
            for keaf in extra_atom_fields.keys():
                if keaf not in self.FIELDS:
                    output += keaf + "\n"
        
        output += self.as_cif_coords(extra_atom_fields)
        output += "#END\n"
    
        with open(file_name, 'w') as f:
            f.write(output)

    def as_cif_coords(self, extra_fields : dict = {}) -> str:
        """ Return the atom information in CIF format.

        Parameters
        ----------
        extra_fields : dict, optional
            A dictionary containing additional atom fields to be written
            to the CIF file. Default is an empty dictionary.

        Returns
        -------
        str
            A string containing the atom information formatted for CIF files,
            including labels, symbols, and fractional coordinates.
        """
        output = ""
        for i in range(len(self)):
            output += f"{self._labels[i]} {self._symbols[i]} "
            output += f"{self._frac[i][0]:.10f} "
            output += f"{self._frac[i][1]:.10f} "
            output += f"{self._frac[i][2]:.10f}"

            if len(extra_fields) != 0:
                for field in extra_fields.values():
                    if field not in self.FIELDS:
                        output += " " + str(field[i])

            output += "\n"

        return output
    
    def apply_symops(self,
                     symops : list = [],
                     wrap_in_cell : bool = False) -> list:
        """ Apply symmetry operations to generate the full crystal structure.

        Parameters
        ----------
        symops : list, optional
            A list of symmetry operation indices to be applied. If empty,
            all available symmetry operations will be used. Default is an
            empty list.
        wrap_in_cell : bool, optional
            A boolean indicating whether to wrap the resulting fractional
            coordinates within the unit cell (i.e., between 0 and 1). Default
            is False.
        
        Returns
        -------
        list
            A list of lists, each containing the label, symbol, and
            fractional coordinates of the atoms after applying the symmetry
            operations.
        
        Raises
        ------
        TypeError
            If the provided symops argument is not a list.
        """
        if not isinstance(symops, list):
            raise TypeError(
                    "Crystal.apply_symops() Requires a list of operations"
            )

        if len(symops) == 0:
            symops = len(self.symops)
        
        resolved_atoms = []
        for so in range(symops):
            op = self.symops[so]
            for a in range(len(self)):
                if "_" in self._labels[a]:
                    temp_label = self._labels[a].split("_")[0]
                else:
                    temp_label = self._labels[a]
                resolved_atoms.append([
                        f"{temp_label}_{so}",
                        self._symbols[a],
                        op['vector'] + op['matrix'] @ self.rel_coords[a]
                    ])
                if wrap_in_cell:
                    resolved_atoms[-1][2] %= 1
                resolved_atoms[-1][2] = np.round(resolved_atoms[-1][2], 12)
        return resolved_atoms
    
    def map_to_unit_cell(self, wrap_in_cell : bool = True) -> None:
        """ Map all atoms to the unit cell and reset the symmetry operations

        Parameters
        ----------
        wrap_in_cell : bool, optional
            A boolean indicating whether to wrap the resulting fractional
            coordinates within the unit cell (i.e., between 0 and 1). Default
            is True.
        """
        # Get the atomic positions after applying all symmetry ops
        new_atoms = self.apply_symops(wrap_in_cell=wrap_in_cell)

        # Reset symmetry operations
        new_symops = SymmetryOperations()
        new_symops.add("1 x,y,z")
        
        # Fill each list individually
        new_frac = []
        new_symbols = []
        new_labels = []
        for atom in new_atoms:
            new_labels.append(atom[0])
            new_symbols.append(atom[1])
            new_frac.append(atom[2])
        
        # Update the lists in the object
        self._labels = deepcopy(new_labels)
        self._symbols = deepcopy(new_symbols)
        self._frac = deepcopy(new_frac)

        # Update the cartesian coordinates
        self.__frac_to_cart()

        # Update the symmetry operations
        self.symops = new_symops

    @classmethod
    def __find_asu(
            cls,
            cell : Cell,
            atoms : list
        ) -> Self:
        """ Find the asymmetric unit of a crystal structure.

        Parameters
        ----------
        cell : Cell
            The unit cell of the crystal structure.
        atoms : list
            A list of lists, each containing the label, symbol, and
            fractional coordinates of the atoms in the crystal structure.

        Returns
        -------
        Crystal
            The asymmetric unit of the crystal structure.
        """
        # Create the new crystal supercell
        new_crystal = Crystal('supercell')

        cell_params = cell.params

        # Set the updated cell parameters
        new_crystal.cell = Cell(
            cell_params["a"],
            cell_params["b"],
            cell_params["c"],
            cell_params["alpha"],
            cell_params["beta"],
            cell_params["gamma"]
        )

        lbls_only = []
        sbls_only = []
        psts_only = []
        for a in atoms:
            sbls_only.append(a[1])
            psts_only.append(a[2])
            lbls_only.append(a[0].split("_")[0])

        # Try to find the space group and symmetry operations of
        # the new cell
        new_sym = {}
        try:
            new_sym = SymmetryOperations.get_sg_from_cell(
                new_crystal.cell,
                psts_only,
                sbls_only,
                full_output = True
            )
        except ValueError as e:
            warn("Crystal.__find_asu() WARNING! No space group "
                 "could be found for the supercell. Continuing with P1.")

            # Add the atoms to the cell
            new_crystal.fill_atoms(deepcopy(atoms), relative=True)

            return new_crystal

        # If the space group and symmetry could be found, re-order the atoms
        # and add the symmetry operations
        if "unique_atoms" in new_sym:
            new_atoms = []
            for a in new_sym["unique_atoms"]:
                new_atoms.append([
                    lbls_only[a],
                    sbls_only[a],
                    psts_only[a]
                ])
            new_symops = SymmetryOperations()
            for i, so in enumerate(new_sym["text"]):
                new_symops.add(f"{i + 1} {so}")
            new_crystal.symops = new_symops
        
        else:
            warn("Crystal.__find_asu() WARNING! No asymmetric unit "
                 "could be found for the supercell. Continuing with P1.")

            new_atoms = deepcopy(atoms)

        # Add the atoms to the cell
        new_crystal.fill_atoms(new_atoms, relative=True)

        return new_crystal

    def reduce_to_asu(self) -> None:
        """ Reduce the crystal structure to its asymmetric unit."""
        # Get all the current atoms in the crystal
        my_atoms = []

        for i in range(len(self)):
            my_atoms.append([
                            self._labels[i],
                            self._symbols[i],
                            self._frac[i]
                        ])
        
        # Find the asymmetric unit and the new symmetry operations
        new_crystal = Crystal.__find_asu(self.cell, my_atoms)

        # If the asymmetric unit has a different number of atoms than the
        # original crystal, update the atom lists and symmetry operations
        if len(new_crystal) != len(self):
            self._cartesian = []
            self._frac = []
            self.symops = deepcopy(new_crystal.symops)
            self._frac = deepcopy(new_crystal._frac)
            self.__frac_to_cart()
            self._symbols = deepcopy(new_crystal._symbols)

    def create_supercell(
            self,
            indices : tuple = (1, 1, 1),
            asu : bool = False
            ) -> Self:
        """ Create a supercell with the given expansion indices
        
        Parameters
        ----------
        indices : tuple
            The expansion indices to create the supercell
        asu : bool (optional)
            Reduce the cell to its asymmetric unit and
            don't keep all the atoms. Defaults to False
        
        Returns
        -------
        supercell : Crystal
            The supercell as another Crystal object
        
        Raises
        ------
        TypeError
            If the provided indices are not a tuple
        ValueError
            If the provided indices are not a 3D tuple
        """
        if not isinstance(indices, tuple):
            raise TypeError(
                    "Crystal.create_supercell() The provided indices is "
                    "not a tuple."
            )
        if len(indices) != 3:
            raise ValueError(
                    "Crystal.create_supercell() The provided indices is "
                    f"has {len(indices)} dimensions, instead of 3."
            )

        all_atoms = self.apply_symops(wrap_in_cell=True)
        cell_params = self.cell.params

        only_labels = [a[0] for a in all_atoms]
        assert len(only_labels) == len(set(only_labels))

        new_atoms = []
        positions_only = []
        symbols_only = []

        # Loop over all 3 cell lengths
        for i in range(indices[0]):
            for j in range(indices[1]):
                for k in range(indices[2]):

                    # Loop over all atoms
                    for at in all_atoms:

                        # Create a new label
                        pre, pos = at[0].split("_")
                        new_label = f"{pre}_{i}.{j}.{k}"
                        new_label += f"_{pos}"

                        # Create the displacement vector
                        d_vect = np.array([i, j, k])

                        # Calculate the new position
                        new_pos = at[2] + d_vect

                        # Scale the relative positions
                        for e, idx in enumerate(indices):
                            new_pos[e] /= idx

                        # Add the symbol and new position to their lists
                        positions_only.append(new_pos)
                        symbols_only.append(at[1])

                        # Add the newly translated atom
                        new_atoms.append([
                            new_label,
                            at[1],
                            new_pos
                        ])

        # Set the updated cell parameters
        new_cell = Cell(
            cell_params["a"] * indices[0],
            cell_params["b"] * indices[1],
            cell_params["c"] * indices[2],
            cell_params["alpha"],
            cell_params["beta"],
            cell_params["gamma"]
        )

        # If the space group and symmetry could be found, re-order the atoms
        # and add the symmetry operations
        if asu:
            return Crystal.__find_asu(new_cell, new_atoms)
        else:
            # Create the new crystal supercell
            supercell = Crystal('supercell')
            # Establish the cell parameters
            supercell.cell = deepcopy(new_cell)
            # Add the atoms to the cell
            supercell.fill_atoms(new_atoms, relative=True)     

            return supercell
    
    def as_molecular_crystal(self, padding : float = 0) -> MolecularCrystal:
        """ Convert the crystal into a molecular crystal

        Parameters
        ----------
        padding : float (optional)
            The padding to apply to the unit cell. Defaults to 0

        Returns
        -------
        mol_crys : MolecularCrystal
            The molecular crystal as a new MolecularCrystal object
        """
        # Crate the supercell
        scell = self.create_supercell(indices=(3,3,3), asu=False)
        scell.map_to_unit_cell(wrap_in_cell=True)

        # Collect all atom objects
        atoms = []
        
        for atom_idx in range(len(scell)):

            # Get the atom's position
            location = scell.cart_coords[atom_idx]

            # Add the atom
            atoms.append(Atom(
                scell.symbols[atom_idx],
                location[0],
                location[1],
                location[2]
            ))

        # Create a collection object and add the atoms
        stu = Structure('mol_crys')
        stu.add_atoms(*atoms)

        # Find the connectivity and resolve the molecules
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            coll = stu.get_sub_structure(force=True)

        if isinstance(coll, Molecule):
            mol_dict = {"mol_0" : deepcopy(coll)}
        elif isinstance(coll, Collection):
            mol_dict = {}
            for mk, mv in coll.molecules.items():
                mol_dict[mk] = deepcopy(mv)
        else:
            raise RuntimeError(
                    "Crystal.as_molecular_crystal() Could not resolve "
                    "the molecules in the crystal."
            )
        
        # Relative coordinates cell walls and limits
        lower_limit = np.ones(3, dtype=float) * (1/3 - padding)
        upper_limit = np.ones(3, dtype=float) * (2/3 + padding)

        molecules_in_cell = []
        molecule_relative_centers = []
        ivec = np.linalg.inv(scell.cell.vector)

        shift = np.ones(3, dtype=float) * (1/3 + padding)
        m_shift = scell.cell.vector @ shift

        # Loop over all found "molecules"
        for mol_k, mol_v in mol_dict.items():

            # Compute the geometric center of the molecule
            mol_center = mol_v.get_center()

            # Locate the center in relative coordinates
            rel_center = np.round(ivec @ mol_center, 12)

            # If the center is within the cell, keep the molecule
            if np.all(rel_center >= lower_limit) and \
                np.all(rel_center < upper_limit):

                # Shift the molecules and their centers to a single unit cell
                n_mol = deepcopy(mol_v)
                n_mol.move_molecule(-1 * m_shift)
                molecules_in_cell.append(n_mol)
                molecule_relative_centers.append(rel_center - shift)
        
        # Removing duplicates (molecules in special positions) based on
        # the relative centers
        unique_centers = []
        unique_molecules = []
        for i, c in enumerate(molecule_relative_centers):
            for uc in unique_centers:

                # Check if the centers are the same
                replica = False
                for mic in product([0, 1], repeat=3):

                    # Check this in cartesian space
                    uc_cart = scell.cell.vector @ (uc + np.array(mic))
                    c_cart = scell.cell.vector @ c
                    if np.abs(uc_cart - c_cart) < 1.0:
                        replica = True
                        break
                
                # If the center is not a replica, add it to the unique lists
                if not replica:
                    unique_centers.append(c)
                    unique_molecules.append(molecules_in_cell[i])
        
        # Get the original cell with applied symmetry operations
        all_atoms = self.apply_symops(wrap_in_cell=True)

        # Creating a molecular crystal
        mc = MolecularCrystal()
        mc.cell = deepcopy(self.cell)

        # Prepare the inverse of the transformation matrix
        ivec = np.linalg.inv(mc.cell.vector)

        mol_atoms = []
        for i, m in enumerate(unique_molecules):
            for a in m.atoms:
                c_rel = ivec @ a.coordinates
                lab = ""
                for ac in all_atoms:
                    if np.linalg.norm((c_rel % 1) - (ac[2] % 1)) < 1E-9:
                        lab = ac[0]
                        break

                if len(lab) == 0:
                    raise RuntimeError(
                            "Crystal.as_molecular_crystal() The label for "
                            "the following atom could not be found:\n"
                            f"{a.element}\t::\t{a.coordinates}\t::\t{c_rel}"
                    )

                mol_atoms.append([
                    lab,
                    a.element,
                    a.coordinates,
                    i
                ])
        
        all_labels = [ma[0] for ma in mol_atoms]
        unq_labels = set(all_labels)
        if len(all_labels) != len(unq_labels):
            for i, e in enumerate(mol_atoms):
                mol_atoms[i][0] += f"~{e[3]}"

        mc.fill_atoms(mol_atoms)

        return mc

# ------------------------------------------------------- #
#               The Molecular Crystal Class               #
# ------------------------------------------------------- #

class MolecularCrystal(Crystal):
    """ Class representing a molecular crystal structure.
    
    This class allows the definition and manipulation of crystal structures,
    including setting the unit cell, adding atoms, and converting between
    fractional and Cartesian coordinates.
    
    Attributes
    ----------
    name : str
        The name of the crystal structure.
    cell : Cell
        The unit cell of the crystal structure.
    _labels : list
        List of atom labels in the crystal.
    _symbols : list
        List of atom symbols in the crystal.
    _frac : list
        List of fractional coordinates of atoms in the crystal.
    _cartesian : list
        List of Cartesian coordinates of atoms in the crystal.
    _mol_idx : list
        List of molecule indices for each atom in the crystal.
    symops : SymmetryOperations
        Symmetry operations associated with the crystal structure.
    
    Methods
    -------
    __init__(nam='crystal') : None
        Initializes the Crystal object with a name and default unit cell.
    __len__() : int
        Returns the number of atoms in the crystal structure.
    __str__() : str
        Returns a string representation of the crystal structure.
    __frac_to_cart() : None
        Converts fractional coordinates to Cartesian coordinates.
    __cart_to_frac() : None
        Converts Cartesian coordinates to fractional coordinates.
    __unique_int_mols() : None
        Generates unique IDs for molecules in the Molecular Crystal.
    __simple_labels() : list
        Generates simplified labels for atoms by removing molecule indices.
    labels : list
        Property to get the list of atom labels.
    __remove_duplicate_atoms() : None
        Removes duplicate atoms based on labels and fractional coordinates.
    fill_atoms(atoms, relative=False) : None
        Adds atoms to the crystal structure.
    read_cif(file_path) : None
        Reads a CIF file and populates the crystal structure.
    as_cif_coords() : str
        Return the atom information in CIF format
    apply_symops(symops, wrap_in_cell) : None
        Applies symmetry operations to generate the full crystal structure.
    molecules : dict
        Property to get a dictionary of molecules in the crystal structure.
    cell : Cell
        Property to get/set the unit cell of the crystal structure.
    __find_asu(cell, atoms) : MolecularCrystal
        Finds the asymmetric unit of a given cell and atom list.
    reduce_to_asu() : None
        Reduces the crystal structure to its asymmetric unit.
    map_to_unit_cell() : None
        Maps all atoms to the unit cell by applying symmetry operations.
    create_supercell(indices) : Crystal
        Applies the crystal's symmetry operations, creates a supercell
        based on the provided indices, and attempts to find the space group and
        symmetry operations of the new crystal.
    """

    def __init__(self, nam : str = 'molcrys'):
        """ Initialize the MolecularCrystal object.

        Parameters
        ----------
        nam : str, optional
            The name of the crystal structure. Default is 'crystal'.
        """
        self.name = nam
        self._cell = Cell(1, 1, 1, 90, 90, 90)
        self._labels = []
        self._symbols = []
        self._frac = []
        self._cartesian = []
        self._mol_idx = []
        self.symops = SymmetryOperations()
        self.symops.add("1 x,y,z")
    
    def __len__(self):
        """ Return the number of atoms in the molecular crystal

        Returns
        -------
        int
            The number of atoms currently stored in the molecular crystal
        """
        return len(self._cartesian)
    
    def __str__(self):
        """ Return a string representation of the molecular crystal

        Returns
        -------
        str
            A string representation of the molecular crystal, including
            cell parameters, symmetry operations, and atom information.
        """
        max_lbl = max([len(lbl) for lbl in self._labels])
        max_mi = max([len(str(i)) for i in self._mol_idx])

        div = "-" * 125
        output = div + "\n"
        for k, v in self.cell.params.items():
            output += f"{k:<8}:{v:>14.6f}\n"
        output += div + "\n"
        output += str(self.cell.vector) + "\n"
        output += div + "\n"
        output += str(self.symops)
        output += div + "\n"
        for i in range(len(self._cartesian)):
            output += f"{i})\t"
            output += f"{self._labels[i]:>{max_lbl}}\t"
            output += f"{self._symbols[i]}\t:: "
            output += f"{self._cartesian[i][0]:14.6f} "
            output += f"{self._cartesian[i][1]:14.6f} "
            output += f"{self._cartesian[i][2]:14.6f}\t:: "
            output += f"{self._frac[i][0]:14.6f} "
            output += f"{self._frac[i][1]:14.6f} "
            output += f"{self._frac[i][2]:14.6f}\t:: "
            output += f"{self._mol_idx[i]:>{max_mi}}\n"
        output += div
        return output
    
    def __frac_to_cart(self):
        """ Convert fractional coordinates to Cartesian coordinates.
        
        Raises
        ------
        ValueError
            If the unit cell has not been defined.
        """
        if self.cell == None:
            raise ValueError(
                    "MolecularCrystal.__frac_to_cart() The unit cell has "
                    "not been defined."
            )
        vec = self.cell.vector

        self._cartesian = []
        for a in self._frac:
            self._cartesian.append(vec @ a)
    
    def __cart_to_frac(self):
        """ Convert Cartesian coordinates to fractional coordinates.
        
        Raises
        ------
        ValueError
            If the unit cell has not been defined.
        """
        if self.cell == None:
            raise ValueError(
                    "MolecularCrystal.__cart_to_frac() The unit cell has "
                    "not been defined."
            )
        ivec = np.linalg.inv(self.cell.vector)

        self._frac = []
        for a in self._cartesian:
            self._frac.append(ivec @ a)
    
    def __unique_int_mols(self):
        """
        Generate unique IDs for molecules in the Molecular Crystal
        """

        temp_idx = deepcopy(self._mol_idx)
        temp_set = list(set(temp_idx))
        new_idx = []

        for idx in temp_idx:
            new_idx.append(temp_set.index(idx))
        
        assert len(self._mol_idx) == len(new_idx)

        self._mol_idx = deepcopy(new_idx)
    
    def __simple_labels(self):
        """
        Remove any unnecessary characters from the labels of the atoms.
        """

        new_labels = []
        for l in self.labels:
            new_labels.append(
                l.split("~")[0]
            )
        
        if len(set(new_labels)) == len(self.labels):
            self.labels = deepcopy(new_labels)

        new_labels = []
        for l in self.labels:
            new_labels.append(
                l.split("_")[0]
            )
        
        if len(set(new_labels)) == len(self.labels):
            self.labels = deepcopy(new_labels)
    
    def __remove_duplicate_atoms(self, threshold : float = 1E-10):
        """
        Remove duplicate atoms from the Molecular Crystal.

        Parameters
        ----------
        threshold : float, optional
            The threshold distance between atoms to be considered
            duplicates. Default is 1E-10.
        """

        unique_atoms = []
        for i in range(len(self)):
            ctrl = False
            for j in unique_atoms:
                dist = np.linalg.norm(self._cartesian[i] - self._cartesian[j])
                if dist < threshold:
                    ctrl = True
                    break
            if not ctrl:
                unique_atoms.append(i)
        
        self._labels = [self._labels[a] for a in unique_atoms]
        self._symbols = [self._symbols[a] for a in unique_atoms]
        self._cartesian = [self._cartesian[a] for a in unique_atoms]
        self._frac = [self._frac[a] for a in unique_atoms]
        self._mol_idx = [self._mol_idx[a] for a in unique_atoms]

    
    def fill_atoms(self, atoms : list, relative : bool = False) -> None:
        """ Add atoms to the crystal structure.

        Parameters
        ----------
        atoms : list
            A list of atoms to be added to the crystal structure. Each atom
            should be represented as a list or tuple containing:
            - label (str): A unique identifier for the atom (e.g., "C1").
            - symbol (str): The chemical symbol of the atom (e.g., "C").
            - coordinates (numpy.ndarray): A 3D numpy array representing the
              atom's coordinates. If `relative` is True, these should be
              fractional coordinates; otherwise, they should be Cartesian
              coordinates.
            - molecule (int or str): An identifier to tell to what molecule
              are the atoms associated.
        relative : bool, optional
            A boolean indicating whether the provided coordinates are
            fractional (True) or Cartesian (False). Default is False.
        
        Raises
        ------
        TypeError
            If the provided atoms are not in the expected format.
        ValueError
            If the provided atoms do not meet the required conditions.
        """
        if not isinstance(atoms, list):
            raise TypeError(
                    "MolecularCrystal.fill_atoms() atoms should be a list."
            )
        if len(atoms) == 0:
            raise ValueError(
                    "MolecularCrystal.fill_atoms() The provided list of "
                    "atoms is empty."
            )
        pre_labels = []
        for i, at in enumerate(atoms):
            if len(at) != 4:
                raise ValueError(
                        "Crystal.fill_atoms() Each entry in atoms should "
                        "contain 4 items"
                )
            if not isinstance(at[0], str) or \
                not isinstance(at[1], str) or \
                not isinstance(at[2], np.ndarray) or \
                not isinstance(at[3], (int, str)):
                raise ValueError(
                        "MolecularCrystal.fill_atoms() Each entry in atoms "
                        "should contain 4 items: a label, a symbol, the atom's "
                        "relative coordinates and the corresponding molecule. "
                        f"Entry {i+1} is {at}"
                )
            if at[1] not in at[0]:
                raise ValueError(
                        "MolecularCrystal.fill_atoms() Inconsistent label "
                        f"found for entry {i+1}. Label '{at[0]}' does not "
                        f"contain symbol {at[1]}"
                )
            pre_labels.append(at[0])
        label_diff = len(atoms) - len(set(pre_labels))
        if label_diff != 0:
            raise ValueError(
                    "MolecularCrystal.fill_atoms() Each atom requires a "
                    f"unique label. There are {label_diff} entries with "
                    "equal labels."
            )
        for at in atoms:
            self._labels.append(at[0])
            self._symbols.append(at[1])
            self._mol_idx.append(at[3])
            if relative:
                self._frac.append(at[2])
            else:
                self._cartesian.append(at[2])
            
        if relative:
            self.__frac_to_cart()
        else:
            self.__cart_to_frac()
    
    def read_cif(self,
                 file_name : str,
                 tolerance : float = 0):
        """ Read a CIF file and populate the molecular crystal.

        Parameters
        ----------
        file_name : str
            The path to the CIF file to be read.
        tolerance : float
            How far away should the algorithm consider to find
            atoms of individual molecules.
        
        Raises
        ------
        FileNotFoundError
            If the specified CIF file does not exist.
        ValueError
            If the CIF file is malformed or missing required information.
        """
        temp_cry = Crystal('temporary')
        temp_cry.read_cif(file_name)

        # Transform
        temp_molcry = temp_cry.as_molecular_crystal(tolerance)

        # Transfer
        self._cell = deepcopy(temp_molcry._cell)
        self._labels = deepcopy(temp_molcry._labels)
        self._symbols = deepcopy(temp_molcry._symbols)
        self._frac = deepcopy(temp_molcry._frac)
        self._cartesian = deepcopy(temp_molcry._cartesian)
        self._mol_idx = deepcopy(temp_molcry._mol_idx)
        self.symops = deepcopy(temp_molcry.symops)

    def apply_symops(
            self,
            symops : list = [],
            wrap_in_cell : bool = False
        ) -> list:
        """ Apply symmetry operations to generate the full crystal structure.

        Parameters
        ----------
        symops : list, optional
            A list of symmetry operation indices to be applied. If empty,
            all available symmetry operations will be used. Default is an
            empty list.
        wrap_in_cell : bool, optional
            A boolean indicating whether to wrap the resulting fractional
            coordinates within the unit cell (i.e., between 0 and 1). Default
            is False.
        
        Returns
        -------
        list
            A list of lists, each containing the label, symbol, fractional
            coordinates and molecular id of the atoms after applying the
            symmetry operations.
        
        Raises
        ------
        TypeError
            If the provided symops argument is not a list.
        """
        if not isinstance(symops, list):
            raise TypeError(
                    "MolecularCrystal.apply_symops() Requires a list "
                    "of operations"
            )

        if len(symops) == 0:
            symops = list(range(len(self.symops)))
        
        resolved_atoms = []
        for so in symops:
            op = self.symops[so]
            for a in range(len(self)):
                # if "_" in self._labels[a]:
                #     temp_label = self._labels[a].split("_")[0]
                # else:
                temp_label = self._labels[a]

                if "_" in str(self._mol_idx[a]):
                    temp_mol_idx = self._mol_idx[a].split("_")[0]
                else:
                    temp_mol_idx = self._mol_idx[a]
                resolved_atoms.append([
                        f"{temp_label}_{so}",
                        self._symbols[a],
                        (op['matrix'] @ self.rel_coords[a]) + op['vector'],
                        f"{temp_mol_idx}_{so}"
                    ])
                if wrap_in_cell:
                    resolved_atoms[-1][2] %= 1
                resolved_atoms[-1][2] = np.round(resolved_atoms[-1][2], 12)
        
        # Remove overlapping atoms if wrapping was used
        if wrap_in_cell:
            allowed_indices = []
            for i, a in enumerate(resolved_atoms):
                ctrl = False
                for j in allowed_indices:
                    if i != j:
                        if np.linalg.norm(a[2] - resolved_atoms[j][2]) < 1E-5:
                            ctrl = True
                            break
                if not ctrl:
                    allowed_indices.append(i)
            
            resolved_atoms = [resolved_atoms[i] for i in allowed_indices]

        return resolved_atoms
    
    @property
    @lru_cache(maxsize=1)
    def molecules(self) -> dict:
        """ Dynamically creates the molecules in the cell
        
        This method is similar to MolecularCrystal.apply_symops()
        
        Returns
        -------
        mols : dict
            A dynamically-created dictionary of all the molecules
            in a resolved crystal.
        """
        mols = {}

        if len(self.symops) != 1:
            res_structures = self.apply_symops()
        
            for rs in res_structures:
                mol_id = f"mol_{rs[3].split('_')[-1]}"
                if mol_id not in mols.keys():
                    mols[mol_id] = Molecule(mol_id)
                mols[mol_id].add_atoms(Atom(
                    rs[1],
                    rs[2][0],
                    rs[2][1],
                    rs[2][2]
                ))
        else:

            assert len(self.symbols) == len(self.cart_coords)
            assert len(self.symbols) == len(self._mol_idx)

            for i in range(len(self)):
                mol_id = str(self._mol_idx[i])
                if mol_id not in mols.keys():
                    mols[mol_id] = Molecule(mol_id)
                mols[mol_id].add_atoms(Atom(
                    self.symbols[i],
                    self._cartesian[i][0],
                    self._cartesian[i][1],
                    self._cartesian[i][2]
                ))
        
        return mols

    @property
    def cell(self) -> Cell:
        """ Retrieves the cell object with its parameters
        
        Returns
        -------
        Cell
            A Cell object representing the unit cell of the crystal structure.
        """
        return self._cell

    @cell.setter
    def cell(self, cell : Cell) -> None:
        """ Set the unit cell of the crystal structure.

        Parameters
        ----------
        cell : Cell
            A Cell object representing the unit cell of the crystal structure.
        
        Raises
        ------
        TypeError
            If the provided object is not a Cell instance.
        """
        if not isinstance(cell, Cell):
            raise TypeError(
                    "MolecularCrystal.cell() The provided object "
                    "is not a cell.")
        
        # If the molecules haven't been identified, ...
        if len(self._mol_idx) == 0:
            # ... and nothing has been defined, just assign the new cell
            if len(self._frac) == 0 and len(self._cartesian) == 0:
                self._cell = cell
            # ... and there ARE atoms defined, this makes no sense!
            else:
                raise RuntimeError(
                    "MolecularCrystal.cell() Could not find any molecular "
                    "indices in this object."
                )
        
        # If there are molecules, but not atoms, just assign the cell
        if len(self._frac) == 0 and len(self._cartesian) == 0:
            self._cell = cell
        # If there are molecules, and some atoms have been defined, ...
        else:
            if len(self._frac) == 0:
                self.__cart_to_frac()
            if len(self._cartesian) == 0:
                self.__frac_to_cart()

            # In principle, all should have the same number of entries
            assert len(self._frac) == len(self._cartesian) and \
                    len(self._frac) == len(self._mol_idx) and \
                    len(self._frac) == len(self._labels)
            
            self.__simple_labels()
            self.map_to_unit_cell()
            
            # Get the Molecular Crystal's molecules
            mols = self.molecules

            # Get their center of mass
            coms_cart = [m.get_center_of_mass() for m in mols.values()]

            # Get the inverse transform (to map the COMs to relative space)
            ivec = np.linalg.inv(self.cell.vector)
            coms_frac = [ivec @ cc for cc in coms_cart]

            # Map the relative COMs to the new cell's space
            coms_reshaped = [cell.vector @ cf for cf in coms_frac]

            # Find how much should the molecules be moved
            diffs = [rs - c for c, rs in zip(coms_cart, coms_reshaped)]

            # Temporary lists
            temp_symbols = []
            temp_cartesian = []

            for i, (k, v) in enumerate(mols.items()):
                # Move the molecules
                mols[k].move_molecule(diffs[i])
                # Get the new cartesian coordinates
                for atom in v.atoms:
                    temp_symbols.append(atom.element)
                    temp_cartesian.append(atom.coordinates)

            # Sanity check and update the cartesian coordinates
            if np.all(temp_symbols == self._symbols):
                self._cartesian = deepcopy(temp_cartesian)
                self._cell = cell
                self.__cart_to_frac()
            else:
                raise RuntimeError(
                    "MolecularCrystal.cell() The symbols of the atoms do "
                    "not match before and after the change in cell parameters."
                )

            self.reduce_to_asu()
    
    @classmethod
    def __find_asu(
            cls,
            cell : Cell,
            atoms : list
        ) -> Self:

        # Create the new crystal
        new_crystal = MolecularCrystal('molcrys_asu')

        # Set the updated cell parameters
        new_crystal.cell = Cell(cell.params)

        lbls_only = []
        sbls_only = []
        psts_only = []
        mols_only = []
        for a in atoms:
            lbls_only.append(a[0].split("_")[0])
            sbls_only.append(a[1])
            psts_only.append(a[2])
            mols_only.append(a[3])

        # Try to find the space group and symmetry operations of
        # the new cell
        new_sym = {}
        try:
            new_sym = SymmetryOperations.get_sg_from_cell(
                new_crystal.cell,
                psts_only,
                sbls_only,
                full_output = True
            )
        except ValueError as e:
            warn("MolecularCrystal.__find_asu() WARNING! No space group "
                 "could be found for the supercell. Continuing with P1.")

            # Add the atoms to the cell
            new_crystal.fill_atoms(deepcopy(atoms), relative=True)

            return new_crystal

        # If the space group and symmetry could be found, re-order the atoms
        # and add the symmetry operations
        if "unique_atoms" in new_sym:
            new_atoms = []
            for a in new_sym["unique_atoms"]:
                new_atoms.append([
                    lbls_only[a],
                    sbls_only[a],
                    psts_only[a],
                    mols_only[a]
                ])
            new_symops = SymmetryOperations()
            for i, so in enumerate(new_sym["text"]):
                new_symops.add(f"{i + 1} {so}")
            new_crystal.symops = new_symops

        else:
            warn("MolecularCrystal.__find_asu() WARNING! No asymmetric "
                 "unit could be found for the supercell. Continuing with P1.")

            new_atoms = deepcopy(atoms)
        
        # Add the atoms to the cell
        new_crystal.fill_atoms(new_atoms, relative=True)

        return new_crystal
    
    def reduce_to_asu(self) -> None:
        """
        Reduce the molecular crystal to the asymmetric unit (ASU).

        This function applies all symmetry operations to the molecular crystal,
        then finds the asymmetric unit (ASU) of the resulting structure.
        """
        cut_atoms = self.apply_symops(wrap_in_cell=True)
        
        new_crystal = MolecularCrystal.__find_asu(self.cell, cut_atoms)

        if len(new_crystal) != len(self):
            self._labels = []
            self._cartesian = []
            self._frac = []
            self.symops = deepcopy(new_crystal.symops)
            self._frac = deepcopy(new_crystal._frac)
            self.__frac_to_cart()
            self._labels = deepcopy(new_crystal._labels)
            self._symbols = deepcopy(new_crystal._symbols)
            self._mol_idx = deepcopy(new_crystal._mol_idx)
        
        # Clean the molecular indices
        self.__unique_int_mols()
        self.__simple_labels()
    
    def map_to_unit_cell(self) -> None:
        """
        Map all atoms in the molecular crystal to the unit cell.

        This function applies all symmetry operations to the molecular crystal,
        then maps all atoms to the unit cell. It also resets the symmetry
        operations to the identity operation and updates the molecular
        indices.
        """
        # Get the atomic positions after applying all symmetry ops
        new_atoms = self.apply_symops()

        # Reset symmetry operations
        new_symops = SymmetryOperations()
        new_symops.add("1 x,y,z")
        
        # Fill each list individually
        new_frac = []
        new_symbols = []
        new_labels = []
        new_mols = []
        for atom in new_atoms:
            new_labels.append(atom[0])
            new_symbols.append(atom[1])
            new_frac.append(atom[2])
            new_mols.append(atom[3])
        
        # Update the lists in the object
        self._labels = deepcopy(new_labels)
        self._symbols = deepcopy(new_symbols)
        self._frac = deepcopy(new_frac)
        self._mol_idx = deepcopy(new_mols)
        self.symops = new_symops

        # Update the cartesian coordinates
        self.__frac_to_cart()

        # Clean the molecular indices
        self.__unique_int_mols()
    
    def create_supercell(
            self,
            indices : tuple = (1, 1, 1),
            asu : bool = False
            ) -> Self:
        """ Create a supercell with the given expansion indices
        
        Parameters
        ----------
        indices : tuple
            The expansion indices to create the supercell
        asu : bool (optional)
            Reduce the cell to its asymmetric unit and
            don't keep all the atoms. Defaults to False
        
        Returns
        -------
        supercell : MolecularCrystal
            The supercell as another MolecularCrystal object
        
        Raises
        ------
        TypeError
            If the provided indices are not a tuple
        ValueError
            If the provided indices are not a 3D tuple
        """

        if not isinstance(indices, tuple):
            raise TypeError(
                    "MolecularCrystal.create_supercell() The provided "
                    "indices is not a tuple."
            )
        if len(indices) != 3:
            raise ValueError(
                    "MolecularCrystal.create_supercell() The provided "
                    f"indices is has {len(indices)} dimensions, instead of 3."
            )

        all_atoms = self.apply_symops()
        cell_params = self.cell.params

        only_labels = [a[0] for a in all_atoms]
        assert len(only_labels) == len(set(only_labels))

        new_atoms = []
        positions_only = []
        symbols_only = []

        # Loop over all 3 cell lengths
        for i in range(indices[0]):
            for j in range(indices[1]):
                for k in range(indices[2]):

                    # Loop over all atoms
                    for at in all_atoms:

                        # Create a new label
                        decom = at[0].split("_")
                        new_label = f"{decom[0]}_{i}.{j}.{k}"
                        new_label += f"_{decom[1]}"

                        # Create the new molecule index
                        new_mol_idx = at[3] + f"_{i}_{j}_{k}"

                        # Create the displacement vector
                        d_vect = np.array([i, j, k])

                        # Calculate the new position
                        new_pos = at[2] + d_vect

                        # Scale the relative positions
                        for e, idx in enumerate(indices):
                            new_pos[e] /= idx

                        # Add the symbol and new position to their lists
                        positions_only.append(new_pos)
                        symbols_only.append(at[1])

                        # Add the newly translated atom
                        new_atoms.append([
                            new_label,
                            at[1],
                            new_pos,
                            new_mol_idx
                        ])

        # Set the updated cell parameters
        new_cell = Cell(
            cell_params["a"] * indices[0],
            cell_params["b"] * indices[1],
            cell_params["c"] * indices[2],
            cell_params["alpha"],
            cell_params["beta"],
            cell_params["gamma"]
        )

        # If the space group and symmetry could be found, re-order the atoms
        # and add the symmetry operations
        if asu:
            return MolecularCrystal.__find_asu(new_cell, new_atoms)
        else:
            # Create the new crystal supercell
            supercell = MolecularCrystal('supercell')
            # Establish the cell parameters
            supercell.cell = deepcopy(new_cell)
            # Add the atoms to the cell
            supercell.fill_atoms(new_atoms, relative=True)
            # Remove duplicates
            supercell.__remove_duplicate_atoms()
            # Ensure a simple naming convention
            supercell.__unique_int_mols()

            return supercell