import ctypes as ct
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))

if os.system("uname -s") == "Windows":
    lib_name = "lib_geometries.dll"
elif os.system("uname -s") == "Darwin":
    lib_name = "lib_geometries.dylib"
elif os.system("uname -s") == "Linux":
    lib_name = "lib_geometries.so"
else:
    raise OSError("Unsupported operating system")

geo_lib = ct.cdll.LoadLibrary(os.path.join(here, lib_name))

f_distance_matrix = geo_lib.distance_matrix
f_distance_matrix.argtypes = [
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.c_int,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_int)]

def distance_matrix(
        coordinates : np.ndarray,
        radii : np.ndarray) -> dict:
    """
    Method to compute the distance matrix and bonds using
    Fortran subroutines

    Parameters
    ----------
    coordinates : ndarray
        A NumPy array with the X, Y and Z coordinates as `float`
    radii : ndarray
        A NumPy array with the VdW radii of each atom as `float`

    Returns
    -------
    dict
        A dictionary with the distance matrix and the bonds
    """
    coords = np.copy(np.asfortranarray(coordinates))
    rads = np.copy(np.asfortranarray(radii))

    v_ptr = coords.ctypes.data_as(ct.POINTER(ct.c_double))
    r_ptr = rads.ctypes.data_as(ct.POINTER(ct.c_double))

    N = len(coordinates)

    distance_matrix = np.zeros(N * N).reshape((N, N), order='F')
    pre_bonds = np.zeros(N * N * 2, dtype=ct.c_int).reshape((N*N, 2), order='F')

    d_ptr = distance_matrix.ctypes.data_as(ct.POINTER(ct.c_double))
    b_ptr = pre_bonds.ctypes.data_as(ct.POINTER(ct.c_int))

    nb_bonds = ct.c_int(0)

    f_distance_matrix(v_ptr, r_ptr, N, d_ptr, nb_bonds, b_ptr)

    bonds = pre_bonds[:nb_bonds.value].reshape((nb_bonds.value, 2))
    bonds =bonds.tolist()

    return {
        "distance_matrix" : distance_matrix,
        "bonds" : bonds
        }

f_find_bonds = geo_lib.find_bonds
f_find_bonds.argtypes = [
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int)]

def get_bonds(
        coordinates : np.ndarray,
        radii : np.ndarray,
        box_size : float = 3.0,
        tolerance : float = 1.15) -> list:
    """
    Method to compute the bonds using Fortran subroutines

    Parameters
    ----------
    coordinates : ndarray
        A NumPy array with the X, Y and Z coordinates as `float`
    radii : ndarray
        A NumPy array with the VdW radii of each atom as `float`

    Returns
    -------
    list
        A list with the bonds
    """
    coords = np.copy(np.asfortranarray(coordinates, dtype=np.float64))
    rads = np.copy(np.asfortranarray(radii, dtype=np.float64))

    c_ptr = coords.ctypes.data_as(ct.POINTER(ct.c_double))
    r_ptr = rads.ctypes.data_as(ct.POINTER(ct.c_double))

    N = len(coordinates)

    pre_bonds = np.asfortranarray(
                    np.zeros(4*N * 2, dtype=np.int32).reshape((4*N, 2))
                )
    b_ptr = pre_bonds.ctypes.data_as(ct.POINTER(ct.c_int))

    nb_bonds = ct.c_int(0)
    box_size = ct.c_double(box_size)
    tolerance = ct.c_double(tolerance)

    f_find_bonds(c_ptr, r_ptr, N, box_size, tolerance, ct.byref(nb_bonds), b_ptr)

    # Remove duplicates
    pre_bonds = pre_bonds[:nb_bonds.value]
    pre_bonds = np.unique(pre_bonds, axis=0)

    bonds = pre_bonds.tolist()

    return bonds