import os                            # To navigate the file system
import shutil                        # To do some operation over the file system
import tempfile                      # To create files in the OS' temp directory
import time                          # A way to keep track of time
import numpy as np                   # To handle numerical calculations
import multiprocessing as mp         # To be able to run things in parallel

from tqdm import tqdm                # To show progress bars
from copy import deepcopy            # To copy complete objects
from subprocess import run           # Method to run external commands
from fractions import Fraction       # To handle tricky decimals
from scipy import constants as cts   # To handle physical constants
from abc import ABC, abstractmethod  # To be able to create several drivers

from .structure import Structure
from .atom import PERIODIC_TABLE, BOHR
from .molecule import Molecule
from .collection import Collection

BOHR2ANG = cts.physical_constants["Bohr radius"][0] * 1e10  # 1 Bohr in Angstrom

class QM_driver(ABC):
    """ Abstract class for all QM drivers
    
    This class is just supposed to work as a base
    class for other drivers. The important methods are
    the constructor and the execute methods.
    """

    def __init__(self,
                 qm_props : dict,
                 sub_structure : Molecule | Collection,
                 calc_name : str = 'QM_calculation',
                 verbose : bool = False):
        """ QM_driver constructor method
        
        Parameters
        ----------
        qm_props : dict
            The calculation's properties (e.g. level of theory, basis set)
        sub_structure : Molecule | Collection
            A Molecule or Collection object to create the inputs for Orca
        calc_name : str
            The name to be given to the calculation directory
        verbose : bool
            If True, print messages
        """
        self.props = qm_props
        self.sub_structure = sub_structure
        self.calc_name = calc_name
        self.verbose = verbose
    
    def __save_frequencies(self,
                         eigenvalues : np.ndarray,
                         path : str,
                         save : bool = True) -> np.ndarray:
        """ Method to save frequencies to a text file
        
        This method will save the frequencies provided
        as argument to a text file located at the path
        provided as argument.
        
        Parameters
        ----------
        eigenvalues : np.ndarray
            A numpy array with the eigenvalues of the Hessian
        path : str
            The path where the frequencies should be saved
        save : bool
            If True, the frequencies will be saved
        
        Returns
        -------
        freq : np.ndarray
            A numpy array with the frequencies

        Notes
        -----

        The eigenvalues are expected to be in Hartree/(Bohr^2 * g/mol) units.

        1 Hartree = 4.3597447222071e-18 J

        1 Bohr = 0.529177210903e-10 m

        The frequencies will be saved in cm^-1 units.
        """
        nb_atoms = self.sub_structure.get_num_atoms()
        zeros = nb_atoms * 3 - len(eigenvalues)

        # Computing the frequencies

        # Square root of the eigenvalues and multiply by 2*pi to get the angular frequencies
        freq = np.sqrt(eigenvalues.astype(complex))
        freq /= (2 * np.pi)

        # Units are now in sqrt(Hartree/(Bohr^2 * g/mol)) = sqrt(Hartree) / (Bohr * sqrt(g/mol))
        # Convert to sqrt(J) / (Bohr * sqrt(g/mol)) = sqrt(kg m^2 / s^2) / (Bohr * sqrt(g/mol))
        freq *= np.sqrt(cts.physical_constants['Hartree energy'][0])

        # Units are now in (sqrt(kg) m / s) / (Bohr * sqrt(g/mol))
        # Convert to (sqrt(kg) m / s) / (m * sqrt(g/mol)) = sqrt(kg) / (s * sqrt(g/mol))
        freq /= cts.physical_constants['Bohr radius'][0]

        # Units are now in sqrt(kg) / (s * sqrt(g/mol)) = sqrt(kg * mol / g) / s = ...
        # ... = sqrt(kg * mol / (kg * 1000)) / s = sqrt(mol / 1000) / s
        # Convert to sqrt(mol / 1000) * sqrt(N_A / mol * 1000) / s = 1 / s
        freq *= np.sqrt(cts.physical_constants["kilogram-atomic mass unit relationship"][0])

        # Units are now in 1/s
        # Convert to cm^-1
        freq /= cts.c
        freq *= 1e-2

        if save:
            # Header
            line_width = 42
            output = ""
            output += "-" * line_width + "\n"
            output += f"{'Frequencies':^{line_width}}\n"
            output += "-" * line_width + "\n"

            # Translations / Rotations
            for j in range(zeros):
                output += f"{j:^6}\t{0:>16.8f} cm^-1\n"
            for i, f in enumerate(freq):
                if f.real == 0 and f.imag != 0:
                    output += f"{i+zeros:^6}\t{-1 * f.imag:>16.8f} cm^-1  ! imag !\n"
                else:
                    output += f"{i+zeros:^6}\t{f.real:>16.8f} cm^-1\n"
            output += "-" * line_width + "\n"
            with open(os.path.join(path, 'frequencies.txt'), 'w') as f:
                f.write(output)
        
        return freq
    
    def __save_normal_modes(
            self,
            modes : np.ndarray,
            path : str
        ) -> None:
        """ Method to save the normal modes to a text file

        This method will save the normal modes provided
        as argument to a text file located at the path
        provided as argument.

        Parameters
        ----------
        modes : np.ndarray
            A numpy array with the normal modes
        path : str
            The path where the normal modes should be saved
        """
        nb_atoms = self.sub_structure.get_num_atoms()
        nb_dims = nb_atoms * 3
        zeros = nb_dims - modes.shape[1]

        full_modes = np.zeros((nb_dims, nb_dims))
        full_modes[:,zeros:] = modes

        nb_blocks = nb_dims // 5
        resid = nb_dims % 5

        if resid != 0:
            nb_blocks += 1

        line_width = 98
        output = ""
        output += "-" * line_width + "\n"
        output += f"{'Normal Modes':^{line_width}}\n"
        output += "-" * line_width + "\n"

        for b in range(nb_blocks):

            ll = b*5
            ul = (b+1)*5 if b+1 != nb_blocks else b*5 + resid

            block_modes = full_modes[:,ll:ul]

            output += "      "
            for k in range(ll, ul):
                output += f"{k:^18}"
            output += "\n"

            for i, r in enumerate(block_modes):
                output += f"{i:^6}"
                for j, c in enumerate(r):
                    output += f"{c:18.10e}"
                output += "\n"
            
            if b != nb_blocks - 1:
                output += "\n"
        output += "-" * line_width

        with open(os.path.join(path, 'normal_modes.txt'), 'w') as f:
            f.write(output)
    
    def __save_hessian(
            self,
            hessian : np.ndarray,
            path : str,
            projected : bool = False,
            mod_name : str = ''
        ) -> None:
        """ Method to save the hessian to a text file
        
        This method will save the hessian provided
        as argument to a text file located at the path
        provided as argument.
        
        Parameters
        ----------
        hessian : np.ndarray
            A numpy array with the hessian matrix
        path : str
            The path where the hessian should be saved
        projected : bool
            Is this the projected hessian?
        mod_name : str
            The modifier name to be added to the file name
        """
        if projected:
            nb_dims = hessian.shape[0]
        else:
            nb_atoms = self.sub_structure.get_num_atoms()
            nb_dims = nb_atoms * 3

        output = f"{nb_dims} {nb_dims}\n"
        for i in range(nb_dims):
            for j in range(nb_dims):
                output += f"{hessian[i][j]:18.10e}"
                if j != nb_dims - 1:
                    output += " "
            output += "\n"

        nm_file = 'projected_hessian' if projected else 'hessian'
        if mod_name != '':
            nm_file += f"_{mod_name}"
        nm_file += '.txt'

        with open(os.path.join(path, nm_file), 'w') as f:
            f.write(output)
        
        np.save(os.path.join(path, nm_file.replace('.txt', '.npy')), hessian)
    
    def __save_gradient(self, gradient : np.ndarray, path : str) -> None:
        """ Method to save the gradient to a text file
        
        This method will save the gradient provided
        as argument to a text file located at the path
        provided as argument.
        
        Parameters
        ----------
        gradient : np.ndarray
            A numpy array with the gradient
        path : str
            The path where the gradient should be saved
        """
        nb_atoms = self.sub_structure.get_num_atoms()

        # Angstrom output
        output_a = f"{nb_atoms} 3\n"
        for i in range(nb_atoms):
            for j in range(3):
                output_a += f"{gradient[i][j]:18.10f}"
                if j != 2:
                    output_a += " "
            output_a += "\n"

        with open(os.path.join(path, 'gradient.txt'), 'w') as f:
            f.write(output_a)
        
        np.save(os.path.join(path, 'gradient.npy'), gradient)

        # Bohr output
        output_b = f"{nb_atoms} 3\n"
        for i in range(nb_atoms):
            for j in range(3):
                output_b += f"{gradient[i][j] / BOHR:18.10f}"
                if j != 2:
                    output_b += " "
            output_b += "\n"

        with open(os.path.join(path, 'gradient_bohr.txt'), 'w') as f:
            f.write(output_b)
        
        np.save(os.path.join(path, 'gradient_bohr.npy'), gradient)

    def _compute_trans_rot(self,
                           masses : np.ndarray,
                           bohr : bool = True) -> list[np.ndarray]:
        """
        Method to compute the translational and rotational
        displacements independently.

        Parameters
        ----------
        masses : np.ndarray
            A numpy array with the masses of the atoms
        bohr : bool
            Should the calculation be done in Bohr, instead of
            Angstrom?
        
        Returns
        -------
        D : list[np.ndarray]
            A list of numpy arrays with the translational and
            rotational displacements
        """

        # Displacements (translations [3] and rotations [3] = 6)
        D = [[] for i in range(6)]

        # The inertia tensor comes with additional information:
        # I      ...   Inertia tensor
        # iv     ...   Eigenvalues of the inertia tensor
        # X      ...   Eigenvectors of the inertia tensor
        # satoms ...   The atoms shifted to the center of mass
        I, iv, X, satoms = self.sub_structure.compute_inertia_tensor(bohr)
        coords = [a[1].tolist() for a in satoms]

        # Getting the shape of the molecule from its moments of inertia
        A, B, C = sorted(iv)
        if self.sub_structure.get_num_atoms() == 1 and \
            np.abs(A) < 1e-7 and np.abs(B) < 1e-7 and np.abs(C) < 1e-7:
            shape = "SINGLE_ATOM"
        elif np.allclose(A, B, atol=1e-7) and np.allclose(B, C, atol=1e-7):
            shape = "SPHERICAL_TOP"
        elif A < 1e-7 and np.allclose(B, C, atol=1e-7):
            shape = "LINEAR"
        elif not np.allclose(A, B, atol=1e-7) and np.allclose(B, C, atol=1e-7) or \
             not np.allclose(B, C, atol=1e-7) and np.allclose(A, B, atol=1e-7):
            shape = "SYMMETRIC_TOP"
        else:
            shape = "ASYMMETRIC_TOP"

        # Reversing the order of the eigenvalues/vectors
        iv = iv[::-1]
        X = X[:,::-1]
        ex, ey, ez = X.T

        # Projecting the coordinates into the principal axes
        P = np.array(coords).dot(X)
        cx, cy, cz = P.T

        masses = np.sqrt(np.array(masses))

        for i, m in enumerate(masses):

            for j in range(3):

                res = j % 3

                # Translations
                D[0].append(m if res == 0 else 0.0)
                D[1].append(m if res == 1 else 0.0)
                D[2].append(m if res == 2 else 0.0)

        # Rotations
        # https://gaussian.com/vib/
        # https://pyscf.org/_modules/pyscf/hessian/thermo.html#dump_normal_mode

        D[3] = (masses[:,None] * (cy[:,None] * ez - cz[:,None] * ey)).ravel()
        D[4] = (masses[:,None] * (cz[:,None] * ex - cx[:,None] * ez)).ravel()
        D[5] = (masses[:,None] * (cx[:,None] * ey - cy[:,None] * ex)).ravel()
        
        # Converting the displacements to numpy arrays
        D = [np.array(d) for d in D]

        D[3] /= BOHR2ANG
        D[4] /= BOHR2ANG
        D[5] /= BOHR2ANG

        if shape == "LINEAR":
            return D[:5], shape
        elif shape == "SINGLE_ATOM":
            return D[:3], shape
        else:
            return D, shape
    
    def __accu_hess(
            self,
            delta : float = 5e-3 * BOHR2ANG,
            n_cores : int = mp.cpu_count()
            ) -> tuple:
        r""" Method to compute the numerical hessian of the
        given structure, using the QM engine.
        
        This method will compute the numerical hessian of
        the current sub_structure using a finite difference
        approximation. It will also compute the points required
        for a Gradient, but the idea is to have enough data
        to fit a polynomial surface and use its derivatives.
        
        Parameters
        ----------
        delta : float
            The displacement to be used for the finite
            difference approach, in Angstrom (1 Bohr = 0.529177 A)
        n_cores : int
            The number of cores to be used for the parallel
            execution of the calculations.
        
        Returns
        -------
        dict
            A dictionary with the Hessian matrix, and
            the path of the working directory.
        
        Notes
        -----
        The fitting of the 2D surface to the 9 points per derivative
        is done with the Least Squares functionality of NumPy.
        Consider the following:

        .. math::
            f\left(q_{1}, q_{2}\right) =
                a \cdot q_{1}^2 *         +
                b \cdot         * q_{2}^2 +
                c \cdot q_{1}   * q_{2}   +
                d \cdot q_{1}             +
                e \cdot        q_{2}      +
                f
        
        This is the polynomial to be fitted. It uses 9 coefficients,
        which means that the system of equations is exact in the worst
        of cases.
        """
        here = os.getcwd()
        tmp_dir = tempfile.gettempdir()

        # Create a temporary working directory
        work = os.path.join(tmp_dir, f'QM_Hessian_AH_{int(time.time())}')
        if os.path.exists(work):
            shutil.rmtree(work)
        os.mkdir(work)
        os.chdir(work)

        # Dictionary to keep track of all paths
        paths = {}

        # First, run the base calculation
        paths[f"_base"] = self.create_input(f"_base")
        self.run_calculation(paths[f"_base"])
        base_results = self.parse_output(paths[f"_base"])
        E_base = base_results['Energy[SCF]']

        # Get the number of atoms, and prepare the displacements
        n_atoms = self.sub_structure.get_num_atoms()

        # Keep a reference to the original structure
        reference = deepcopy(self.sub_structure)

        # Loop over each atom
        for a in range(n_atoms):
            # Loop over all 3 dimensions
            for i, q in enumerate('xyz'):
                # Loop over each atom
                for b in range(a, n_atoms):
                    # Loop over all 3 dimensions
                    for j, p in enumerate('xyz'):

                        # Compute the displacement and create
                        # the inputs for the calculation

                        # Positive-zero
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] += delta
                        paths[f"_{a}p{q}_{b}--"] = self.create_input(
                                                    f"_{a}p{q}_{b}--",
                                                    ref_path=paths[f"_base"]
                                                    )
                        
                        # Zero-positive
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[b][1][j] += delta
                        paths[f"_{a}--_{b}p{p}"] = self.create_input(
                                                    f"_{a}--_{b}p{p}",
                                                    ref_path=paths[f"_base"]
                                                    )

                        # Positive-positive
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] += delta
                        self.sub_structure[b][1][j] += delta
                        paths[f"_{a}p{q}_{b}p{p}"] = self.create_input(
                                                    f"_{a}p{q}_{b}p{p}",
                                                    ref_path=paths[f"_base"]
                                                    )
                        # Positive-negative
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] += delta
                        self.sub_structure[b][1][j] -= delta
                        paths[f"_{a}p{q}_{b}n{p}"] = self.create_input(
                                                    f"_{a}p{q}_{b}n{p}",
                                                    ref_path=paths[f"_base"]
                                                    )
                        # Negative-positive
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] -= delta
                        self.sub_structure[b][1][j] += delta
                        paths[f"_{a}n{q}_{b}p{p}"] = self.create_input(
                                                    f"_{a}n{q}_{b}p{p}",
                                                    ref_path=paths[f"_base"]
                                                    )
                        # Negative-negative
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] -= delta
                        self.sub_structure[b][1][j] -= delta
                        paths[f"_{a}n{q}_{b}n{p}"] = self.create_input(
                                                    f"_{a}n{q}_{b}n{p}",
                                                    ref_path=paths[f"_base"]
                                                    )
                        
                        # Negative-zero
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] -= delta
                        paths[f"_{a}n{q}_{b}--"] = self.create_input(
                                                    f"_{a}n{q}_{b}--",
                                                    ref_path=paths[f"_base"]
                                                    )
                        
                        # Zero-negative
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[b][1][j] -= delta
                        paths[f"_{a}--_{b}n{p}"] = self.create_input(
                                                    f"_{a}--_{b}n{p}",
                                                    ref_path=paths[f"_base"]
                                                    )
        
        # Restore the original structure
        self.sub_structure = deepcopy(reference)

        if self.verbose:
            print(("\nComputing the displacements for the Hessian"
                  f"using {n_cores} cores"), end=" ... ", flush=True)
        if n_cores == 1:
            results = []
            for k, v in tqdm(paths.items()):
                self.run_calculation(v)
                res = self.parse_output(v)
                results.append((k, res['Energy[SCF]']))
        else:

            # Define the worker function for parallel execution
            global worker
            def worker(kvp : tuple) -> tuple:
                self.run_calculation(kvp[1])
                results = self.parse_output(kvp[1])
                return (kvp[0], results['Energy[SCF]'])
            
            # Run the calculations in parallel
            pool = mp.Pool(n_cores)

            results = list(
                        tqdm(
                            pool.imap_unordered(
                                worker,
                                [(k, v) for k, v in paths.items()]
                            ),
                            total = len(paths),
                            disable = not self.verbose
                        )
                    )
            pool.close()
            pool.join()
            
            # with mp.Pool(n_cores) as pool:
            #     results = pool.map(worker, [(k, v) for k, v in paths.items()])

        if self.verbose:
            print("done.\n", flush=True)

        # Switch back to the base directory
        os.chdir(here)

        # Create an empty Hessian matrix
        hess = np.zeros((n_atoms * 3, n_atoms * 3))

        # Delta in bohr
        delta_bohr = delta * BOHR

        energies = {k: v for k, v in results}
        for a in range(n_atoms):
            for i, q in enumerate('xyz'):
                for b in range(a, n_atoms):
                    for j, p in enumerate('xyz'):

                        # Hessian indices
                        xx = a*3 + i
                        yy = b*3 + j

                        # Get the energies for this particular atom and direction
                        E_p0 = energies[f"_{a}p{q}_{b}--"]
                        E_0p = energies[f"_{a}--_{b}p{p}"]
                        E_pp = energies[f"_{a}p{q}_{b}p{p}"]
                        E_pn = energies[f"_{a}p{q}_{b}n{p}"]
                        E_np = energies[f"_{a}n{q}_{b}p{p}"]
                        E_nn = energies[f"_{a}n{q}_{b}n{p}"]
                        E_n0 = energies[f"_{a}n{q}_{b}--"]
                        E_0n = energies[f"_{a}--_{b}n{p}"]

                        # Fitting the polynomial surface
                        px = np.linspace(-1,1,3) * delta_bohr
                        py = np.linspace(-1,1,3) * delta_bohr
                        pX, pY = np.meshgrid(px, py, copy=False)
                        pX = pX.flatten()
                        pY = pY.flatten()
                        pZ = np.array([
                            E_nn, E_0n, E_pn,
                            E_n0, E_base, E_p0,
                            E_np, E_0p, E_pp
                        ])
                        A = np.array([
                            pX**2,
                            pY**2,
                            pX * pY,
                            pX,
                            pY,
                            np.ones(len(pZ))
                        ]).T
                        B = pZ.flatten()
                        coeff, r, rank, s = np.linalg.lstsq(A,B)

                        # Fill the Hessian matrix
                        if a == b:
                            if i == j:
                                hess[xx,xx] = coeff[0] * 2
                            else:
                                hess[xx,yy] = hess[yy,xx] = coeff[2] * 2
                        else:
                            hess[xx,yy] = hess[yy,xx] = coeff[2]

        return {
            'hessian' : hess,
            'work_dir' : work
        }
    
    def __full_hess(
            self,
            delta : float = 5e-3 * BOHR2ANG,
            n_cores : int = mp.cpu_count()
            ) -> tuple:
        """ Method to compute the numerical hessian of the
        given structure, using the QM engine.
        
        This method will compute the numerical hessian of
        the current sub_structure using a finite difference
        approach. The calculations will be run in parallel,
        each in its own folder.
        
        Parameters
        ----------
        delta : float
            The displacement to be used for the finite
            difference approach, in Angstrom (1 Bohr = 0.529177 A)
        n_cores : int
            The number of cores to be used for the parallel
            execution of the calculations.
        
        Returns
        -------
        dict
            A dictionary with the Hessian matrix, and
            the path of the working directory.
        """
        here = os.getcwd()
        tmp_dir = tempfile.gettempdir()

        # Create a temporary working directory
        work = os.path.join(tmp_dir, f'QM_Hessian_FH_{int(time.time())}')
        if os.path.exists(work):
            shutil.rmtree(work)
        os.mkdir(work)
        os.chdir(work)

        # Dictionary to keep track of all paths
        paths = {}

        # First, run the base calculation
        paths[f"_base"] = self.create_input(f"_base")
        self.run_calculation(paths[f"_base"])
        base_results = self.parse_output(paths[f"_base"])
        base_energy = base_results['Energy[SCF]']

        # Get the number of atoms, and prepare the displacements
        n_atoms = self.sub_structure.get_num_atoms()

        # Keep a reference to the original structure
        reference = deepcopy(self.sub_structure)

        # Loop over each atom
        for a in range(n_atoms):
            # Loop over all 3 dimensions
            for i, q in enumerate('xyz'):
                # Loop over each atom
                for b in range(a, n_atoms):
                    # Loop over all 3 dimensions
                    for j, p in enumerate('xyz'):

                        # Compute the displacement and create
                        # the inputs for the calculation

                        # Positive-positive
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] += delta
                        self.sub_structure[b][1][j] += delta
                        paths[f"_{a}p{q}_{b}p{p}"] = self.create_input(
                                                    f"_{a}p{q}_{b}p{p}",
                                                    ref_path=paths[f"_base"]
                                                    )
                        # Positive-negative
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] += delta
                        self.sub_structure[b][1][j] -= delta
                        paths[f"_{a}p{q}_{b}n{p}"] = self.create_input(
                                                    f"_{a}p{q}_{b}n{p}",
                                                    ref_path=paths[f"_base"]
                                                    )
                        # Negative-positive
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] -= delta
                        self.sub_structure[b][1][j] += delta
                        paths[f"_{a}n{q}_{b}p{p}"] = self.create_input(
                                                    f"_{a}n{q}_{b}p{p}",
                                                    ref_path=paths[f"_base"]
                                                    )
                        # Negative-negative
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] -= delta
                        self.sub_structure[b][1][j] -= delta
                        paths[f"_{a}n{q}_{b}n{p}"] = self.create_input(
                                                    f"_{a}n{q}_{b}n{p}",
                                                    ref_path=paths[f"_base"]
                                                    )
        
        # Restore the original structure
        self.sub_structure = deepcopy(reference)

        if self.verbose:
            print(("\nComputing the displacements for the Hessian"
                  f"using {n_cores} cores"), end=" ... ", flush=True)
        
        if n_cores == 1:
            results = []
            for k, v in tqdm(paths.items(), disable = not self.verbose):
                self.run_calculation(v)
                res = self.parse_output(v)
                results.append((k, res['Energy[SCF]']))
        else:

            # Define the worker function for parallel execution
            global worker
            def worker(kvp : tuple) -> tuple:
                self.run_calculation(kvp[1])
                results = self.parse_output(kvp[1])
                return (kvp[0], results['Energy[SCF]'])
            
            # Run the calculations in parallel
            pool = mp.Pool(n_cores)
            results = list(
                        tqdm(
                            pool.imap_unordered(
                                worker,
                                [(k, v) for k, v in paths.items()]
                            ),
                            total = len(paths),
                            disable = not self.verbose
                        )
                    )
            pool.close()
            pool.join()
            
            # with mp.Pool(n_cores) as pool:
            #     results = pool.map(worker, [(k, v) for k, v in paths.items()])

        if self.verbose:
            print("done.\n", flush=True)

        # Switch back to the base directory
        os.chdir(here)

        # Create an empty Hessian matrix
        hess = np.zeros((n_atoms * 3, n_atoms * 3))

        # Delta in bohr
        delta_bohr = delta * BOHR

        energies = {k: v for k, v in results}
        for a in range(n_atoms):
            for i, q in enumerate('xyz'):
                for b in range(a, n_atoms):
                    for j, p in enumerate('xyz'):

                        # Hessian indices
                        xx = a*3 + i
                        yy = b*3 + j

                        # Get the energies for this particular atom and direction
                        E_pp = energies[f"_{a}p{q}_{b}p{p}"]
                        E_pn = energies[f"_{a}p{q}_{b}n{p}"]
                        E_np = energies[f"_{a}n{q}_{b}p{p}"]
                        E_nn = energies[f"_{a}n{q}_{b}n{p}"]

                        # Finite difference formula
                        hess[xx,yy] = hess[yy,xx] = (
                            E_pp - E_pn - E_np + E_nn
                        ) / (4 * delta_bohr**2)

        return {
            'hessian' : hess,
            'work_dir' : work
        }
    
    def __grad_hess(
            self,
            delta : float = 5e-3 * BOHR2ANG,
            n_cores : int = mp.cpu_count(),
            pre_grad : dict = {}) -> dict:
        """ Method to compute the numerical hessian of the
        given structure, using the QM engine.
        
        This method will compute the numerical hessian of
        the current sub_structure using a cheap finite difference
        approach: it will compute the gradient first, and then use
        only two (instead of four) double displacements per
        atom+coordinate. The calculations will be run in parallel,
        each in its own folder.
        
        Parameters
        ----------
        delta : float
            The displacement to be used for the finite
            difference approach, in Angstrom (1 Bohr = 0.529177 A)
        n_cores : int
            The number of cores to be used for the parallel
            execution of the calculations.
        pre_grad : dict, optional
            If you have a previously calculated gradient, please
            include it here to minimize the number of energy evaluations.
        
        Returns
        -------
        dict
            A dictionary with the Hessian matrix, and
            the path of the working directory.
        """
        # Get the gradient as a starting point
        if not isinstance(pre_grad, dict):
            raise ValueError("QM_driver.num_hessian() The provided gradient "
                             "is not a dictionary.")
        
        here = os.getcwd()
        tmp_dir = tempfile.gettempdir()

        # Create a temporary working directory
        work = os.path.join(tmp_dir, f'QM_Hessian_GH_{int(time.time())}')
        if os.path.exists(work):
            shutil.rmtree(work)
        os.mkdir(work)
        os.chdir(work)
        
        expected = set([
            'gradient',
            'coefficients',
            'delta',
            'points',
            'displacement_energies'
        ])
        if set(list(pre_grad.keys())).intersection(expected) != expected:
            pre_grad = self.num_gradient(delta=delta, n_cores=n_cores)
        
        # Dictionary to keep track of all paths
        paths = {}

        # First, run the base calculation
        paths[f"_base"] = pre_grad['paths'][f"_base"]
        base_results = self.parse_output(paths[f"_base"])
        E_base = base_results['Energy[SCF]']

        # Get the number of atoms, and prepare the displacements
        n_atoms = self.sub_structure.get_num_atoms()

        # Keep a reference to the original structure
        reference = deepcopy(self.sub_structure)

        # Loop over each atom
        for a in range(n_atoms):
            # Loop over all 3 dimensions
            for i, q in enumerate('xyz'):
                # Loop over each atom
                for b in range(a, n_atoms):
                    # Loop over all 3 dimensions
                    for j, p in enumerate('xyz'):

                        # Compute the displacement and create
                        # the inputs for the calculation

                        # Skip self-displacements (the diagonal)
                        if a == b and i == j:
                            continue

                        # Positive-positive
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] += delta
                        self.sub_structure[b][1][j] += delta
                        paths[f"_{a}p{q}_{b}p{p}"] = self.create_input(
                                                    f"_{a}p{q}_{b}p{p}",
                                                    ref_path=paths[f"_base"]
                                                    )

                        # Negative-negative
                        self.sub_structure = deepcopy(reference)
                        self.sub_structure[a][1][i] -= delta
                        self.sub_structure[b][1][j] -= delta
                        paths[f"_{a}n{q}_{b}n{p}"] = self.create_input(
                                                    f"_{a}n{q}_{b}n{p}",
                                                    ref_path=paths[f"_base"]
                                                    )
        
        # Restore the original structure
        self.sub_structure = deepcopy(reference)

        if self.verbose:
            print(("\nComputing the displacements for the Hessian"
                  f"using {n_cores} cores"), end=" ... ", flush=True)
        
        if n_cores == 1:
            results = []
            for k, v in tqdm(paths.items(), disable = not self.verbose):
                self.run_calculation(v)
                res = self.parse_output(v)
                results.append((k, res['Energy[SCF]']))
        else:
        
            # Define the worker function for parallel execution
            global worker
            def worker(kvp : tuple) -> tuple:
                self.run_calculation(kvp[1])
                results = self.parse_output(kvp[1])
                return (kvp[0], results['Energy[SCF]'])
            
            # Run the calculations in parallel
            
            pool = mp.Pool(n_cores)
            results = list(
                        tqdm(
                            pool.imap_unordered(
                                worker,
                                [(k, v) for k, v in paths.items()]
                            ),
                            total = len(paths),
                            disable = not self.verbose
                        )
                    )
            pool.close()
            pool.join()

        if self.verbose:
            print("... done.\n", flush=True)

        # Switch back to the base directory
        os.chdir(here)

        # Create an empty Hessian matrix
        hess = np.zeros((n_atoms * 3, n_atoms * 3))

        # Delta in bohr
        delta_bohr = delta * BOHR

        energies = {k: v for k, v in results}
        for a in range(n_atoms):
            for i, q in enumerate('xyz'):
                for b in range(a, n_atoms):
                    for j, p in enumerate('xyz'):

                        # Skip self-displacements (the diagonal)
                        if a == b and i == j:
                            continue

                        # Hessian indices
                        xx = a*3 + i
                        yy = b*3 + j

                        if hess[xx,yy] != 0.0:
                            continue

                        # Get the energies for this particular atom and direction
                        E_pp = energies[f"_{a}p{q}_{b}p{p}"]
                        E_nn = energies[f"_{a}n{q}_{b}n{p}"]

                        E_p0 = pre_grad['displacement_energies'][f"_{a}_{q}_1"]
                        E_n0 = pre_grad['displacement_energies'][f"_{a}_{q}_-1"]

                        E_0p = pre_grad['displacement_energies'][f"_{b}_{p}_1"]
                        E_0n = pre_grad['displacement_energies'][f"_{b}_{p}_-1"]

                        # Finite difference formula
                        hess[xx,yy] = hess[yy,xx] = (
                            E_pp - E_p0 - E_0p + 2 * E_base - E_n0 - E_0n + E_nn
                        ) / (2 * delta_bohr**2)
        
        # Create the diagonal of the matrix
        hess_diag = np.zeros(n_atoms * 3)
        for a in range(n_atoms):
            for i, q in enumerate('xyz'):

                E_p = pre_grad['displacement_energies'][f"_{a}_{q}_1"]
                E_n = pre_grad['displacement_energies'][f"_{a}_{q}_-1"]

                hess_diag[a*3 + i] = (E_p - 2 * E_base + E_n) / (delta_bohr**2)

        # Symmetrical matrix
        hess = hess + np.diag(hess_diag)

        return {
            'hessian' : hess,
            'work_dir' : work
        }

    @abstractmethod
    def create_input(self,
                     x_name : str = "",
                     ref_path : str = "") -> str:
        """ Abstract method to create the input files for the QM calculation
        
        This method is supposed to create the input file(s) required
        for the QM calculation. The path to the directory where the
        input files have been placed should be returned.

        Parameters
        ----------
        x_name : str
            Additional name to add to this particular run
        ref_path : str
            Path to a reference directory where some files could be
            copied from
        
        Returns
        -------
        work : str
            The path to the directory where the input files have been
            placed, and where the QM calculation should be executed
        """
        pass

    @abstractmethod
    def run_calculation(self, wd : str) -> None:
        """ Abstract method to run the QM calculation
        
        This method is supposed to run the actual QM calculation
        using the input file created by create_input and located
        at the working directory provided as argument.
        
        Parameters
        ----------
        wd : str
            The path to the directory where the input file has been
            placed, and where the QM calculation should be executed
        """
        pass

    @abstractmethod
    def parse_output(self, wd : str) -> dict:
        """ Abstract method to parse the output files of the QM calculation
        
        This method is supposed to read the output files generated by
        the QM calculation and return a dictionary with the parsed
        values.
        
        Parameters
        ----------
        wd : str
            The path to the directory where the output files are
            located
        
        Returns
        -------
        results : dict
            A dictionary with the parsed values from the output
            files
        """
        pass

    def execute(self) -> dict:
        """ Method to execute a QM calculation
        
        This method will do a QM calculation from start to
        finish, including generating the input files, and
        parsing the output files.

        Returns
        -------
        result : dict
            A dictionary with all the results of the QM
            calculation
        """
        # Preparing and creating the input files
        path = self.create_input()
        
        # Run the calculation
        self.run_calculation(path)
        
        # Parse the results from the calculation
        result = self.parse_output(path)

        return result
    
    def num_gradient(
                self,
                delta : float = 5e-3 * BOHR2ANG,
                points : int = 1,
                n_cores : int = mp.cpu_count()) -> dict:
        """ Method to compute the numerical gradient of the
        given structure, using the QM engine.

        This method will compute the numerical gradient of
        the current sub_structure using a finite difference
        approach. The calculations will be run in parallel,
        each in its own folder.

        Parameters
        ----------
        delta : float
            The displacement to be used for the finite
            difference approach, in Angstrom (1 Bohr = 0.529177 A)
        points : int
            The number of points to be used in each direction.
            For example, if points = 1, the displacements will
            be -delta, 0, +delta. If points = 2, the displacements
            will be -2*delta, -delta, 0, +delta, +2*delta.
        n_cores : int
            The number of cores to be used for the parallel
            execution of the calculations.
        
        Returns
        -------
        dict
            A dictionary with the information used to build the
            gradient, and a nx3 matrix containing the 3D vectors
            of the gradient for each atom in the system.
        """

        # Function to create the key for the dictionary
        key_maker = lambda d: int(np.round(d / delta))

        if self.sub_structure.get_num_atoms() < 1:
            raise ValueError("QM_driver.num_gradient() The sub_structure "
                             "must contain at least 1 atom to compute "
                             "a Gradient vector.")
        
        if not isinstance(n_cores, int):
            n_cores = mp.cpu_count()

        if n_cores < 1:
            n_cores = 1

        if n_cores > mp.cpu_count():
            n_cores = mp.cpu_count()

        here = os.getcwd()
        tmp_dir = tempfile.gettempdir()

        # Create a temporary working directory
        work = os.path.join(tmp_dir, f'QM_Gradient_{int(time.time())}')
        if os.path.exists(work):
            shutil.rmtree(work)
        os.mkdir(work)
        os.chdir(work)

        # Dictionary to keep track of all paths
        paths = {}

        # First, run the base calculation
        paths[f"_base"] = self.create_input(f"_base")
        self.run_calculation(paths[f"_base"])
        base_results = self.parse_output(paths[f"_base"])
        base_energy = base_results['Energy[SCF]']

        # Get the number of atoms, and prepare the displacements
        n_atoms = self.sub_structure.get_num_atoms()
        deltas = np.linspace(- delta * points, delta * points, 2 * points + 1)
        deltas = np.delete(deltas, points)  # Remove the zero displacement
        deltas = [float(Fraction(d).limit_denominator()) for d in deltas]

        # Keep a reference to the original structure
        reference = deepcopy(self.sub_structure)

        # Loop over each atom
        for a in range(n_atoms):
            # Loop over all 3 dimensions
            for i, q in enumerate('xyz'):
                # Loop over all displacements
                for d in deltas:

                    # Create a copy of the reference structure
                    self.sub_structure = deepcopy(reference)

                    # Compute the displacement
                    self.sub_structure[a][1][i] += d

                    # Create the displacement key
                    kd = key_maker(d)

                    # Create the inputs for the calculation
                    paths[f"_{a}_{q}_{kd}"] = self.create_input(
                                                    f"_{a}_{q}_{kd}",
                                                    ref_path=paths[f"_base"]
                                                    )
        
        # Restore the original structure
        self.sub_structure = deepcopy(reference)

        if self.verbose:
            print(("\nComputing the displacements for the Hessian"
                  f"using {n_cores} cores"), end=" ... ", flush=True)
        if n_cores == 1:

            # Run the calculations sequentially
            results = []
            for k, v in tqdm(paths.items(), disable = not self.verbose):
                self.run_calculation(v)
                res = self.parse_output(v)
                results.append((k, res['Energy[SCF]']))
        else:

            # Define the worker function for parallel execution
            global worker
            def worker(kvp : tuple) -> tuple:
                self.run_calculation(kvp[1])
                results = self.parse_output(kvp[1])
                return (kvp[0], results['Energy[SCF]'])
            
            # Run the calculations in parallel
            pool = mp.Pool(n_cores)
            results = list(
                        tqdm(
                            pool.imap_unordered(
                                worker,
                                [(k, v) for k, v in paths.items()]
                            ),
                            total=len(paths),
                            disable = not self.verbose
                        )
                    )
            pool.close()
            pool.join()
        
        if self.verbose:
            print("... done.\n", flush=True)

        # with mp.Pool(n_cores) as pool:
        #     results = pool.map(worker, [(k, v) for k, v in paths.items()])

        # Switch back to the base directory
        os.chdir(here)

        # Compute the gradients
        gradient = np.zeros((n_atoms, 3))

        # All deltas
        x = np.insert(deltas, points, 0.0)

        # All coefficients
        coeffs = [{} for a in range(n_atoms)]

        energies = {k: v for k, v in results}
        for a in range(n_atoms):
            for i, q in enumerate('xyz'):

                # Get the energies for this particular atom and direction
                dim_energies = [energies[f"_{a}_{q}_{key_maker(d)}"] for d in deltas]
                dim_energies.insert(points, base_energy)
                avg_energies = np.average(dim_energies)

                # Variance times the number of points
                SST = np.sum([(e - avg_energies)**2 for e in dim_energies])
                
                # Polynomial fitting to get the gradient
                for p in range(2, 2 * points + 1):
                    reg = np.polyfit(
                                x,
                                y = dim_energies,
                                deg = p,
                                full = True
                        )
                    
                    # Sum of residuals (zero if the polynomial's degree
                    # is points - 1)
                    SSR = float(reg[1]) if len(reg[1]) > 0 else 0

                    # Determination coefficient
                    R2 = 1 - SSR/SST

                    # Criterion to stop
                    if R2 > 0.95:
                        break

                # The gradient is the first derivative at d = 0
                gradient[a][i] = reg[0][-2]

                # Add the coefficients
                coeffs[a][q] = reg[0]

        # Save the gradient to a text file
        self.__save_gradient(gradient, work)

        return {
            'gradient' : gradient,
            'coefficients' : coeffs,
            'delta' : delta,
            'points' : points,
            'displacement_energies' : energies,
            'paths' : paths
        }
    
    def _hess_postprocessing(
                self,
                hess : np.ndarray,
                accuracy : str,
                save : bool = True
                ) -> dict:
        """
        Post-processing of the Hessian matrix.

        This function will mass-weight the Hessian matrix, remove the
        linear dependencies, diagonalize the projected-out coordinates,
        project the Hessian into the internal space, and finally diagonalize
        the projected-out coordinates to compute the frequencies
        and normal modes.

        Parameters
        ----------
        hess : np.ndarray
            The Hessian matrix
        accuracy : str
            The accuracy of the Hessian computation
        save : bool (default = True)
            If True, the Hessian matrix, the Mass-weighted Hessian,
            the Hessian in internal coordinates, the frequencies,
            and the normal modes will be saved

        Returns
        -------
        dict
            A dictionary with the Hessian matrix, the frequencies,
            the normal modes, and the working directory
        """
        # Mass-weighting the Hessian
        nb_atoms = self.sub_structure.get_num_atoms()

        masses = np.zeros(nb_atoms)
        i_sqrt_mass = np.zeros(nb_atoms * 3)
        for a in range(nb_atoms):
            masses[a] = self.sub_structure[a].mass
            for i in range(3):
                i_sqrt_mass[3*a + i] = 1 / np.sqrt(masses[a])
    
        w = i_sqrt_mass[:, np.newaxis]
        W = w @ w.T

        hess_mw = np.multiply(hess, W)
        
        # Get the translations and rotations
        valid, shape = self._compute_trans_rot(masses, False)
        valid = np.vstack(valid).T

        # Gram-Schmidt Orthogonalization via QR algorithm to get the orthonormal basis Q
        Q, R = np.linalg.qr(valid)

        # Remove the "projected-out" coordinates from the identity
        P = np.eye(len(hess_mw)) - Q @ Q.T

        # Diagonalize the projected out coordinates to get the internal eigenvectors
        vals, vecs = np.linalg.eigh(P)

        # Remove the vectors with linear dependence among the projected-out coordinates
        bvecs = vecs[:, vals > 1E-7]

        # Project the hessian into the internal space
        hessian = bvecs.T @ hess_mw @ bvecs

        # Diagonalize the hessian with projected-out coordinates
        eig_val, eig_vec = np.linalg.eigh(hessian)

        # Computing the modes
        modes = bvecs.dot(eig_vec)

        # Create the results folder and move there
        here = os.getcwd()
        results_dir = os.path.join(here, f'Results_{accuracy}_{int(time.time())}')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        
        # self.__save_hessian(W, results_dir, mod_name='masses')

        # Save the frequencies to a text file
        freq = self.__save_frequencies(eig_val, results_dir, save)

        if save:
            # Save the normal modes
            self.__save_normal_modes(modes, results_dir)

            # Save the untouched hessian to a text file
            self.__save_hessian(hess, results_dir)

            # Save the mass-weighted hessian to a text file
            # self.__save_hessian(hess_mw, results_dir, mod_name='mw')

            # Save the projected mass-weighted hessian to a text file
            self.__save_hessian(hessian, results_dir, projected=True)

        return {
            'hessian' : hess,
            'frequencies' : freq,
            'normal_modes' : modes,
        }

    def num_hessian(
            self,
            accuracy : str = "medium",
            delta : float = 5e-3 * BOHR2ANG,
            n_cores : int = mp.cpu_count(),
            pre_grad : dict = {}) -> None:
        r""" Method to compute the numerical hessian of the
        given structure, using the QM engine.
        
        This method will compute the numerical hessian of
        the current sub_structure using a finite difference
        approach. The calculations will be run in parallel,
        each in its own folder. The hessian will be mass-weighted,
        and the frequencies and normal modes will be saved
        to text files in the working directory.
        
        Parameters
        ----------
        accuracy : str
            If "light", a cheaper finite difference approach
            will be used, based on pre-computed gradients.
            If "medium", the standard approach will be used
            (see equations in the notes).
            If "high", the gradient and the standard numerical
            displacements will be used to fit a quadrtic surface
            and get the derivative from it. 
        delta : float
            The displacement to be used for the finite
            difference approach, in Angstrom (1 Bohr = 0.529177 A)
        n_cores : int
            The number of cores to be used for the parallel
            execution of the calculations.
        pre_grad : dict, optional
            If you have a previously calculated gradient, please
            include it here to minimize the number of energy evaluations.
        
        Raises
        ------
        ValueError
            If the sub_structure has less than 2 atoms
        
        Notes
        -----
        The standard method to compute a Hessian via finite differences
        uses the following equation:

        .. math::
            \hat{H}_{x,y} = \frac{
                            E(x + \delta, y + \delta) -
                            E(x + \delta, y - \delta) -
                            E(x - \delta, y + \delta) +
                            E(x - \delta, y - \delta)
                            }
                            {4 \delta^2}
        """
        if self.sub_structure.get_num_atoms() < 2:
            raise ValueError("QM_driver.num_hessian() The sub_structure "
                             "must contain at least 2 atoms to compute "
                             "a Hessian matrix.")

        if not isinstance(n_cores, int):
            n_cores = mp.cpu_count()

        if n_cores < 1:
            n_cores = 1

        if n_cores > mp.cpu_count():
            n_cores = mp.cpu_count()
        
        if accuracy == "light":
            calc = self.__grad_hess(
                            delta = delta,
                            n_cores = n_cores,
                            pre_grad = pre_grad
                        )
        elif accuracy == "medium":
            calc = self.__full_hess(
                            delta = delta,
                            n_cores = n_cores
                        )
        elif accuracy == "high":
            calc = self.__accu_hess(
                            delta = delta,
                            n_cores = n_cores
                        )
        else:
            raise ValueError("QM_driver.num_hessian() The selected "
                             "accuracy is not a valid option.")
        
        
        hess = calc['hessian']
        work = calc['work_dir']

        # Mass-weighting, projecting into internal coords,
        # computing frequencies and normal modes, and saving
        # all results
        processed = self._hess_postprocessing(hess, accuracy)

        processed["work_dir"] = work

        return processed


class ORCA_driver(QM_driver):
    """ Class to run QM calculations in ORCA
    
    Attributes
    ----------
    orca_path : str
        The path of the Orca executable, including the executable itself
    props : dict
        The properties of the calculation:
            - method: level of theory
            - basis: basis set to be used
            - charge: total charge of the system
            - multipl: multiplicity of the system
            - modifiers: type of calculation, hardware specs, etc.
    mol : Molecule
        A molecule which will save an XYZ file to be used by Orca for the
        calculation
    calc_name : str
        The name to be given to the calculation directory
    """

    def __init__(self,
                 path : str,
                 qm_props : dict,
                 sub_structure : Molecule | Collection,
                 calc_name : str = 'ORCA_calculation',
                 verbose : bool = False):
        """ ORCA_driver constructor method
        
        Parameters
        ----------
        path : str
            The path of the Orca executable, including the executable itself
        qm_props : dict
            The calculation's properties (e.g. level of theory, basis set)
        sub_structure : Molecule | Collection
            A Molecule or Collection object to create the inputs for Orca
        calc_name : str
            The name to be given to the calculation directory
        verbose : bool
            If True, the execution of the calculations will be printed
            to the console
        """
        self.orca_path = path
        self.props = qm_props
        self.sub_structure = sub_structure
        self.calc_name = calc_name
        self.verbose = verbose
    
    def create_input(self,
                     x_name : str = "",
                     ref_path : str = "") -> str:
        """ Method to create the input files for the Orca calculation

        Parameters
        ----------
        x_name : str
            Additional name to add to this particular run
        ref_path : str
            Path to a reference directory where some files could be
            copied from
        
        Returns
        -------
        work : str
            The path to the directory where the input files have been
            placed, and where the Orca calculation should be executed
        """
        # First line(s) of the Orca input file
        header = (f'! {self.props["method"]} {self.props["basis"]} DEFGRID3'
                  f' TightSCF {self.props["modifiers"]}\n')
        
        # Specifying the name of the XYZ file, ...
        # ... its charge and multiplicity
        geom = (f'*xyzfile {self.props["charge"]} '
                f'{self.props["multipl"]} geometry.xyz\n')
        
        # Renaming the molecule or collection as "geometry"
        self.sub_structure.name = "geometry"

        # Get the current working directory, and creating the directory
        # for the Orca calculation
        here = os.getcwd()
        work = os.path.join(here, f'{self.calc_name}{x_name}_{int(time.time() * 1000)}')
        if os.path.exists(work):
            shutil.rmtree(work)
        os.mkdir(work)
        os.chdir(work)

        # Copy any reference files if provided
        if ref_path != "":
            if os.path.exists(os.path.join(ref_path, 'input.gbw')):
                shutil.copy(os.path.join(ref_path, 'input.gbw'), 'reference.gbw')
                xtra_input = "%scf\n" + \
                             '    Guess  MORead\n' + \
                             '    MOInp "reference.gbw"\n' + \
                             'end\n'
            else:
                raise FileNotFoundError("ORCA_driver.create_input() "
                                        "The provided reference path "
                                        "does not contain an input.gbw file!")

        # Saving the input file
        with open('input.inp', 'w') as f:
            if ref_path != "":
                f.write(
                    header +
                    xtra_input +
                    geom
                )
            else:
                f.write(header + geom)

        # Saving the geometry as an XYZ file
        self.sub_structure.save_as_xyz()

        # Return to the base directory
        os.chdir(here)
        return work
    
    def run_calculation(self, wd : str) -> None:
        """ Method to actually run the Orca calculation
        
        Raises
        ------
        ChildProcessError
            If the Orca calculation does not end correctly
        
        Parameters
        ----------
        wd : str
            The path of the directory where the Orca calculation
            should be performed
        """
        
        # Create the paths to both input and output files
        inp = os.path.join(wd, 'input.inp')
        out = os.path.join(wd, 'output.out')

        # Switch to the working directory
        os.chdir(wd)

        # Run the calculation
        with open(out, 'w') as g:
            orca_run = run([self.orca_path, inp], stdout=g)
        
        # Switch back to the base directory
        os.chdir(os.path.join(wd, '..'))
        
        # Complain if the process was not finished correctly
        if orca_run.returncode != 0:
            raise ChildProcessError("ORCA_driver.run_calculation() "
                    "Orca didn't finish the calculation correctly!")
    
    def parse_output(self, wd : str) -> dict:
        """ Method to parse the output file from Orca
        
        Parameters
        ----------
        wd : str
            The path of the directory where the Orca calculation
            should be performed
        
        Returns
        -------
        result : dict
            A dictionary with all the results of the Orca
            calculation
        """
        # Create the path to the output file
        out = os.path.join(wd, 'output.out')
        
        # Empty dictionary to store the results
        results = {}

        # Get the geometry and interpret it as a Molecule
        # or as a Collection
        struc = Structure(self.sub_structure.name)
        # Loading the information from the XYZ file
        struc.read_xyz(os.path.join(wd, 'geometry.xyz'))
        # Saving the structure
        results['Geometry'] = struc.get_sub_structure()

        # Creating an empty dictionary for the charges
        results['Charges'] = {}

        # Open the output file and get the data
        with open(out, 'r') as h:
            data = h.readlines()

        # Iterate over all lines
        for i, l in enumerate(data):
            
            # Parse the final energy
            if 'FINAL SINGLE POINT ENERGY' in l:
                temp = l.split()
                results['Final Energy'] = float(temp[-1])
            
            # Parse the SCF energy
            if 'TOTAL SCF ENERGY' in l:
                temp = data[i + 3].split()
                results['Energy[SCF]'] = float(temp[3])

            # Parse the orbital energies
            if 'ORBITAL ENERGIES' in l:
                orb_energs = []
                for j in range(self.sub_structure.get_num_atoms() * 5):
                    try:
                        temp = data[i + 4 + j].split()
                        temp = [
                            float(temp[1]),
                            float(temp[2])
                            ]
                        orb_energs.append(temp)
                    except (ValueError, IndexError) as e:
                        break
                results['OrbitalEnergies[Eh]'] = orb_energs

            # Parse the Mulliken charges
            if 'MULLIKEN ATOMIC CHARGES' in l:
                results['Charges']['Mulliken'] = []
                for j in range(self.sub_structure.get_num_atoms()):
                    temp = data[i + 2 + j].split()
                    results['Charges']['Mulliken'].append([
                                                        temp[1],
                                                        float(temp[3])
                                                        ])

            # Parse the Loewding charges
            if 'LOEWDIN ATOMIC CHARGES' in l:
                results['Charges']['Loewdin'] = []
                for j in range(self.sub_structure.get_num_atoms()):
                    temp = data[i + 2 + j].split()
                    results['Charges']['Loewdin'].append([
                                                        temp[1],
                                                        float(temp[3])
                                                        ])
            
            # Parse the dipole moment
            if 'Total Dipole Moment' in l:
                results['Dipole[Debye]'] = []
                temp = l.split()
                for j in temp[4:]:
                    results['Dipole[Debye]'].append( float(j) * 2.5417464519 )

        return results

class PSI4_driver(QM_driver):
    """ Class to run QM calculations in Psi4
    
    Attributes
    ----------
    psi4_path : str
        The path of the Psi4 executable, including the executable itself
    props : dict
        The properties of the calculation:
            - method: level of theory
            - basis: basis set to be used
            - charge: total charge of the system
            - multipl: multiplicity of the system
            - modifiers: type of calculation, hardware specs, etc.
    mol : Molecule | Collection
        A molecule or collection which will save an XYZ file to be used 
        by Psi4 for the calculation
    """

    def __init__(self,
                 qm_props : dict,
                 sub_structure : Molecule | Collection,
                 calc_name : str = 'PSI4_calculation',
                 verbose : bool = False):
        """ PSI4_driver constructor method
        
        Parameters
        ----------
        qm_props : dict
            The calculation's properties (e.g. level of theory, basis set)
        sub_structure : Molecule | Collection
            A Molecule or Collection object to create the inputs for Psi4
        calc_name : str
            The name to be given to the calculation directory
        verbose : bool
            If True, print messages about the progress of the calculations
        """
        try:
            import psi4
        except ImportError:
            raise ImportError("Psi4_driver requires Psi4 to be installed!")
        
        self.props = qm_props
        self.sub_structure = sub_structure
        self.calc_name = calc_name
        self.verbose = verbose

        self.psi4_geom = ''
        self.final_energy = 0.0
    
    def create_input(self,
                     x_name : str = "",
                     ref_path : str = "") -> None:
        """ Method to create the input files for the Psi4 calculation

        Parameters
        ----------
        x_name : str
            Additional name to add to this particular run
        ref_path : str
            Path to a reference directory where some files could be
            copied from
        """
        # Renaming the molecule as "geometry"
        self.sub_structure.name = "geometry"
        
        # Get the current working directory, and creating the directory
        # for the Psi4 calculation
        here = os.getcwd()
        work = os.path.join(here, f'{self.calc_name}{x_name}_{int(time.time() * 1000)}')
        if os.path.exists(work):
            shutil.rmtree(work)
        os.mkdir(work)
        os.chdir(work)

        # Writing the XYZ file
        self.sub_structure.save_as_xyz()

        # Copy any reference files if provided
        if ref_path != "":
            if os.path.exists(os.path.join(ref_path, 'wavefunction.npy')):
                shutil.copy(
                    os.path.join(ref_path, 'wavefunction.npy'),
                    'ref_wfn.npy'
                )
                xtra_input = 'set guess read\n'
                mod_input = ", restart_file='ref_wfn.npy'"
            else:
                raise FileNotFoundError("Psi4_driver.create_input() "
                                        "The provided reference path does "
                                        "not contain an wavefunction.npy file!")
        else:
            xtra_input = ''
            mod_input = ''
        
        # Create the actual Psi4 input file
        psi4_inp = (
            "# Psi4 calculation launched by "
            "InformalFF\n\nmolecule geometry {\n"
            f" {self.props['charge']} {self.props['multipl']}\n"
        )
        
        # Are there any modifiers
        if len(self.props['modifiers']) != 0:
            mods = f'{self.props["modifiers"]}\n'
        else:
            mods = ''

        # Get all the atoms as a list
        all_atoms = self.sub_structure.get_coords()
        for a in all_atoms:
            psi4_inp += f'{a[0]:>3} {a[1]:18.10f} {a[2]:18.10f} {a[3]:18.10f}\n'
        
        psi4_inp += (' units angstrom\n'
                     f'}}\n\n{xtra_input}'
                     f'set basis {self.props["basis"]}\n'
                     f'{mods}'
                     'd_convergence = 1e-8\n'
                     f'E, wfn = energy("{self.props["method"]}"{mod_input}, '
                     'return_wfn=True)\n'
                     'oeprop(wfn, "MULLIKEN_CHARGES", "LOWDIN_CHARGES", '
                     'title="Psi4 Results")\n'
                     'wfn.to_file("wavefunction")\n'
                    )

        # Write the Psi4 input file
        with open(os.path.join(work, 'input.dat'), 'w') as g:
            g.write(psi4_inp)

        # Switch back to the base directory
        os.chdir(here)

        return work
    
    def run_calculation(self, wd : str) -> None:
        """ Method to run the Psi4 calculation
        
        Parameters
        ----------
        wd : str
            The path of the directory where the Psi4 calculation
            should be performed
        """
        # Create the path to the output file
        out = os.path.join(wd, 'output.dat')

        # Get the current working directory
        here = os.getcwd()

        # Switch to the working directory
        os.chdir(wd)

        # Using subprocess to run Psi4
        with open(out, 'w') as f:
            psi4_run = run(['psi4', 'input.dat'], stdout=f)
        
        # Switch back to the base directory
        os.chdir(here)

    def parse_output(self, wd : str) -> dict:
        """ Method to parse the Psi4 results
        
        Parameters
        ----------
        wd : str
            The path of the directory where the Psi4 calculation
            should be performed
        
        Returns
        -------
        results : dict
            The parsed Psi4 results
        """
        # Create the path to the output file
        out = os.path.join(wd, 'output.dat')

        # Empty dictionary to store the results
        results = {}

        # Get the geometry and interpret it as a Molecule
        # or as a Collection
        struc = Structure(self.sub_structure.name)
        # Loading the information from the XYZ file
        struc.read_xyz(os.path.join(wd, 'geometry.xyz'))
        # Saving the structure
        results['Geometry'] = struc.get_sub_structure()
        
        # Creating an empty dictionary for the charges
        results['Charges'] = {}

        # Parse the results
        with open(out, 'r') as f:
            data = f.readlines()
        
        # Iterate over all lines
        for i, l in enumerate(data):

            # Parse the electronic energy
            if ('Total Energy' in l and
                'Computation Completed' in data[i+2]):
                temp = l.split()
                results['Energy[SCF]'] = float(temp[3])
            
            # Adding the final energy
            if ('Final Energy:' in l and
                '=> Energetics <=' in data[i+2]):
                results['Final Energy'] = float(l.split()[-1])

            # Parse the orbital energies
            if 'Orbital Energies [Eh]' in l:

                preCharges = []
                occ = 0.0

                # Iterate over next lines to get the orbital energies
                for j in range(i+1, len(data)):

                    if 'Doubly Occupied:' in data[j]:
                        occ = 2.0
                    if 'Virtual:' in data[j]:
                        occ = 0.0
                    if 'Final Occupation by Irrep:' in data[j]:
                        break

                    orbLine = data[j].split()

                    if len(orbLine) == 6:
                        temp = [[occ,float(orbLine[k])] for k in range(1,6,2)]
                        preCharges += temp

                # Get the orbital charges
                results['OrbitalEnergies[Eh]'] = preCharges
            
            # Parse the Mulliken charges
            if 'Mulliken Charges: (a.u.)' in l:
                results['Charges']['Mulliken'] = []
                for j in range(self.sub_structure.get_num_atoms()):
                    temp = data[i + 2 + j].split()
                    results['Charges']['Mulliken'].append([
                                                        temp[1],
                                                        float(temp[5])
                                                        ])

            # Parse the Loewding charges
            if 'Lowdin Charges: (a.u.)' in l:
                results['Charges']['Loewdin'] = []
                for j in range(self.sub_structure.get_num_atoms()):
                    temp = data[i + 2 + j].split()
                    results['Charges']['Loewdin'].append([
                                                        temp[1],
                                                        float(temp[5])
                                                        ])

            # Parse the dipole moment
            if 'Multipole Moments:' in l:

                dipole = []

                # Iterate over next lines to get the dipole moment
                for j in range(i+1, len(data)):

                    if len(dipole) == 3:
                        break

                    if 'Dipole' in data[j]:
                        temp = data[j].split()
                        dipole.append(float(temp[-1]) * 2.5417464519)
                
                # Get the dipole moment
                results['Dipole[Debye]'] = dipole

        return results