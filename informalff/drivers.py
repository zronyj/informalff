import os                # To navigate the file system
import shutil            # To do some operation over the file system
import time              # A way to keep track of time
import warnings          # To throw warnings instead of raising errors
from subprocess import run # Method to run external commands
from multiprocessing import Pool # To parallelize jobs
from abc import ABC, abstractmethod # To be able to create several drivers
from .molecule import Molecule
from .collection import Collection

class QM_driver(ABC):
    """ Factory class for all QM drivers
    
    This class is just supposed to work as a base
    class for other drivers. The immportant methods are
    the constructor and the execute methods.
    """

    def __init__(self):
        pass

    @abstractmethod
    def create_input(self) -> str:
        pass

    @abstractmethod
    def run_calculation(self, wd : str) -> None:
        pass

    @abstractmethod
    def parse_output(self, wd : str) -> dict:
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

class ORCA_driver(QM_driver):
    """ Class to run QM calculations in ORCA
    
    Attriutes
    ---------
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
    """

    def __init__(self,
                 path : str,
                 qm_props : dict,
                 mol : Molecule):
        """ ORCA_driver constructor method
        
        Parameters
        ----------
        path : str
            The path of the Orca executable, including the executable itself
        qm_props : dict
            The calculation's properties (e.g. level of theory, basis set)
        mol : Molecule
            A Molecule object to create the inputs for Orca
        """
        self.orca_path = path
        self.props = qm_props
        self.molecule = mol        
    
    def create_input(self) -> str:
        """ Method to create the input files for the Orca calculation
        
        Returns
        -------
        work : str
            The path to the directory where the input files have been
            placed, and where the Orca calculation should be executed
        """
        # First line(s) of the Orca input file
        header = (f'! {self.props["method"]} {self.props["basis"]}'
                  f' {self.props["modifiers"]}\n')
        
        # Specifying the name of the XYZ file, ...
        # ... its charge and multiplicity
        geom = (f'*xyzfile {self.props["charge"]} '
                f'{self.props["multipl"]} geometry.xyz\n')
        
        # Renaming the molecule as "geometry"
        self.molecule.name = "geometry"

        # Get the current working directory, and creating the directory
        # for the Orca calculation
        here = os.getcwd()
        work = os.path.join(here, f'Orca_calculation_{int(time.time())}')
        if os.path.exists(work):
            shutil.rmtree(work)
        os.mkdir(work)
        os.chdir(work)

        # Saving the input file
        with open('input.inp', 'w') as f:
            f.write(header + geom)

        # Saving the geometry as an XYZ file
        self.molecule.save_as_xyz()

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

        # Creating empty Molecule object
        results['Geometry'] = Molecule(self.molecule.name)
        # Loading the information from the XYZ file
        results['Geometry'].read_xyz(os.path.join(wd, 'geometry.xyz'))
        # Creating an empty dictionary for the charges
        results['Charges'] = {}

        # Open the output file and get the data
        with open(out, 'r') as h:
            data = h.readlines()

        # Iterate over all lines
        for i, l in enumerate(data):
            
            # Parse the electronic energy
            if 'FINAL SINGLE POINT ENERGY' in l:
                temp = l.split()
                results['Energy[SPE]'] = float(temp[-1])

            # Parse the orbital energies
            if 'ORBITAL ENERGIES' in l:
                orb_energs = []
                for j in range(self.molecule.get_num_atoms() * 5):
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
                for j in range(self.molecule.get_num_atoms()):
                    temp = data[i + 2 + j].split()
                    results['Charges']['Mulliken'].append([
                                                        temp[1],
                                                        float(temp[3])
                                                        ])

            # Parse the Loewding charges
            if 'LOEWDIN ATOMIC CHARGES' in l:
                results['Charges']['Loewdin'] = []
                for j in range(self.molecule.get_num_atoms()):
                    temp = data[i + 2 + j].split()
                    results['Charges']['Loewdin'].append([
                                                        temp[1],
                                                        float(temp[3])
                                                        ])
            
            # Parse the dipole moment
            if 'Total Dipole Moment' in l:
                temp = l.split()
                results['Dipole'] = [ float(j) for j in temp[4:] ]

        return results