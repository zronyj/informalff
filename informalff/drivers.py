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
        path = self.create_input()
        self.run_calculation(path)
        result = self.parse_output(path)
        return result

class ORCA_driver(QM_driver):

    def __init__(self,
                 path : str,
                 qm_props : dict,
                 mol : Molecule):
        self.orca_path = path
        self.props = qm_props
        self.molecule = mol        
    
    def create_input(self) -> str:
        header = (f'! {self.props["method"]} {self.props["basis"]}'
                  f' {self.props["modifiers"]}\n')
        geom = (f'*xyzfile {self.props["charge"]} '
                f'{self.props["multipl"]} geometry.xyz\n')
        self.molecule.name = "geometry"

        here = os.getcwd()
        work = os.path.join(here, f'Orca_calculation_{int(time.time())}')
        if os.path.exists(work):
            shutil.rmtree(work)
        os.mkdir(work)
        os.chdir(work)

        with open('input.inp', 'w') as f:
            f.write(header + geom)

        self.molecule.save_as_xyz()

        os.chdir(here)
        return work
    
    def run_calculation(self, wd : str) -> None:
        
        inp = os.path.join(wd, 'input.inp')
        out = os.path.join(wd, 'output.out')

        os.chdir(wd)
        with open(out, 'w') as g:
            orca_run = run([self.orca_path, inp], stdout=g)
        os.chdir(os.path.join(wd, '..'))
        
        # Inform if the process finished correctly
        if orca_run.returncode != 0:
            raise Exception("Orca didn't finish the calculation correctly!")
    
    def parse_output(self, wd : str) -> dict:

        out = os.path.join(wd, 'output.out')
        
        results = {}

        results['geometry'] = Molecule(self.molecule.name)
        results['geometry'].read_xyz(os.path.join(wd, 'geometry.xyz'))

        with open(out, 'r') as h:
            data = h.readlines()

        for l in data:
            if 'FINAL SINGLE POINT ENERGY' in l:
                temp = l.split()
                results['Energy[SPE]'] = float(temp[-1])

        return results