import os                                # To navigate the file system
from pandas import read_csv              # To manage tables and databases
from pathlib import Path                 # To locate files in the file system
from dataclasses import dataclass, field # To create data classes

# ------------------------------------------------------- #
#              Setting up the Periodic Table              #
# ------------------------------------------------------- #
# National Center for Biotechnology Information. "Periodic Table of Elements"
# PubChem, https://pubchem.ncbi.nlm.nih.gov/periodic-table/.
# Accessed 20 February, 2024.
# ------------------------------------------------------- #
here = Path(globals().get("__file__", "./_")).absolute().parent
pte_file = os.path.join(here, "data", "PubChemElements_all.csv")
periodic_data = read_csv(pte_file)
PERIODIC_TABLE = periodic_data.set_index("Symbol")
all_symbols = set(PERIODIC_TABLE.index.to_list())

@dataclass(frozen=True)
class Element:
    """Class to represent an element in the periodic table
    
    Attributes
    ----------
    symbol : str
        The symbol of the element (e.g. "H" for hydrogen, "C" for carbon, etc.)
    name : str
        The name of the element (e.g. "Hydrogen", "Carbon", etc.)
    atomic_number : int
        The atomic number of the element (e.g. 1 for hydrogen, 6 for carbon, etc.)
    atomic_mass : float
        The atomic mass of the element (e.g. 1.008 for hydrogen, 12.011 for carbon, etc.)
    covalent_radius : float
        The covalent radius of the element in Ångstroms (e.g. 0.31 for hydrogen, 0.76 for carbon, etc.)
    vdw_radius : float
        The van der Waals radius of the element in Ångstroms (e.g. 1.20 for hydrogen, 1.70 for carbon, etc.)
    electronegativity : float
        The electronegativity of the element (e.g. 2.20 for hydrogen, 2.55 for carbon, etc.)
    ionization_energy : float
        The ionization energy of the element in eV (e.g. 13.598 for hydrogen, 11.260 for carbon, etc.)
    electron_affinity : float
        The electron affinity of the element in eV (e.g. 0.754 for hydrogen, 1.262 for carbon, etc.)
    oxidation_states : list
        The oxidation states of the element (e.g. [-1, 0, 1] for hydrogen, [-4, -3, -2, -1, 0, 1, 2, 3, 4] for carbon, etc.)
    electron_configuration : dict
        The electron configuration of the element (e.g. {"1s": 1 for hydrogen, {"1s": 2, "2s": 2, "2p": 2} for carbon, etc.)
    """
    symbol: str
    name: str
    number: int
    mass: float
    ionization_energy: float
    electron_affinity: float
    covalent_radius: float = field(init=False)
    vdw_radius: float = field(init=False)
    electronegativity: float = field(init=False)
    oxidation_states: list = field(init=False)
    electron_configuration: dict = field(init=False)

    def __post_init__(self):
        cov_rad, vdw_rad = self.__get_atomic_radii()
        object.__setattr__(self,
                           "covalent_radius",
                           cov_rad)
        object.__setattr__(self,
                           "vdw_radius",
                           vdw_rad)
        object.__setattr__(self,
                           "electronegativity",
                           self.__get_electronegativity())
        object.__setattr__(self,
                           "oxidation_states",
                           self.__get_oxidation_states())
        object.__setattr__(self,
                           "electron_configuration",
                           self.__parse_electron_configuration())
    
    def __get_atomic_radii(self) -> tuple:
        """
        Get the atomic radius of an atom of the given element.

        Returns
        -------
        atomic_radius : float
            Atomic radius of the atom.
        vdw_radius : float
            Van der Waals radius of the atom.
        """
        # Get the atomic radius of the atom
        # Convert from pm to Ångstrom
        cr = PERIODIC_TABLE.loc[self.symbol, "CovalentRadius"] / 100
        vdw = PERIODIC_TABLE.loc[self.symbol, "AtomicRadius"] / 100

        # Check if the value exists
        if cr != "":
            return cr, vdw
        else:
            raise ValueError(
                    "Atom.__get_atomic_radius(): "
                    f"Atomic radius not found for {self.symbol}")

    def __get_electronegativity(self) -> float:
        """
        Get the electronegativity of an element.

        Returns
        -------
        electronegativity : float
            Electronegativity of the element.
        """
        # Get the electronegativity of the element
        en = PERIODIC_TABLE.loc[self.symbol, "Electronegativity"]

        # Check if the value exists
        if en != "":
            return float(en)
        else:
            raise ValueError("Element.__get_electronegativity(): "
                             f"Electronegativity not found for {self.symbol}")
    
    def __get_oxidation_states(self) -> list:
        """
        Get the oxidation states of an element.

        Returns
        -------
        oxidation_states : list
            Oxidation states of the element.
        """
        # Get the oxidation states of the element
        oxs = PERIODIC_TABLE.loc[self.symbol, "OxidationStates"]

        # Check if the value exists
        if oxs != "":
            # Maybe it is a single oxidation state already
            try:
                return float(oxs)
            # Maybe not, and we need to split it into a list
            except ValueError:
                return [int(ox) for ox in oxs.split(",")]
        else:
            raise ValueError("Element.__get_oxidation_states(): "
                             f"Oxidation states not found for {self.symbol}")

    def __parse_electron_configuration(self) -> dict:
        """
        Get the electron configuration of an element.

        Returns
        -------
        electron_configuration : dict
            Electron configuration of the element.
        """
        # Get the electron configuration of the element
        ec = PERIODIC_TABLE.loc[self.symbol, "ElectronConfiguration"]

        # If it's not H or He, remove the lower shells
        if "]" in ec:
            ec = ec.split("]")[1]
        
        # If the electron configuration is not fully defined
        if "(" in ec:
            ec = ec.split("(")[0]
        
        # Split the electron configuration into orbital types
        orb_types = ec.split(" ")

        # Remove any empty strings
        orb_types = [orb for orb in orb_types if orb != ""]

        # Create a dictionary with the orbital types as keys
        # and the number of electrons as values
        electron_configuration = {}
        for orb_type in orb_types:
            electron_configuration[orb_type] = int(orb_type[-1])
            
        return electron_configuration

class PeriodicTable(dict):
    """
    Class to represent the periodic table as a frozen dictionary
    
    The keys are the element symbols (e.g. "H" for hydrogen,
    "C" for carbon, etc.) and the values are Element objects
    containing the properties of the elements.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_frozen", True)
    
    def __setitem__(self, key, value):
        if getattr(self, "_frozen", False):
            raise TypeError("PeriodicTable is immutable")
        super().__setitem__(key, value)
    
    def __delitem__(self, key):
        if getattr(self, "_frozen", False):
            raise TypeError("PeriodicTable is immutable")
        super().__delitem__(key)
    
    def pop(self, *args, **kwargs):
        if getattr(self, "_frozen", False):
            raise TypeError("PeriodicTable is immutable")
        return super().pop(*args, **kwargs)
    
    def clear(self):
        if getattr(self, "_frozen", False):
            raise TypeError("PeriodicTable is immutable")
        super().clear()
    
    def update(self, *args, **kwargs):
        if getattr(self, "_frozen", False):
            raise TypeError("PeriodicTable is immutable")
        super().update(*args, **kwargs)
    
    def __setattr__(self, key, value):
        if getattr(self, "_frozen", False):
            raise TypeError("PeriodicTable is immutable")
        super().__setattr__(key, value)

pre_PTE = {}
for symbol in all_symbols:
    data = PERIODIC_TABLE.loc[symbol]
    element = Element(
        symbol=symbol,
        name=data["Name"],
        number=data["AtomicNumber"],
        mass=data["AtomicMass"],
        ionization_energy=data["IonizationEnergy"],
        electron_affinity=data["ElectronAffinity"]
    )
    pre_PTE[symbol] = element

PTE = PeriodicTable(**pre_PTE)
"""
This is the periodic table as a frozen dictionary of Element objects.

The keys are the element symbols (e.g. "H" for hydrogen,
"C" for carbon, etc.) and the values are Element objects
containing the properties of the elements.

Each Element object contains the following properties:
- symbol: The symbol of the element (e.g. "H" for hydrogen)
- name: The name of the element (e.g. "Hydrogen")
- atomic_number: The atomic number of the element (e.g. 1 for hydrogen)
- atomic_mass: The atomic mass of the element (e.g. 1.008 for hydrogen)
- covalent_radius: The covalent radius of the element
                    in Ångstroms (e.g. 0.31 for hydrogen)
- vdw_radius: The van der Waals radius of the element
                in Ångstroms (e.g. 1.20 for hydrogen)
- electronegativity: The electronegativity of the element
                        (e.g. 2.20 for hydrogen)
- ionization_energy: The ionization energy of the element in eV
                        (e.g. 13.598 for hydrogen)
- electron_affinity: The electron affinity of the element in eV
                        (e.g. 0.754 for hydrogen)
- oxidation_states: The oxidation states of the element
                    (e.g. [-1, 0, 1] for hydrogen)
- electron_configuration: The electron configuration of the element
                            as a dictionary (e.g. {"1s": 1} for hydrogen)
"""

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("This library was not intended as a standalone program.")