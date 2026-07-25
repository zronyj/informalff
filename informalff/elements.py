import os                                 # To navigate the file system
from numpy import nan                     # To handle missing values
from json import load                     # To load the geometries from a JSON file    
from pandas import read_csv               # To manage tables and databases
from pathlib import Path                  # To locate files in the file system
from warnings import warn, filterwarnings # To issue warnings to the user
from dataclasses import dataclass, field  # To create data classes

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
periodic_data.replace(nan, "", inplace=True)
PERIODIC_TABLE = periodic_data.set_index("Symbol")
all_symbols = set(PERIODIC_TABLE.index.to_list())

# Load the geometries from the JSON file
with open(os.path.join(here, "data", "geometries.json"), "r") as f:
    raw_geometries = load(f)

filterwarnings("ignore")

@dataclass(frozen=True)
class Element:
    """Class to represent an element in the periodic table
    
    Attributes
    ----------
    symbol : str
        The symbol of the element (e.g. "H" for hydrogen, "C" for carbon, etc.)
    name : str
        The name of the element (e.g. "Hydrogen", "Carbon", etc.)
    number : int
        The atomic number of the element (e.g. 1 for hydrogen, 6 for carbon, etc.)
    mass : float
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
        cr = PERIODIC_TABLE.loc[self.symbol, "CovalentRadius"]
        vdw = PERIODIC_TABLE.loc[self.symbol, "AtomicRadius"]

        # Check if the value exists
        cr = float(cr) / 100 if cr != "" else 0.0
        vdw = float(vdw) / 100 if vdw != "" else 0.0

        if cr == 0.0 or vdw == 0.0:
            warn("Atom.__get_atomic_radius(): "
                 f"Atomic radius not found for {self.symbol}")

        return cr, vdw

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
        en = float(en) if en != "" else 0.0

        if en == 0.0:
            warn("Element.__get_electronegativity(): "
                 f"Electronegativity not found for {self.symbol}")

        # Check if the value exists
        return en
    
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
                return [int(oxs)]
            # Maybe not, and we need to split it into a list
            except ValueError:
                return [int(ox) for ox in oxs.split(",")]
        else:
            warn("Element.__get_oxidation_states(): "
                 f"Oxidation states not found for {self.symbol}")
            return []

    def __parse_electron_configuration(self) -> dict:
        """
        Get the electron configuration of an element.

        Returns
        -------
        electron_configuration : dict
            Electron configuration of the element.
        """
        # Possible shells
        shells = 'spdfg'

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
            for shell in shells:
                if shell in orb_type:
                    # Add the orbital type and the number of electrons
                    num_shell, num_electrons = orb_type.split(shell)
                    subshell = num_shell + shell
                    electron_configuration[subshell] = int(num_electrons)
                    break
            
        return electron_configuration

@dataclass(frozen=True)
class Geometry:
    """Class to represent an atomic geometry
    
    Attributes
    ----------
    char : str
        A character representing the geometry (e.g. "T" for tetrahedral,
        "S" for square planar, etc.)
    geometry : str
        The name of the geometry (e.g. "Tetrahedral",
        "Square Planar", etc.)
    angles : tuple
        A tuple of the ideal bond angles in the geometry
        (e.g. (109.5,) for tetrahedral, (90, 180) for square planar, etc.)
    """
    char: str
    geometry: str
    angles: tuple

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

class AtomGeometry(dict):
    """
    Class to represent the possible atomic geometries as a frozen dictionary
    
    The keys are the coordination number and the number of
    lone electron pairs (e.g. (4, 0) for a tetrahedral geometry,
    (4, 1) for a seesaw geometry, etc.) and the values are
    Geometry objects containing the properties of the geometries.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_frozen", True)
    
    def __setitem__(self, key, value):
        if getattr(self, "_frozen", False):
            raise TypeError("AtomGeometry is immutable")
        super().__setitem__(key, value)
    
    def __delitem__(self, key):
        if getattr(self, "_frozen", False):
            raise TypeError("AtomGeometry is immutable")
        super().__delitem__(key)
    
    def pop(self, *args, **kwargs):
        if getattr(self, "_frozen", False):
            raise TypeError("AtomGeometry is immutable")
        return super().pop(*args, **kwargs)
    
    def clear(self):
        if getattr(self, "_frozen", False):
            raise TypeError("AtomGeometry is immutable")
        super().clear()
    
    def update(self, *args, **kwargs):
        if getattr(self, "_frozen", False):
            raise TypeError("AtomGeometry is immutable")
        super().update(*args, **kwargs)
    
    def __setattr__(self, key, value):
        if getattr(self, "_frozen", False):
            raise TypeError("AtomGeometry is immutable")
        super().__setattr__(key, value)
    
    def __str__(self):
        output = "-" * 80 + "\n"
        output += f"| ({'Coordination':^14}, {'Lone Pairs':^12}) | "
        output += f"{'ID':^5} | {'Geometry':^20} | {'Angles':^12} |\n"
        output += "-" * 80 + "\n"
        for (coordination, lone_pairs), geometry in self.items():
            output += f"| ({coordination:^14}, {lone_pairs:^12}) | "
            output += f"{geometry.char:^5} | {geometry.geometry:>20} | "
            output += f"{','.join(str(a) for a in geometry.angles):>12} |\n"
        output += "-" * 80
        return output

pre_PTE = {}
for symbol in all_symbols:
    data = PERIODIC_TABLE.loc[symbol]
    i_energy = data["IonizationEnergy"]
    e_affinity = data["ElectronAffinity"]
    element = Element(
        symbol = symbol,
        name = data["Name"],
        number = int(data["AtomicNumber"]),
        mass = float(data["AtomicMass"]),
        ionization_energy = float(i_energy) if i_energy != "" else 0.0,
        electron_affinity = float(e_affinity) if e_affinity != "" else 0.0
    )
    pre_PTE[symbol] = element

PTE = PeriodicTable(pre_PTE)
"""This is the periodic table as a frozen dictionary of Element objects.

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

# Create a geometries dictionary
geometries = {}
for geom in raw_geometries:
    for config in geom["configuration"]:
        key = (config["coordination"], config["lone_pairs"])
        geometries[key] = Geometry(
            char=geom["key"],
            geometry=geom["geometry"],
            angles=tuple(config["angles"])
        )

# Create an atom geometries dictionary
GEOMETRIES = AtomGeometry(geometries)
"""This is a frozen dictionary of the possible atomic geometries.

The values are Geometry objects containing the properties of the geometries.
The keys are the coordination number and the number of lone electron pairs
(e.g. (4, 0) for a tetrahedral geometry).

Each Geometry object contains:
- char: A character representing the geometry (e.g. "T" for tetrahedral)
- geometry: The name of the geometry (e.g. "Tetrahedral")
- angles: The angles of the geometry in degrees (e.g. [109.5, 109.5, 109.5])
"""

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("This library was not intended as a standalone program.")