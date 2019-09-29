import numpy as np

_ELEMENT_DATA = (
    # name symbol cov vdw mass
    ("hydrogen", "H", 0.23, 1.09, 1.00794),
    ("helium", "He", 1.50, 1.40, 4.002602),
    ("lithium", "Li", 1.28, 1.82, 6.941),
    ("beryllium", "Be", 0.96, 2.00, 9.012182),
    ("boron", "B", 0.83, 2.00, 10.811),
    ("carbon", "C", 0.68, 1.70, 12.0107),
    ("nitrogen", "N", 0.68, 1.55, 14.0067),
    ("oxygen", "O", 0.68, 1.52, 15.9994),
    ("fluorine", "F", 0.64, 1.47, 18.998403),
    ("neon", "Ne", 1.50, 1.54, 20.1797),
    ("sodium", "Na", 1.66, 2.27, 22.98977),
    ("magnesium", "Mg", 1.41, 1.73, 24.305),
    ("aluminium", "Al", 1.21, 2.00, 26.981538),
    ("silicon", "Si", 1.20, 2.10, 28.0855),
    ("phosphorus", "P", 1.05, 1.80, 30.973761),
    ("sulfur", "S", 1.02, 1.80, 32.065),
    ("chlorine", "Cl", 0.99, 1.75, 35.453),
    ("argon", "Ar", 1.51, 1.88, 39.948),
    ("potassium", "K", 2.03, 2.75, 39.0983),
    ("calcium", "Ca", 1.76, 2.00, 40.078),
    ("scandium", "Sc", 1.70, 2.00, 44.95591),
    ("titanium", "Ti", 1.60, 2.00, 47.867),
    ("vanadium", "V", 1.53, 2.00, 50.9415),
    ("chromium", "Cr", 1.39, 2.00, 51.9961),
    ("manganese", "Mn", 1.61, 2.00, 54.938049),
    ("iron", "Fe", 1.52, 2.00, 55.845),
    ("cobalt", "Co", 1.26, 2.00, 58.9332),
    ("nickel", "Ni", 1.24, 1.63, 58.6934),
    ("copper", "Cu", 1.32, 1.40, 63.546),
    ("zinc", "Zn", 1.22, 1.39, 65.409),
    ("gallium", "Ga", 1.22, 1.87, 69.723),
    ("germanium", "Ge", 1.17, 2.00, 72.64),
    ("arsenic", "As", 1.21, 1.85, 74.9216),
    ("selenium", "Se", 1.22, 1.90, 78.96),
    ("bromine", "Br", 1.21, 1.85, 79.904),
    ("krypton", "Kr", 1.50, 2.02, 83.798),
    ("rubidium", "Rb", 2.20, 2.00, 85.4678),
    ("strontium", "Sr", 1.95, 2.00, 87.62),
    ("yttrium", "Y", 1.90, 2.00, 88.90585),
    ("zirconium", "Zr", 1.75, 2.00, 91.224),
    ("niobium", "Nb", 1.64, 2.00, 92.90638),
    ("molybdenum", "Mo", 1.54, 2.00, 95.94),
    ("technetium", "Tc", 1.47, 2.00, 98.0),
    ("ruthenium", "Ru", 1.46, 2.00, 101.07),
    ("rhodium", "Rh", 1.45, 2.00, 102.9055),
    ("palladium", "Pd", 1.39, 1.63, 106.42),
    ("silver", "Ag", 1.45, 1.72, 107.8682),
    ("cadmium", "Cd", 1.44, 1.58, 112.411),
    ("indium", "In", 1.42, 1.93, 114.818),
    ("tin", "Sn", 1.39, 2.17, 118.71),
    ("antimony", "Sb", 1.39, 2.00, 121.76),
    ("tellurium", "Te", 1.47, 2.06, 127.6),
    ("iodine", "I", 1.40, 1.98, 126.90447),
    ("xenon", "Xe", 1.50, 2.16, 131.293),
    ("caesium", "Cs", 2.44, 2.00, 132.90545),
    ("barium", "Ba", 2.15, 2.00, 137.327),
    ("lanthanum", "La", 2.07, 2.00, 138.9055),
    ("cerium", "Ce", 2.04, 2.00, 140.116),
    ("praseodymium", "Pr", 2.03, 2.00, 140.90765),
    ("neodymium", "Nd", 2.01, 2.00, 144.24),
    ("promethium", "Pm", 1.99, 2.00, 145.0),
    ("samarium", "Sm", 1.98, 2.00, 150.36),
    ("europium", "Eu", 1.98, 2.00, 151.964),
    ("gadolinium", "Gd", 1.96, 2.00, 157.25),
    ("terbium", "Tb", 1.94, 2.00, 158.92534),
    ("dysprosium", "Dy", 1.92, 2.00, 162.5),
    ("holmium", "Ho", 1.92, 2.00, 164.93032),
    ("erbium", "Er", 1.89, 2.00, 167.259),
    ("thulium", "Tm", 1.90, 2.00, 168.93421),
    ("Ytterbium", "Yb", 1.87, 2.00, 173.04),
    ("lutetium", "Lu", 1.87, 2.00, 174.967),
    ("hafnium", "Hf", 1.75, 2.00, 178.49),
    ("tantalum", "Ta", 1.70, 2.00, 180.9479),
    ("tungsten", "W", 1.62, 2.00, 183.84),
    ("rhenium", "Re", 1.51, 2.00, 186.207),
    ("osmium", "Os", 1.44, 2.00, 190.23),
    ("iridium", "Ir", 1.41, 2.00, 192.217),
    ("platinum", "Pt", 1.36, 1.72, 195.078),
    ("gold", "Au", 1.50, 1.66, 196.96655),
    ("mercury", "Hg", 1.32, 1.55, 200.59),
    ("thallium", "Tl", 1.45, 1.96, 204.3833),
    ("lead", "Pb", 1.46, 2.02, 207.2),
    ("bismuth", "Bi", 1.48, 2.00, 208.98038),
    ("polonium", "Po", 1.40, 2.00, 290.0),
    ("astatine", "At", 1.21, 2.00, 210.0),
    ("radon", "Rn", 1.50, 2.00, 222.0),
    ("francium", "Fr", 2.60, 2.00, 223.0),
    ("radium", "Ra", 2.21, 2.00, 226.0),
    ("actinium", "Ac", 2.15, 2.00, 227.0),
    ("thorium", "Th", 2.06, 2.00, 232.0381),
    ("protactinium", "Pa", 2.00, 2.00, 231.03588),
    ("uranium", "U", 1.96, 1.86, 238.02891),
    ("neptunium", "Np", 1.90, 2.00, 237.0),
    ("plutonium", "Pu", 1.87, 2.00, 244.0),
    ("americium", "Am", 1.80, 2.00, 243.0),
    ("curium", "Cm", 1.69, 2.00, 247.0),
    ("berkelium", "Bk", 1.54, 2.00, 247.0),
    ("californium", "Cf", 1.83, 2.00, 251.0),
    ("einsteinium", "Es", 1.50, 2.00, 252.0),
    ("fermium", "Fm", 1.50, 2.00, 257.0),
    ("mendelevium", "Md", 1.50, 2.00, 258.0),
    ("nobelium", "No", 1.50, 2.00, 259.0),
    ("lawrencium", "Lr", 1.50, 2.00, 262.0),
)

_EL_FROM_SYM = {
    s: (i, n, s, rcov, rvdw, m)
    for i, (n, s, rcov, rvdw, m) in enumerate(_ELEMENT_DATA, start=1)
}


def cov_radii(atomic_numbers):
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 103):
        raise ValueError("All elements must be atomic numbers between [1,103]")
    return np.array([_ELEMENT_DATA[i - 1][2] for i in atomic_numbers])

def vdw_radii(atomic_numbers):
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 103):
        raise ValueError("All elements must be atomic numbers between [1,103]")
    return np.array([_ELEMENT_DATA[i - 1][3] for i in atomic_numbers])

def element_names(atomic_numbers):
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 103):
        raise ValueError("All elements must be atomic numbers between [1,103]")
    return [_ELEMENT_DATA[i - 1][0] for i in atomic_numbers]

def element_symbols(atomic_numbers):
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 103):
        raise ValueError("All elements must be atomic numbers between [1,103]")
    return [_ELEMENT_DATA[i - 1][1] for i in atomic_numbers]

def atomic_number(el):
    symbol = el.strip().capitalize()
    if el.isdigit():
        return int(el)
    if symbol not in _EL_FROM_SYM:
        msg = "Unknown element: '{}'".format(el)
        raise ValueError(msg)
    return _EL_FROM_SYM[symbol][0]
