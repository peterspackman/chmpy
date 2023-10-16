import re

SUBSCRIPT_MAP = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "a": "ₐ",
    "b": "♭",
    "c": "꜀",
    "d": "ᑯ",
    "e": "ₑ",
    "f": "բ",
    "g": "₉",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "q": "૧",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "w": "w",
    "x": "ₓ",
    "y": "ᵧ",
    "z": "₂",
    "A": "ₐ",
    "B": "₈",
    "C": "C",
    "D": "D",
    "E": "ₑ",
    "F": "բ",
    "G": "G",
    "H": "ₕ",
    "I": "ᵢ",
    "J": "ⱼ",
    "K": "ₖ",
    "L": "ₗ",
    "M": "ₘ",
    "N": "ₙ",
    "O": "ₒ",
    "P": "ₚ",
    "Q": "Q",
    "R": "ᵣ",
    "S": "ₛ",
    "T": "ₜ",
    "U": "ᵤ",
    "V": "ᵥ",
    "W": "w",
    "X": "ₓ",
    "Y": "ᵧ",
    "Z": "Z",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
}


def subscript(x: str) -> str:
    """
    Convert the provided string to its subscript
    equivalent in unicode

    Args:
        x (str): the string to be converted

    Returns:
        str: the converted string
    """
    return SUBSCRIPT_MAP.get(x, x)


def overline(x: str) -> str:
    """
    Add a unicode overline modifier
    to the provided string.

    Args:
        x (str): the string to be overlined

    Returns:
        str: the overlined string
    """
    return f"\u0305{x}"


def natural_sort_key(s: str, _nsre=re.compile(r"([a-zA-Z]+)(\d+)")):
    """
    Utility function for sorting strings of the form A1, B_2, A12
    etc. so that the suffixes will be in numeric order rather than
    lexicographical order.

    Args:
        s (str): the string whose sort key to determine

    Returns:
        tuple: the (str, int) natural sort key for the provided string
    """
    m = _nsre.match(s)
    if not m:
        return s
    c, i = m.groups(0)
    return (c, int(i))
