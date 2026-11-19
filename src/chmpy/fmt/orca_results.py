"""ORCA's machine-readable results: `<job>.property.txt` and `<job>.hess`.

The property file is ORCA's record of a run: a sequence of `$Name ... $End`
blocks, each a list of typed fields,

    &NAtoms [&Type "Integer"] 4
    &ExcitationEnergies [&Type "ArrayOfDoubles", &Dim (6,11)]
    &CartesianCoordinates [&Type "Coordinates", &Dim(4,4), &Units "Bohr"]

with arrays printed in chunks of columns under a row of column indices.
`parse_property_file` reads every block generically. A block name can repeat
-- once per geometry, or once per representation -- so blocks come back as a
list in file order. Electronic spectra (`absorption_spectrum`,
`ecd_spectrum`) are read from it.

The Hessian file holds the vibrational analysis, including the IR and Raman
intensities, which the property file does not; `parse_hess_file` reads it and
`vibrational_spectrum` gives the frequencies, normal modes and intensities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_FIELD = re.compile(r"^\s*&(\w+)\s*(?:\[(.*?)\])?\s*(.*?)\s*$")
_TYPE = re.compile(r'&Type\s*"(\w+)"')
_DIM = re.compile(r"&Dim\s*\((\d+)\s*,\s*(\d+)\)")
_UNITS = re.compile(r'&Units\s*"([^"]*)"')
_QUOTED = re.compile(r'"([^"]*)"')


@dataclass
class PropertyBlock:
    """One `$Name ... $End` block.

    Attributes:
        name: the block name, e.g. "CIS_Absorption_Spectrum"
        values: field name to value -- int, float, bool, str, a numpy array,
            or for coordinates an (elements, positions) pair
        units: field name to the units ORCA gave, where it gave any
        comments: field name to the description ORCA printed, where it did
    """

    name: str
    values: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    comments: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.values[key]

    def __contains__(self, key):
        return key in self.values

    def get(self, key, default=None):
        return self.values.get(key, default)

    @property
    def geometry_index(self) -> int | None:
        return self.values.get("GeometryIndex")

    def __repr__(self) -> str:
        return f"<PropertyBlock {self.name}: {', '.join(self.values)}>"


def _text(path_or_text) -> str:
    if isinstance(path_or_text, str) and "\n" in path_or_text:
        return path_or_text
    return Path(path_or_text).read_text(errors="replace")


def parse_property_file(path_or_text) -> list[PropertyBlock]:
    """Every block of an ORCA property file, in file order.

    Args:
        path_or_text: path to `<job>.property.txt`, or its contents

    Returns:
        list of `PropertyBlock`
    """
    lines = _text(path_or_text).splitlines()
    blocks, block, i = [], None, 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        i += 1
        if stripped.startswith("$"):
            name = stripped[1:].split()[0] if len(stripped) > 1 else ""
            if name == "End":
                block = None
            else:
                block = PropertyBlock(name)
                blocks.append(block)
            continue
        if block is None or not stripped.startswith("&"):
            continue
        match = _FIELD.match(line)
        if match is None:
            continue
        key, meta, rest = match.groups()
        kind = _TYPE.search(meta).group(1) if meta and _TYPE.search(meta) else None
        units = _UNITS.search(meta) if meta else None
        if units:
            block.units[key] = units.group(1)
        dim = _DIM.search(meta) if meta else None
        if kind in ("ArrayOfDoubles", "ArrayOfIntegers"):
            shape = (int(dim.group(1)), int(dim.group(2)))
            value, i = _array(lines, i, shape, int if "Integers" in kind else float)
            comment = _QUOTED.search(rest)
        elif kind == "Coordinates":
            rows = int(dim.group(1))
            value = _coordinates(lines[i : i + rows])
            i += rows
            comment = None
        else:
            value, comment = _scalar(kind, rest)
        block.values[key] = value
        if comment:
            block.comments[key] = comment.group(1)
    return blocks


def _scalar(kind, rest):
    "A scalar field's value, and the match for any trailing description"
    if kind == "String":
        quoted = list(_QUOTED.finditer(rest))
        value = quoted[0].group(1) if quoted else ""
        return value, quoted[1] if len(quoted) > 1 else None
    token, _, tail = rest.partition(" ")
    comment = _QUOTED.search(tail)
    if kind == "Boolean":
        return token.lower() == "true", comment
    if kind == "Double":
        return float(token), comment
    return int(token), comment


def _array(lines, i, shape, kind):
    """A (rows, columns) array printed in column chunks, and the next line.

    Each chunk is a row of column indices, a blank line, then one line per
    row: its index followed by that chunk's values.
    """
    rows, columns = shape
    values = np.zeros(shape, dtype=kind)
    filled = 0
    while filled < columns:
        while not lines[i].strip():
            i += 1
        chunk = [int(c) for c in lines[i].split()]
        i += 1
        read = 0
        while read < rows:
            tokens = lines[i].split()
            i += 1
            if not tokens:
                continue
            values[int(tokens[0]), chunk] = [kind(float(t)) for t in tokens[1:]]
            read += 1
        filled += len(chunk)
    return values, i


def _coordinates(lines):
    "(element symbols, (n, 3) positions) in the units the field states"
    elements, positions = [], []
    for line in lines:
        tokens = line.split()
        elements.append(tokens[0])
        positions.append([float(x) for x in tokens[1:4]])
    return elements, np.array(positions)


def find_blocks(blocks, suffix) -> list[PropertyBlock]:
    "Blocks whose name is `suffix` or ends with `_suffix`, in file order"
    return [b for b in blocks if b.name == suffix or b.name.endswith(f"_{suffix}")]


# -- line shapes ---------------------------------------------------------------


def broaden(centres, intensities, grid, fwhm, shape="gaussian") -> np.ndarray:
    """Sum of unit-area line shapes, each scaled by its intensity.

    Args:
        centres: (n,) line positions
        intensities: (n,) line intensities
        grid: (m,) points to evaluate at, in the units of `centres`
        fwhm: full width at half maximum, in the same units
        shape: "gaussian" or "lorentzian"

    Returns:
        (m,) the broadened spectrum
    """
    grid = np.asarray(grid, dtype=float)[:, None]
    offset = grid - np.asarray(centres, dtype=float)[None, :]
    if shape == "gaussian":
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        lines = np.exp(-0.5 * (offset / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    elif shape == "lorentzian":
        gamma = 0.5 * fwhm
        lines = gamma / (np.pi * (offset**2 + gamma**2))
    else:
        raise ValueError(f"shape must be 'gaussian' or 'lorentzian', not {shape!r}")
    return lines @ np.asarray(intensities, dtype=float)


# -- electronic spectra --------------------------------------------------------

#: column layout of ORCA's spectrum arrays: energy in eV, cm^-1 and nm, the
#: intensity, then for absorption |D|^2, then (real, imaginary) x, y, z of the
#: transition moment
_ENERGY_COLUMNS = (0, 1, 2)
_INTENSITY_COLUMN = 3


@dataclass
class ElectronicSpectrum:
    """Excitations from the ground state, as ORCA reports them.

    Attributes:
        kind: "absorption" or "ecd"
        method: the block prefix, e.g. "CIS" for CIS/TDA/TD-DFT
        representation: "Length" or "Velocity"
        energies: (n,) excitation energies in eV
        wavenumbers: (n,) in cm^-1
        wavelengths: (n,) in nm
        intensities: (n,) oscillator strengths (absorption) or rotatory
            strengths in 1e-40 cgs (ecd)
        moments: (n, 3) real part of the transition dipole (absorption, from
            the electric or velocity operator) or of the magnetic transition
            dipole (ecd), atomic units
        final_states: (n,) index of each excited state
    """

    kind: str
    method: str
    representation: str
    energies: np.ndarray
    wavenumbers: np.ndarray
    wavelengths: np.ndarray
    intensities: np.ndarray
    moments: np.ndarray
    final_states: np.ndarray

    def profile(self, grid, fwhm=0.3, shape="gaussian"):
        """The spectrum broadened onto `grid`, in eV.

        Args:
            grid: (m,) energies in eV
            fwhm: line width in eV
            shape: "gaussian" or "lorentzian"
        """
        return broaden(self.energies, self.intensities, grid, fwhm, shape)

    def __len__(self) -> int:
        return len(self.energies)

    def __repr__(self) -> str:
        return (
            f"<ElectronicSpectrum {self.kind} {self.method} {self.representation}, "
            f"{len(self)} transitions>"
        )


def _electronic(path_or_blocks, suffix, kind, moment_start, method, representation):
    blocks = _blocks(path_or_blocks)
    candidates = [
        b
        for b in find_blocks(blocks, suffix)
        if b.get("Representation", "").lower() == representation.lower()
        and (method is None or b.name == f"{method}_{suffix}")
    ]
    if not candidates:
        found = sorted({b.name for b in find_blocks(blocks, suffix)})
        raise ValueError(
            f"no {representation} {suffix} block"
            + (f" for {method}" if method else "")
            + (f"; found {found}" if found else "")
        )
    block = candidates[-1]
    table = block["ExcitationEnergies"]
    ev, wavenumber, nm = (table[:, c] for c in _ENERGY_COLUMNS)
    states = block.get("States")
    return ElectronicSpectrum(
        kind=kind,
        method=block.name[: -len(suffix) - 1] if block.name != suffix else "",
        representation=block["Representation"],
        energies=ev,
        wavenumbers=wavenumber,
        wavelengths=nm,
        intensities=table[:, _INTENSITY_COLUMN],
        moments=table[:, moment_start : moment_start + 6 : 2],
        final_states=states[:, 2] if states is not None else np.arange(1, len(ev) + 1),
    )


def absorption_spectrum(path_or_blocks, representation="Length", method=None):
    """The UV-vis absorption spectrum: oscillator strengths per excitation.

    Args:
        path_or_blocks: a property file path or text, or parsed blocks
        representation: "Length" (transition electric dipoles) or "Velocity"
        method: block prefix, e.g. "CIS" (CIS, TDA and TD-DFT); by default the
            last absorption block of the requested representation

    Returns:
        ElectronicSpectrum
    """
    return _electronic(
        path_or_blocks, "Absorption_Spectrum", "absorption", 5, method, representation
    )


def ecd_spectrum(path_or_blocks, representation="Length", method=None):
    """The electronic circular dichroism spectrum: rotatory strengths.

    Args: as for `absorption_spectrum`

    Returns:
        ElectronicSpectrum
    """
    return _electronic(path_or_blocks, "ECD_Spectrum", "ecd", 4, method, representation)


def _blocks(path_or_blocks):
    if isinstance(path_or_blocks, list):
        return path_or_blocks
    return parse_property_file(path_or_blocks)


# -- the Hessian file and vibrational spectra ---------------------------------

#: frequency below which a mode is a projected translation or rotation, cm^-1
ZERO_FREQUENCY = 1e-6


def parse_hess_file(path_or_text) -> dict:
    """The blocks of an ORCA `.hess` file.

    Known blocks are decoded: "hessian" and "normal_modes" to arrays,
    "vibrational_frequencies" to a vector, "atoms" to (elements, masses,
    positions in Bohr), "dipole_derivatives", "polarizability_derivatives",
    "ir_spectrum" and "raman_spectrum" to row arrays, and the single-value
    blocks to numbers. Anything else is kept as its list of lines.

    Args:
        path_or_text: path to a `.hess` file, or its contents

    Returns:
        dict of block name (without the `$`) to its value
    """
    sections, name = {}, None
    for line in _text(path_or_text).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("$"):
            name = stripped[1:]
            sections[name] = []
            continue
        if name is not None and stripped:
            sections[name].append(stripped)
    sections.pop("end", None)
    return {name: _hess_block(name, lines) for name, lines in sections.items()}


def _hess_block(name, lines):
    if name in ("hessian", "normal_modes"):
        dims = [int(x) for x in lines[0].split()]
        shape = (dims[0], dims[-1])
        values, _ = _array(lines[1:], 0, shape, float)
        return values
    if name == "vibrational_frequencies":
        return np.array([float(line.split()[1]) for line in lines[1:]])
    if name == "atoms":
        rows = [line.split() for line in lines[1:]]
        return (
            [row[0] for row in rows],
            np.array([float(row[1]) for row in rows]),
            np.array([[float(x) for x in row[2:5]] for row in rows]),
        )
    if name in (
        "dipole_derivatives",
        "polarizability_derivatives",
        "ir_spectrum",
        "raman_spectrum",
    ):
        return np.array([[float(x) for x in line.split()] for line in lines[1:]])
    if len(lines) == 1:
        token = lines[0].split()[0]
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return lines[0]
    return lines


@dataclass
class VibrationalSpectrum:
    """Harmonic frequencies with their IR and Raman intensities.

    Only vibrations are kept: the projected translations and rotations, which
    ORCA reports at exactly zero, are dropped. Imaginary modes are kept, as
    negative frequencies.

    Attributes:
        frequencies: (n,) in cm^-1
        modes: (3N, n) normal modes, one per column, as ORCA writes them
            (Cartesian, each column of unit length)
        mode_indices: (n,) the index of each mode among all 3N
        ir_intensities: (n,) km/mol, or None if no IR data
        transition_dipoles: (n, 3) dipole derivative along each mode
            including the vibrational overlap, atomic units, or None
        raman_activities: (n,) A^4/amu, or None if no Raman data
        depolarization: (n,) depolarisation ratios, or None
    """

    frequencies: np.ndarray
    modes: np.ndarray
    mode_indices: np.ndarray
    ir_intensities: np.ndarray | None = None
    transition_dipoles: np.ndarray | None = None
    raman_activities: np.ndarray | None = None
    depolarization: np.ndarray | None = None

    @property
    def imaginary(self) -> np.ndarray:
        "Imaginary frequencies, as negative cm^-1"
        return self.frequencies[self.frequencies < 0]

    def ir_profile(self, grid, fwhm=10.0, shape="lorentzian"):
        """The IR spectrum broadened onto `grid` (cm^-1), in km/mol per cm^-1"""
        if self.ir_intensities is None:
            raise ValueError("this calculation has no IR intensities")
        return broaden(self.frequencies, self.ir_intensities, grid, fwhm, shape)

    def raman_profile(self, grid, fwhm=10.0, shape="lorentzian"):
        """Raman activities broadened onto `grid` (cm^-1)"""
        if self.raman_activities is None:
            raise ValueError(
                "this calculation has no Raman activities; they need the "
                "polarisability derivatives, e.g. %elprop Polar 1 end with Freq"
            )
        return broaden(self.frequencies, self.raman_activities, grid, fwhm, shape)

    def __len__(self) -> int:
        return len(self.frequencies)

    def __repr__(self) -> str:
        has = [
            name
            for name, value in (
                ("IR", self.ir_intensities),
                ("Raman", self.raman_activities),
            )
            if value is not None
        ]
        return (
            f"<VibrationalSpectrum {len(self)} modes"
            f"{', ' + ' and '.join(has) if has else ''}"
            f"{f', {len(self.imaginary)} imaginary' if len(self.imaginary) else ''}>"
        )


def vibrational_spectrum(path_or_hess) -> VibrationalSpectrum:
    """Frequencies, normal modes and IR/Raman intensities from a `.hess` file.

    Args:
        path_or_hess: path to `<job>.hess`, its text, or `parse_hess_file`'s
            result

    Returns:
        VibrationalSpectrum
    """
    hess = (
        path_or_hess
        if isinstance(path_or_hess, dict)
        else parse_hess_file(path_or_hess)
    )
    if "vibrational_frequencies" not in hess:
        raise ValueError("this Hessian file has no vibrational frequencies")
    frequencies = hess["vibrational_frequencies"]
    keep = np.flatnonzero(np.abs(frequencies) > ZERO_FREQUENCY)
    spectrum = VibrationalSpectrum(
        frequencies=frequencies[keep],
        modes=hess["normal_modes"][:, keep],
        mode_indices=keep,
    )
    ir = hess.get("ir_spectrum")
    if ir is not None and len(ir):
        spectrum.ir_intensities = ir[keep, 2]
        spectrum.transition_dipoles = ir[keep, 3:6]
    raman = hess.get("raman_spectrum")
    if raman is not None and len(raman):
        spectrum.raman_activities = raman[keep, 1]
        spectrum.depolarization = raman[keep, 2]
    return spectrum
