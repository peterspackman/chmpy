"""ORCA input writing and output parsing.

Ghost atoms (basis functions without nucleus or electrons) are written the
ORCA way, as the element symbol followed by a colon: `C : x y z`.

Geometry optimisation constraints, relaxed scans and fragments go in a
`GeomSettings`, passed as `to_orca_input(..., geom=...)`. Atom indices are
0-based, as in ORCA; distances are in Angstrom and angles in degrees.

    geom = GeomSettings(
        constraints=[Bond(0, 1), Angle(2, 0, 3, 120.0), Fixed(range(10, 18))],
        scans=[Scan(Dihedral(0, 1, 2, 3), 180, 360, 19)],
    )
    to_orca_input(elements, positions, "B3LYP def2-SVP Opt", geom=geom)

Results ORCA writes for machines -- `<job>.property.txt` and `<job>.hess`,
with UV-vis, ECD, IR and Raman spectra -- are read by `chmpy.fmt.orca_results`.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chmpy.opt.constraints import (  # noqa: F401  re-exported for input writing
    Angle,
    Bond,
    Dihedral,
    Fixed,
    check_atoms,
)
from chmpy.templates import load_template

LOG = logging.getLogger(__name__)
ORCA_INPUT_TEMPLATE = load_template("orca_input")

_FINAL_ENERGY = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_NORMAL_TERMINATION = "ORCA TERMINATED NORMALLY"


def _read(path_or_text):
    """The text itself, or the contents of the file it names (None if missing).

    A string containing a newline is taken to be text; anything else a path.
    """
    if isinstance(path_or_text, str) and "\n" in path_or_text:
        return path_or_text
    path = Path(path_or_text)
    if not path.is_file():
        return None
    return path.read_text(errors="replace")


def parse_final_energy(path_or_text):
    """The last FINAL SINGLE POINT ENERGY, in Hartree, of a run that finished.

    Truncated or crashed runs can still contain SCF energies, so the energy is
    only returned when ORCA reports normal termination.

    Args:
        path_or_text: path to an ORCA output, or its contents

    Returns:
        float, or None if the file is missing, the run did not terminate
        normally, or no final energy was printed
    """
    text = _read(path_or_text)
    if text is None or _NORMAL_TERMINATION not in text:
        return None
    energies = _FINAL_ENERGY.findall(text)
    return float(energies[-1]) if energies else None


def parse_input_geometry(path_or_text):
    """The `* xyz charge mult ... *` block of an ORCA input.

    Args:
        path_or_text: path to an ORCA input, or its contents

    Returns:
        (elements, positions, ghost): element symbols, (n, 3) positions in
        Angstrom, and whether each atom is a ghost
    """
    text = _read(path_or_text)
    if text is None:
        raise FileNotFoundError(f"no ORCA input at {path_or_text}")
    lines = iter(text.splitlines())
    for line in lines:
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0] == "*" and tokens[1].lower() == "xyz":
            break
    else:
        raise ValueError("no '* xyz' geometry block in the ORCA input")

    elements, positions, ghost = [], [], []
    for line in lines:
        tokens = line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if tokens[0] == "*":
            break
        is_ghost = tokens[0].endswith(":") or (len(tokens) > 1 and tokens[1] == ":")
        symbol = tokens[0].rstrip(":")
        coordinates = [t for t in tokens[1:] if t != ":"][:3]
        elements.append(symbol)
        positions.append([float(x) for x in coordinates])
        ghost.append(is_ghost)
    else:
        raise ValueError("the ORCA geometry block is not closed with '*'")
    return elements, np.array(positions, dtype=float).reshape(-1, 3), ghost


def to_orca_input(
    elements,
    positions,
    keywords,
    blocks="",
    nprocs=8,
    maxcore=3000,
    charge=0,
    multiplicity=1,
    ghost=(),
    geom=None,
):
    """An ORCA input with an inline Cartesian geometry.

    Args:
        elements: element symbols (or `Element`s)
        positions: (n, 3) positions in Angstrom
        keywords: the simple input line, without the leading `!`
        blocks: extra `%` blocks, written verbatim, e.g. "%mdci STol 5e-5 end"
        nprocs: MPI processes (`%pal`)
        maxcore: memory per process in MB
        charge: total charge
        multiplicity: spin multiplicity
        ghost: indices of atoms to write as ghosts
        geom: `GeomSettings` with constraints, scans or fragments, written as
            `%geom` (and `%frag`, after the coordinates, as ORCA requires)

    Returns:
        str: the input file contents
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(elements) != len(positions):
        raise ValueError(f"{len(elements)} elements but {len(positions)} positions")
    ghost = set(ghost)
    lines = []
    for index, (element, (x, y, z)) in enumerate(zip(elements, positions, strict=True)):
        label = f"{element} :" if index in ghost else f"{element}"
        lines.append(f"{label}{x:15.8f}{y:15.8f}{z:15.8f}")
    blocks = blocks.strip()
    trailing = ""
    if geom is not None:
        for name in ("%geom", "%frag"):
            if name in blocks.lower():
                raise ValueError(f"`blocks` already has a {name} block; use geom=")
        geom_block, trailing = geom.render(positions, keywords)
        blocks = "\n".join(b for b in (blocks, geom_block) if b)
    return (
        ORCA_INPUT_TEMPLATE.render(
            keywords=keywords.strip(),
            maxcore=maxcore,
            nprocs=nprocs,
            blocks=blocks,
            charge=charge,
            multiplicity=multiplicity,
            geometry="\n".join(lines),
            trailing=trailing,
        )
        + "\n"
    )


def counterpoise_inputs(mol_a, mol_b, keywords, **kwargs):
    """Boys-Bernardi counterpoise inputs for the dimer of two molecules.

    `E_int = E(AB) - E(A) - E(B)`, where A and B are each computed in the full
    dimer basis, the partner's atoms written as ghosts.

    Args:
        mol_a, mol_b: `Molecule`s, with their own charge and multiplicity
        keywords: the simple input line
        **kwargs: passed to `to_orca_input` (blocks, nprocs, maxcore). The
            dimer's multiplicity must be given as `multiplicity` unless both
            monomers are singlets.

    Returns:
        dict with keys "AB", "A" and "B"
    """
    multiplicity = kwargs.pop("multiplicity", None)
    if multiplicity is None:
        if mol_a.multiplicity != 1 or mol_b.multiplicity != 1:
            raise ValueError(
                "the dimer multiplicity is ambiguous for open-shell monomers; "
                "pass multiplicity="
            )
        multiplicity = 1
    kwargs.pop("charge", None)
    kwargs.pop("ghost", None)

    n_a = len(mol_a)
    elements = [str(e) for e in mol_a.elements] + [str(e) for e in mol_b.elements]
    positions = np.vstack([mol_a.positions, mol_b.positions])
    in_a = range(n_a)
    in_b = range(n_a, len(elements))
    return {
        "AB": to_orca_input(
            elements,
            positions,
            keywords,
            charge=mol_a.charge + mol_b.charge,
            multiplicity=multiplicity,
            **kwargs,
        ),
        "A": to_orca_input(
            elements,
            positions,
            keywords,
            charge=mol_a.charge,
            multiplicity=mol_a.multiplicity,
            ghost=in_b,
            **kwargs,
        ),
        "B": to_orca_input(
            elements,
            positions,
            keywords,
            charge=mol_b.charge,
            multiplicity=mol_b.multiplicity,
            ghost=in_a,
            **kwargs,
        ),
    }


# -- geometry optimisation settings ------------------------------------------

#: a target this far from the starting geometry earns a warning; ORCA's manual
#: asks for targets close to the start, Angstrom and degrees
FAR_BOND = 0.3
FAR_ANGLE = 30.0

#: ORCA scans at most this many coordinates in one calculation
MAX_SCANS = 3


@dataclass(frozen=True)
class Scan:
    """A relaxed scan of one coordinate, over `start` to `stop` in `steps`
    points or over an explicit list of `values`."""

    coordinate: object
    start: float | None = None
    stop: float | None = None
    steps: int | None = None
    values: tuple | None = None

    def __post_init__(self):
        if self.coordinate.value is not None:
            raise ValueError("a scanned coordinate takes no target value")
        ranged = (self.start, self.stop, self.steps)
        if self.values is not None:
            if any(x is not None for x in ranged):
                raise ValueError("give either start/stop/steps or values, not both")
            object.__setattr__(self, "values", tuple(float(v) for v in self.values))
            if len(self.values) < 2:
                raise ValueError("a scan needs at least two values")
        elif any(x is None for x in ranged):
            raise ValueError("a scan needs start, stop and steps, or values")
        elif self.steps < 2:
            raise ValueError("a scan needs at least two steps")

    @property
    def n_points(self):
        return len(self.values) if self.values is not None else self.steps


@dataclass
class GeomSettings:
    """Constraints, relaxed scans and fragments for an ORCA optimisation.

    Attributes:
        constraints: `Bond`, `Angle`, `Dihedral` and `Fixed` entries
        scans: up to three `Scan`s; nested loops unless `simultaneous`
        simultaneous: scan all coordinates together (equal step counts)
        fragments: lists of atom indices, one per fragment, together covering
            every atom once; numbered from 1 in the order given
        constrain_fragments: fragment numbers whose internal coordinates are
            all held (ConstrainFragments), e.g. rigid monomers in a cluster
        rigid_fragments: fragment numbers moved as rigid bodies (RigidFrags;
            needs RIGIDBODYOPT in the keywords)
        options: further `%geom` keywords, e.g. {"MaxIter": 200}
    """

    constraints: list = field(default_factory=list)
    scans: list = field(default_factory=list)
    simultaneous: bool = False
    fragments: list | None = None
    constrain_fragments: list = field(default_factory=list)
    rigid_fragments: list = field(default_factory=list)
    options: dict = field(default_factory=dict)

    def render(self, positions, keywords=""):
        """The `%geom` block and the `%frag` block that follows the coordinates.

        Args:
            positions: (n, 3) positions in Angstrom, to check indices and
                targets against
            keywords: the simple input line, to check what the settings need

        Returns:
            (geom block, frag block); either may be empty
        """
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        self._check(positions, keywords)

        lines = []
        if self.constraints:
            lines.append("  Constraints")
            lines.extend(f"    {entry}" for entry in self._constraint_lines())
            lines.append("  end")
        for scan in self.scans:
            lines.append(f"  Scan {_scan_line(scan)} end")
        if self.simultaneous:
            lines.append("  Simul_Scan true")
        if self.constrain_fragments:
            lines.append(
                f"  ConstrainFragments {_braced(self.constrain_fragments)} end"
            )
        if self.rigid_fragments:
            lines.append(f"  RigidFrags {_braced(self.rigid_fragments)} end")
        for key, value in self.options.items():
            lines.append(f"  {key} {_option(value)}")
        geom = "\n".join(["%geom", *lines, "end"]) if lines else ""

        frag = ""
        if self.fragments is not None:
            definitions = [
                f"    {number} {{{' '.join(_runs(atoms))}}} end"
                for number, atoms in enumerate(self.fragments, start=1)
            ]
            frag = "\n".join(["%frag", "  Definition", *definitions, "  end", "end"])
        return geom, frag

    def _constraint_lines(self):
        for entry in self.constraints:
            if isinstance(entry, Fixed):
                codes = (
                    ["C"] if entry.axes == "xyz" else [a.upper() for a in entry.axes]
                )
                for run in _runs(entry.atoms):
                    for code in codes:
                        yield f"{{ {code} {run} C }}"
            else:
                yield f"{{ {_coordinate(entry)} C }}"

    def _check(self, positions, keywords):
        n = len(positions)
        internals = [c for c in self.constraints if not isinstance(c, Fixed)]
        scanned = [s.coordinate for s in self.scans]
        check_atoms([*self.constraints, *scanned], n)

        for entry in internals:
            if entry.value is not None:
                _warn_if_far(entry, positions)

        held = {_key(c) for c in internals}
        clash = [s for s in scanned if _key(s) in held]
        if clash:
            raise ValueError(f"{clash[0]} is both constrained and scanned")
        if len({_key(s) for s in scanned}) != len(scanned):
            raise ValueError("the same coordinate is scanned twice")
        if len(self.scans) > MAX_SCANS:
            raise ValueError(f"ORCA scans at most {MAX_SCANS} coordinates at once")
        if self.simultaneous:
            if len(self.scans) < 2:
                raise ValueError("a simultaneous scan needs at least two scans")
            if len({s.n_points for s in self.scans}) != 1:
                raise ValueError("simultaneous scans need equal numbers of points")
        self._check_fragments(n, keywords)

    def _check_fragments(self, n, keywords):
        if self.fragments is None:
            if self.constrain_fragments or self.rigid_fragments:
                raise ValueError("fragment options need `fragments`")
            return
        assigned = [int(a) for atoms in self.fragments for a in atoms]
        if sorted(assigned) != list(range(n)):
            repeated = sorted({a for a in assigned if assigned.count(a) > 1})
            missing = sorted(set(range(n)) - set(assigned))
            raise ValueError(
                f"fragments must cover each of the {n} atoms once "
                f"(missing {missing}, repeated {repeated})"
            )
        numbers = range(1, len(self.fragments) + 1)
        for name in ("constrain_fragments", "rigid_fragments"):
            unknown = [f for f in getattr(self, name) if f not in numbers]
            if unknown:
                raise ValueError(
                    f"{name} names fragments {unknown}; they are numbered "
                    f"1-{len(self.fragments)}"
                )
        if self.rigid_fragments and "RIGIDBODYOPT" not in keywords.upper():
            raise ValueError("rigid_fragments needs RIGIDBODYOPT in the keywords")


def _coordinate(entry):
    "'B 0 1', with the target value if there is one"
    text = f"{entry.code} {' '.join(str(a) for a in entry.atoms)}"
    return text if entry.value is None else f"{text} {entry.value:g}"


def _scan_line(scan):
    coordinate = _coordinate(scan.coordinate)
    if scan.values is not None:
        return f"{coordinate} [{' '.join(f'{v:g}' for v in scan.values)}]"
    return f"{coordinate} = {scan.start:g}, {scan.stop:g}, {scan.steps}"


def _key(coordinate):
    "The same coordinate read in either direction compares equal"
    atoms = coordinate.atoms
    return (coordinate.code, min(atoms, atoms[::-1]))


def _warn_if_far(coordinate, positions):
    current = coordinate.measure(positions)
    difference = abs(coordinate.value - current) % 360.0
    if isinstance(coordinate, Dihedral):
        difference = min(difference, 360.0 - difference)
    limit = FAR_BOND if isinstance(coordinate, Bond) else FAR_ANGLE
    if difference > limit:
        LOG.warning(
            "%s targets %g but starts at %.3f; ORCA expects targets close to "
            "the starting geometry",
            _coordinate(coordinate),
            coordinate.value,
            current,
        )


def _runs(atoms):
    "Indices as ORCA ranges: [0, 1, 2, 5] -> ['0:2', '5']"
    atoms = sorted({int(a) for a in atoms})
    runs, start = [], None
    for index, atom in enumerate(atoms):
        if start is None:
            start = atom
        if index + 1 == len(atoms) or atoms[index + 1] != atom + 1:
            runs.append(f"{start}:{atom}" if atom > start else f"{start}")
            start = None
    return runs


def _braced(numbers):
    return "{ " + " ".join(str(int(n)) for n in numbers) + " }"


def _option(value):
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)
