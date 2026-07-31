"""Loading structures and ensembles, with clear errors about what is supported.

`Crystal.load` and `Molecule.load` are separate entry points that raise a bare
`KeyError` on anything they do not recognise, and the caller has to know in
advance which one to use. Often you do not - a file named `geometry.in` may be
either - so this module dispatches on its own and always returns the same
thing: a list of frames.

One file may hold several structures - a trajectory, a multi-block CIF - and
several files may be viewed as one ensemble, so a single structure is just the
one-frame case rather than a separate path through the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chmpy.core.molecule import Molecule
from chmpy.crystal.crystal import Crystal
from chmpy.util.optional import require

#: formats that always describe a periodic structure
CRYSTAL_SUFFIXES = {".cif": "CIF", ".res": "SHELX", ".gen": "DFTB+ gen",
                    ".vasp": "VASP", ".pdb": "PDB"}
CRYSTAL_NAMES = {"POSCAR": "VASP", "CONTCAR": "VASP"}
#: formats that always describe an isolated molecule
MOLECULE_SUFFIXES = {".sdf": "MDL SDF", ".mol2": "SYBYL mol2",
                     ".fchk": "Gaussian formatted checkpoint",
                     ".coord": "Turbomole"}
#: may be either, decided by whether the file carries lattice vectors
EITHER_SUFFIXES = {".in": "FHI-aims geometry.in", ".next_step": "FHI-aims"}
EITHER_NAMES = {"geometry.in": "FHI-aims", "geometry.in.next_step": "FHI-aims"}
#: may hold many frames
TRAJECTORY_SUFFIXES = {".xyz": "XMol XYZ (single frame or trajectory)",
                       ".extxyz": "extended XYZ", ".traj": "ASE trajectory"}


#: names a structure reports when the file gave it none worth showing
GENERIC_NAMES = {"Molecule", "Crystal", "GENERIC_NAME"}


@dataclass
class Frame:
    """One structure out of a possibly larger ensemble.

    `source` is the file it came from. The other two are both "which one is
    this", but they answer different questions and so cannot share a field: a
    CIF data block *names a structure*, while a trajectory comment *describes
    a state* of one structure. Only the former belongs in a title.
    """

    structure: object
    source: str
    label: str = ""  # what this frame is a snapshot of: a trajectory comment
    block: str = ""  # what names this structure in the file: a CIF data block

    @property
    def is_crystal(self):
        return isinstance(self.structure, Crystal)

    @property
    def title(self):
        """What the file calls this structure.

        Preference order: the name the structure carries, then the block that
        names it in the file, then the file name. A multi-block CIF should
        show the block, since the file name is the same for every structure
        in it.
        """
        name = getattr(self.structure, "titl", None) or getattr(
            self.structure, "name", None
        )
        if not name or name in GENERIC_NAMES:
            name = self.block or Path(self.source).stem
        return str(name)


def supported() -> str:
    """A human-readable summary of what can be loaded."""
    groups = [
        ("periodic", {**CRYSTAL_SUFFIXES, **CRYSTAL_NAMES}),
        ("molecular", MOLECULE_SUFFIXES),
        ("either", {**EITHER_SUFFIXES, **EITHER_NAMES}),
        ("multi-frame", TRAJECTORY_SUFFIXES),
    ]
    lines = []
    for name, table in groups:
        entries = sorted({f"{k} ({v})" for k, v in table.items()})
        lines.append(f"  {name:12s} {', '.join(entries)}")
    lines.append("  anything else is passed to ASE, if it is installed")
    return "\n".join(lines)


class UnknownFormat(ValueError):
    """Raised when nothing knows how to read a file."""

    def __init__(self, path, reason=""):
        detail = f" ({reason})" if reason else ""
        super().__init__(
            f"don't know how to read {Path(path).name}{detail}\n"
            f"supported formats:\n{supported()}"
        )


def _frame_label(index, comment=""):
    """What distinguishes one frame of a multi-frame file from the next."""
    comment = comment.strip()
    # a trajectory comment is usually the step or the energy, which beats a
    # bare index when there is one
    return comment[:48] if comment else f"#{index + 1}"


def _from_trajectory(path):
    """Every frame of an XYZ file; a plain XYZ is just the one-frame case.

    chmpy's trajectory parser walks the file counting atoms and trips over a
    trailing blank line, which most XYZ writers leave behind, so the text is
    trimmed first and a single-frame read stands as the fallback.
    """
    from chmpy.fmt.xyz_file import parse_traj_string

    text = Path(path).read_text().strip()
    try:
        parsed = parse_traj_string(text)
    except (ValueError, IndexError):
        return [Frame(Molecule.load(str(path)), Path(path).name)]

    name = Path(path).name
    single = len(parsed) == 1
    return [
        Frame(
            Molecule(elements, positions),
            name,
            "" if single else _frame_label(i, comment),
        )
        for i, (elements, comment, positions) in enumerate(parsed)
    ]


def _from_crystal_file(path, **kwargs):
    """Crystal.load, unpacking the dict it returns for a multi-block CIF."""
    loaded = Crystal.load(str(path), **kwargs)
    if isinstance(loaded, dict):
        return [
            Frame(c, Path(path).name, block=name) for name, c in loaded.items()
        ]
    return [Frame(loaded, Path(path).name)]


def _from_aims(path):
    """FHI-aims geometry: periodic exactly when it declares lattice vectors.

    Decided from the contents rather than the file name, because the same
    format arrives as geometry.in, geometry.in.next_step, and whatever a run
    directory happened to call it.
    """
    text = Path(path).read_text()
    periodic = any(
        line.strip().startswith("lattice_vector") for line in text.splitlines()
    )
    loader = Crystal.from_aims_string if periodic else Molecule.from_aims_string
    return [Frame(loader(text), Path(path).name)]


def _from_ase(path):
    """Fall back to ASE, which reads far more formats than chmpy does."""
    try:
        read = require("ase.io", "reading an unrecognised format").read
    except ImportError as exc:
        raise UnknownFormat(path, str(exc).split("\n")[0]) from exc
    try:
        images = read(str(path), index=":")
    except Exception as exc:
        raise UnknownFormat(path, f"ASE could not read it: {exc}") from exc

    frames = []
    for i, atoms in enumerate(images):
        pbc = getattr(atoms, "pbc", None)
        periodic = pbc is not None and bool(pbc.any())
        structure = (
            Crystal.from_ase_atoms(atoms)
            if periodic
            else Molecule.from_ase_atoms(atoms)
        )
        frames.append(
            Frame(structure, Path(path).name, "" if len(images) == 1 else f"#{i + 1}")
        )
    return frames


def load_file(path) -> list[Frame]:
    """Every structure in one file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    suffix, name = path.suffix.lower(), path.name

    # the other multi-frame formats go to ASE, which already reads them as one
    if suffix == ".xyz":
        return _from_trajectory(path)
    if name in CRYSTAL_NAMES or suffix in CRYSTAL_SUFFIXES:
        return _from_crystal_file(path)
    if name in EITHER_NAMES or suffix in EITHER_SUFFIXES:
        return _from_aims(path)
    if suffix in MOLECULE_SUFFIXES:
        return [Frame(Molecule.load(str(path)), path.name)]
    return _from_ase(path)


def sniff(text):
    """Guess a format from file contents, for input arriving without a name."""
    # aims declares its atoms with a keyword, so it is recognisable whether or
    # not the geometry happens to be periodic
    if any(
        line.lstrip().startswith(("lattice_vector", "atom ", "atom_frac"))
        for line in text.splitlines()
    ):
        return "geometry.in"
    if "data_" in text or "_cell_length_a" in text or "loop_" in text:
        return "structure.cif"
    first = text.lstrip().split("\n", 1)[0].strip()
    if first.isdigit():
        return "structure.xyz"
    raise UnknownFormat("<stdin>", "cannot tell what format this is")


def load_stdin(text) -> list[Frame]:
    """Read an ensemble from piped text, guessing the format from content."""
    import tempfile

    name = sniff(text)
    # every reader here takes a path, and a temporary file is a great deal
    # simpler than threading an optional string through all of them
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(text)
        frames = load_file(path)
    for frame in frames:
        frame.source = "<stdin>"
    return frames


def load_frames(sources) -> list[Frame]:
    """Every structure across several files, viewed as one ensemble.

    A directory contributes every file in it that can be read, sorted by name,
    which is the usual shape of an optimisation or a set of candidates.
    """
    if isinstance(sources, str | Path):
        sources = [sources]

    paths = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.is_file()))
        else:
            paths.append(path)

    frames, failures = [], []
    for path in paths:
        try:
            frames.extend(load_file(path))
        except (UnknownFormat, FileNotFoundError, ValueError) as exc:
            failures.append((path, exc))

    if not frames:
        if len(failures) == 1:
            raise failures[0][1]
        detail = "\n".join(f"  {p.name}: {e}".split("\n")[0] for p, e in failures)
        raise UnknownFormat(paths[0] if paths else "?", f"nothing readable\n{detail}")
    return frames
