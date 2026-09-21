"""Read and write CIF (Crystallographic Information File) data.

The parser is pure Python and imports nothing outside the standard library,
not even :mod:`re`.  It is built around one observation about real CIF files:
almost every token in them is a bare, whitespace-delimited word, and only a
small minority are quoted strings, comments, text fields or data names.  So
rather than matching a pattern against every token, it runs one
:meth:`str.find` cursor per interesting character (``'``, ``"``, ``#``, ``;``
and ``_``) and lets :meth:`str.split` -- a single C-level call -- handle
everything in between.  Those cursors also record where the data names are,
which is what lets the parser step from one to the next without looking at
the values.

The result behaves like a dictionary of dictionaries::

    >>> cif = Cif.from_string('''
    ... data_example
    ... _cell_length_a   5.1234(3)
    ... loop_
    ... _atom_site_label
    ... _atom_site_fract_x
    ... C1 0.1234
    ... O1 0.5678
    ... ''')
    >>> cif["example"]["cell_length_a"]
    5.1234
    >>> cif["example"]["atom_site_label"]
    ['C1', 'O1']

Data names are matched case-insensitively, with or without the leading
underscore, and across the naming conventions of different CIF flavours, so
an mmCIF and a small-molecule CIF can be read through the same keys::

    >>> mm = Cif.from_string("data_x\\n_cell.length_a  5.1234\\n")
    >>> mm["x"]["cell_length_a"]
    5.1234
"""

import logging
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

LOG = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_OPTIONS",
    "Cif",
    "CifBlock",
    "CifOptions",
    "CifParseError",
    "canonical_data_name",
    "format_field",
    "is_scalar",
    "needs_quote",
    "parse_quote",
    "parse_value",
    "register_aliases",
]


class CifParseError(ValueError):
    """Raised when the contents of a CIF cannot be interpreted."""


@dataclass(frozen=True)
class CifOptions:
    """Settings for reading a CIF, passed to :meth:`Cif.from_file` and
    :meth:`Cif.from_string`.

    Both default to the CIF 1.1 reading that suits nearly every file in
    circulation; each is here because a reasonable person might want the
    other behaviour.

    Args:
        join_text_field_lines: join the lines of a multi-line text field into
            one line separated by single spaces, instead of keeping the text
            as it was written.  Off by default, because the line structure is
            part of the value -- but it is what chmpy did before, and it is
            convenient when a text field only ever holds a wrapped sentence.
        cif2_triple_quotes:
            read ``'''...'''`` and ``\"\"\"...\"\"\"`` as CIF 2.0
            triple-quoted strings, which may span lines and contain the
            delimiter singly or doubly.  Off by default because the same
            characters mean something else under CIF 1.1, where ``'''a'''``
            is the string ``''a''``.  Single and double quotes keep their
            CIF 1.1 meaning either way, so an apostrophe inside a quoted
            string still needs no escaping.

    >>> text = "data_a\\n_x\\n;\\nfirst line\\nsecond line\\n;\\n"
    >>> Cif.from_string(text)["a"]["x"]
    'first line\\nsecond line'
    >>> Cif.from_string(text, options=CifOptions(join_text_field_lines=True))["a"]["x"]
    'first line second line'
    """

    join_text_field_lines: bool = False
    cif2_triple_quotes: bool = False


#: how a CIF is read when no :class:`CifOptions` are given
DEFAULT_OPTIONS = CifOptions()


# ---------------------------------------------------------------------------
# data name aliases
# ---------------------------------------------------------------------------

#: Data names that mean the same thing in different CIF flavours.  Each tuple
#: is one equivalence class; lookups try every member of the class before
#: giving up.  Names are written without the leading underscore and are
#: matched case-insensitively.  mmCIF (PDBx) spells its categories with a dot
#: where the core CIF dictionary uses an underscore.
CIF_ALIAS_GROUPS = (
    # unit cell
    ("cell_length_a", "cell.length_a"),
    ("cell_length_b", "cell.length_b"),
    ("cell_length_c", "cell.length_c"),
    ("cell_angle_alpha", "cell.angle_alpha"),
    ("cell_angle_beta", "cell.angle_beta"),
    ("cell_angle_gamma", "cell.angle_gamma"),
    ("cell_volume", "cell.volume"),
    ("cell_formula_units_z", "cell.formula_units_z", "cell.z_pdb"),
    # space group / symmetry
    (
        "symmetry_space_group_name_h-m",
        "space_group_name_h-m_alt",
        "space_group_name_h-m_full",
        "symmetry.space_group_name_h-m",
        "space_group.name_h-m_alt",
        "space_group.name_h-m_full",
    ),
    (
        "symmetry_space_group_name_hall",
        "space_group_name_hall",
        "symmetry.space_group_name_hall",
        "space_group.name_hall",
    ),
    (
        "symmetry_int_tables_number",
        "space_group_it_number",
        "symmetry.int_tables_number",
        "space_group.it_number",
    ),
    (
        "symmetry_equiv_pos_as_xyz",
        "space_group_symop_operation_xyz",
        "symmetry_equiv.pos_as_xyz",
        "space_group_symop.operation_xyz",
    ),
    (
        "symmetry_equiv_pos_site_id",
        "space_group_symop_id",
        "symmetry_equiv.id",
        "space_group_symop.id",
    ),
    (
        "symmetry_cell_setting",
        "space_group_crystal_system",
        "symmetry.cell_setting",
        "space_group.crystal_system",
    ),
    # atom sites
    ("atom_site_label", "atom_site.label", "atom_site.label_atom_id"),
    ("atom_site_type_symbol", "atom_site.type_symbol"),
    ("atom_site_fract_x", "atom_site.fract_x"),
    ("atom_site_fract_y", "atom_site.fract_y"),
    ("atom_site_fract_z", "atom_site.fract_z"),
    ("atom_site_cartn_x", "atom_site.cartn_x"),
    ("atom_site_cartn_y", "atom_site.cartn_y"),
    ("atom_site_cartn_z", "atom_site.cartn_z"),
    ("atom_site_occupancy", "atom_site.occupancy"),
    ("atom_site_u_iso_or_equiv", "atom_site.u_iso_or_equiv"),
    ("atom_site_b_iso_or_equiv", "atom_site.b_iso_or_equiv"),
    ("atom_site_adp_type", "atom_site.adp_type", "atom_site.thermal_displace_type"),
    ("atom_site_symmetry_multiplicity", "atom_site.symmetry_multiplicity"),
    ("atom_site_disorder_group", "atom_site.disorder_group"),
    ("atom_site_disorder_assembly", "atom_site.disorder_assembly"),
    # anisotropic displacement parameters
    ("atom_site_aniso_label", "atom_site_anisotrop.id"),
    ("atom_site_aniso_u_11", "atom_site_anisotrop.u[1][1]"),
    ("atom_site_aniso_u_22", "atom_site_anisotrop.u[2][2]"),
    ("atom_site_aniso_u_33", "atom_site_anisotrop.u[3][3]"),
    ("atom_site_aniso_u_12", "atom_site_anisotrop.u[1][2]"),
    ("atom_site_aniso_u_13", "atom_site_anisotrop.u[1][3]"),
    ("atom_site_aniso_u_23", "atom_site_anisotrop.u[2][3]"),
    # chemistry and provenance
    ("chemical_formula_sum", "chemical_formula.sum", "chem_comp.formula"),
    ("chemical_formula_moiety", "chemical_formula.moiety"),
    ("chemical_name_systematic", "chemical.name_systematic"),
    ("chemical_name_common", "chemical.name_common", "chem_comp.name"),
    ("cell_measurement_temperature", "diffrn.ambient_temp"),
    ("diffrn_radiation_wavelength", "diffrn_radiation_wavelength.wavelength"),
)

#: lowercase data name -> the other names in its equivalence class
CIF_ALIASES = {}


def register_aliases(*names):
    """Declare that ``names`` are different spellings of the same data name.

    Useful for programs that write CIFs with their own data names.  Existing
    equivalence classes containing any of ``names`` are merged into one.

    >>> register_aliases("cell_length_a", "my_program_cell_a")
    >>> cif = Cif.from_string("data_x\\n_my_program_cell_a 3.0\\n")
    >>> cif["x"]["cell_length_a"]
    3.0
    """
    group = set()
    for name in names:
        low = name.lower().lstrip("_")
        group.add(low)
        group.update(CIF_ALIASES.get(low, ()))
    for name in group:
        CIF_ALIASES[name] = tuple(group - {name})


for _group in CIF_ALIAS_GROUPS:
    register_aliases(*_group)
del _group


def canonical_data_name(name):
    """The first spelling of ``name``'s equivalence class, or ``name`` itself.

    >>> canonical_data_name("_cell.length_a")
    'cell_length_a'
    >>> canonical_data_name("not_a_known_name")
    'not_a_known_name'
    """
    low = name.lower().lstrip("_")
    if low in CIF_ALIASES:
        for group in CIF_ALIAS_GROUPS:
            if low in group:
                return group[0]
    return low


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

#: characters that begin a token str.split alone cannot handle, plus the
#: underscore, which is what every data name and reserved word is found by
_SPECIAL = "'\"#;_"
_SPECIAL_INDEX = {c: i for i, c in enumerate(_SPECIAL)}
_WHITESPACE = " \t\n\r\f\v"
_WS = frozenset(_WHITESPACE)
#: reserved words, by the text preceding their trailing underscore
_RESERVED_WORDS = frozenset(("loop", "data", "save", "stop", "global"))
_RESERVED_TAIL = frozenset("paelPAEL")


def _line_of(text, index):
    return text.count("\n", 0, index) + 1


def _reserved_start(text, i):
    """Where the reserved word whose trailing ``_`` is at ``i`` begins, or -1.

    ``loop_``, ``data_``, ``save_``, ``stop_`` and ``global_`` all carry an
    underscore at a fixed offset from their start, so the scan for data names
    finds them too and this only has to confirm what it found.  The caller has
    already checked that the character before the underscore is one that ends
    a reserved word.
    """
    start = i - 6 if text[i - 1] in "lL" else i - 4
    if start < 0 or (start and text[start - 1] not in _WS):
        return -1
    return start if text[start:i].lower() in _RESERVED_WORDS else -1


def _scan(text, options=DEFAULT_OPTIONS):
    """Split CIF text into raw tokens, noting which of them carry structure.

    Returns the tokens and the indices of those that are data names or
    reserved words.  Recording them here is what lets the parser step from
    one data name to the next instead of inspecting every value in between.
    """
    join_lines = options.join_text_field_lines
    triple_quotes = options.cif2_triple_quotes
    tokens = []
    marks = []
    add = tokens.append
    extend = tokens.extend
    mark = marks.append
    n = len(text)
    pos = 0

    # One find cursor per special character, merged by hand.  str.find drops
    # to memchr for a single character while a regular expression character
    # class does not, which makes this an order of magnitude faster than one
    # pattern matching all five.
    cursors = [n if j < 0 else j for j in (text.find(c) for c in _SPECIAL)]
    while True:
        i = min(cursors)
        if i >= n:
            break
        char = text[i]
        cursor = _SPECIAL_INDEX[char]

        if i >= pos:  # otherwise it sits inside text already consumed
            if char == "_":
                previous = text[i - 1] if i else "\n"
                if previous in _WS:
                    start = i
                elif previous in _RESERVED_TAIL:
                    start = _reserved_start(text, i)
                else:
                    start = -1  # just an underscore inside a value or a name
                if start >= pos:
                    if start > pos:
                        extend(text[pos:start].split())
                    mark(len(tokens))
                    # leave pos here: the word itself comes out of the next
                    # split, so its own underscores need no special handling
                    pos = start

            elif i == 0 or text[i - 1] in _WS:  # else e.g. the quote in 5'-AMP
                if char == "#":
                    if i > pos:
                        extend(text[pos:i].split())
                    line_end = text.find("\n", i)
                    pos = n if line_end < 0 else line_end + 1
                elif char == ";":
                    if i == 0 or text[i - 1] == "\n":  # else a lone semicolon
                        field_end = text.find("\n;", i)
                        if field_end < 0:
                            raise CifParseError(
                                "unterminated text field starting on line "
                                f"{_line_of(text, i)}"
                            )
                        if i > pos:
                            extend(text[pos:i].split())
                        field = text[i : field_end + 2]
                        add(_join_lines(field) if join_lines else field)
                        pos = field_end + 2
                elif triple_quotes and text[i + 1 : i + 3] == char + char:
                    quote_end = text.find(char * 3, i + 3)
                    if quote_end < 0:
                        raise CifParseError(
                            "unterminated triple-quoted string on line "
                            f"{_line_of(text, i)}"
                        )
                    if i > pos:
                        extend(text[pos:i].split())
                    # a CIF 2.0 triple-quoted string holds the same kind of
                    # value as a text field -- many lines, taken verbatim --
                    # so it is handed on as one
                    add(f";{text[i + 3 : quote_end]}\n;")
                    pos = quote_end + 3
                else:
                    quote_end = _closing_quote(text, i, char, n)
                    if i > pos:
                        extend(text[pos:i].split())
                    add(text[i : quote_end + 1])
                    pos = quote_end + 1

        # resume past whatever was just consumed, so a run of comment hashes
        # or a text field full of underscores costs one find, not hundreds
        found = text.find(char, pos if pos > i else i + 1)
        cursors[cursor] = n if found < 0 else found

    if pos < n:
        extend(text[pos:].split())
    return tokens, marks


def _join_lines(field):
    """A text field token with its line structure flattened to single spaces."""
    lines = (line.strip() for line in field[1:-2].split("\n"))
    return ";" + " ".join(line for line in lines if line) + "\n;"


def tokenize(text, options=DEFAULT_OPTIONS):
    """Split CIF text into raw tokens.

    Quoted values keep their delimiters and text fields keep their leading and
    trailing semicolons, so that a token always says what kind of value it is.
    Comments are dropped.

    >>> tokenize("_a 1 _b 'two words' # comment\\n")
    ['_a', '1', '_b', "'two words'"]
    >>> tokenize("_x\\n;\\nline one\\nline two\\n;\\n")
    ['_x', ';\\nline one\\nline two\\n;']
    """
    return _scan(text, options)[0]


def _closing_quote(text, start, char, n):
    """Index of the quote closing the string opened at ``start``.

    A quote only closes a string when whitespace or the end of the file
    follows it, so apostrophes inside a quoted string need no escaping.
    """
    end = text.find(char, start + 1)
    while 0 <= end < n - 1 and text[end + 1] not in _WS:
        end = text.find(char, end + 1)
    line_end = text.find("\n", start)
    if end < 0 or (0 <= line_end < end):
        raise CifParseError(
            f"unterminated quoted string on line {_line_of(text, start)}"
        )
    return end


# ---------------------------------------------------------------------------
# value conversion
# ---------------------------------------------------------------------------

_NUM_START = frozenset("0123456789+-.")
#: last characters a token can have and still be a number
_NUM_END = frozenset("0123456789.")
#: first characters of a token that _convert might have to change
_NEEDS_WORK = frozenset("0123456789+-.'\";")


def _convert(token):
    """Convert one raw token into a Python value."""
    char = token[0]
    # int() and float() read "_" as a digit separator, which would turn the
    # symmetry code 1_555 into the number 1555, so numbers never contain one
    if char in _NUM_START and "_" not in token:
        text = token
        if text[-1] == ")":  # a standard uncertainty, e.g. 5.1234(3)
            text = text.partition("(")[0]
        # only try to parse what could be a number: raising and catching a
        # ValueError costs more than parsing a number does, and "." -- the
        # commonest value in a large CIF -- would raise one every time
        if text[-1:] in _NUM_END and text != ".":
            try:
                if "." in text or "e" in text or "E" in text:
                    return float(text)
                return int(text)
            except ValueError:
                pass
        return token
    if char == "'" or char == '"':
        return token[1:-1]
    if char == ";" and token[-1] == ";" and len(token) > 1:
        body = token[1:-2]
        return body[1:] if body[:1] == "\n" else body
    return token


def _convert_column(column):
    """Convert a whole loop column, taking the whole column at once if it is
    uniformly numeric -- which is where nearly all of a large CIF's tokens are.
    """
    if not column:
        return column
    first = column[0]
    if (
        first[0] in _NUM_START
        and "_" not in first
        and (first[-1] in _NUM_END or first[-1] == ")")
        and first != "."
    ):
        try:
            if first[-1] == ")":
                return [float(v.partition("(")[0]) for v in column]
            if "." in first or "e" in first or "E" in first:
                return list(map(float, column))
            values = list(map(int, column))
        except ValueError:
            pass
        else:
            # see _convert: only trust int() over a column free of "_"
            if "_" not in "".join(column):
                return values
    # anything else is a column of labels, symbols or flags, where the common
    # case is a value that needs no work at all -- so check for that inline
    return [v if v[0] not in _NEEDS_WORK else _convert(v) for v in column]


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------

_MISSING = object()


class CifBlock(dict):
    """One ``data_`` block: a dict of data names to values.

    Keys are stored as they were written in the file, minus the leading
    underscore.  An exact key lookup is a plain dict lookup; anything else
    falls back to matching without regard to case, the leading underscore, or
    which CIF dictionary the name came from (see :data:`CIF_ALIAS_GROUPS`).

    >>> block = CifBlock({"Cell_Length_A": 5.0}, name="x")
    >>> block["Cell_Length_A"], block["_cell_length_a"], block["cell.length_a"]
    (5.0, 5.0, 5.0)
    >>> "cell_volume" in block
    False
    """

    def __init__(self, *args, name="", **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        #: the data names of each ``loop_``, in the order they were read
        self.loops = []
        #: nested ``save_`` frames, by name (only used by CIF dictionaries)
        self.save_frames = {}
        self._index = None

    def _resolve(self, key):
        """The stored key matching ``key``, or None."""
        index = self._index
        if index is None or len(index) != len(self):
            index = self._index = {k.lower(): k for k in self}
        name = key[1:] if key[:1] == "_" else key
        name = name.lower()
        found = index.get(name)
        if found is not None:
            return found
        for alias in CIF_ALIASES.get(name, ()):
            found = index.get(alias)
            if found is not None:
                return found
        return None

    def __missing__(self, key):
        found = self._resolve(key)
        if found is None:
            raise KeyError(key)
        return dict.__getitem__(self, found)

    def __contains__(self, key):
        return dict.__contains__(self, key) or self._resolve(key) is not None

    def get(self, key, default=None):
        value = dict.get(self, key, _MISSING)
        if value is not _MISSING:
            return value
        found = self._resolve(key)
        return default if found is None else dict.__getitem__(self, found)

    def is_loop(self, key):
        "whether ``key`` was read as a column of a ``loop_``"
        found = self._resolve(key)
        return any(found in loop for loop in self.loops)

    def __repr__(self):
        return f"<CifBlock {self.name!r}: {len(self)} data names>"


class Cif:
    """The contents of a CIF: an ordered mapping of block name to
    :class:`CifBlock`.

    Args:
        cif_data: existing ``{block name: {data name: value}}`` contents

    >>> cif = Cif.from_string("data_one\\n_a 1\\ndata_two\\n_a 2\\n")
    >>> list(cif)
    ['one', 'two']
    >>> cif["two"]["a"]
    2
    """

    def __init__(self, cif_data=None):
        self.data = {} if cif_data is None else dict(cif_data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, key):
        return key in self.data

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def get(self, key, default=None):
        return self.data.get(key, default)

    @property
    def first_block(self):
        "the first data block in the file"
        return next(iter(self.data.values()))

    def __repr__(self):
        return f"<Cif: {', '.join(self.data)}>"

    @classmethod
    def from_file(cls, filename, options=None):
        """initialize a :obj:`Cif` from a file path

        ``options`` is a :class:`CifOptions`; the default reading is CIF 1.1.
        """
        return cls.from_string(Path(filename).read_text(), options=options)

    @classmethod
    def from_string(cls, contents, options=None):
        """initialize a :obj:`Cif` from string contents

        ``options`` is a :class:`CifOptions`; the default reading is CIF 1.1.
        """
        cif = cls()
        cif.data = _parse(contents, options or DEFAULT_OPTIONS)
        return cif

    def to_string(self):
        "represent the data in this :obj:`Cif` textually in the CIF format"
        lines = []
        for block_name, block in self.data.items():
            lines.append(f"data_{_data_block_name(block_name)}")
            _write_block(block, lines)
        lines.append("#END")
        return "\n".join(lines)

    def to_file(self, filename):
        "write this :obj:`Cif` to file"
        Path(filename).write_text(self.to_string())


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _parse(text, options=DEFAULT_OPTIONS):
    """Parse CIF text into ``{block name: CifBlock}``."""
    tokens, marks = _scan(text, options)
    n_tokens = len(tokens)
    n_marks = len(marks)
    blocks = {}
    block = None
    frame_stack = []
    k = 0
    while k < n_marks:
        index = marks[k]
        token = tokens[index]

        if token[0] == "_":
            if block is None:
                block = blocks.setdefault("unknown", CifBlock(name="unknown"))
            value = index + 1
            if value >= n_tokens or (k + 1 < n_marks and marks[k + 1] == value):
                raise CifParseError(f"data name {token} has no value")
            block[token[1:]] = _convert(tokens[value])
            k += 1
            continue

        low = token.lower()
        if low == "loop_":
            if block is None:
                block = blocks.setdefault("unknown", CifBlock(name="unknown"))
            k = _parse_loop(tokens, marks, k, n_tokens, n_marks, block)
        elif low[:5] == "data_":
            name = token[5:]
            block = blocks.setdefault(name, CifBlock(name=name))
            frame_stack.clear()
            k += 1
        elif low[:5] == "save_":
            if block is None:
                block = blocks.setdefault("unknown", CifBlock(name="unknown"))
            if len(token) > 5:
                frame_stack.append(block)
                name = token[5:]
                block = block.save_frames.setdefault(name, CifBlock(name=name))
            elif frame_stack:
                block = frame_stack.pop()
            k += 1
        else:  # global_ or stop_
            k += 1
    return blocks


def _parse_loop(tokens, marks, k, n_tokens, n_marks, block):
    """Read the ``loop_`` whose keyword is marks[k]; return the next mark."""
    names = []
    expected = marks[k] + 1
    k += 1
    while k < n_marks and marks[k] == expected and tokens[expected][0] == "_":
        names.append(tokens[expected][1:])
        expected += 1
        k += 1
    if not names:
        raise CifParseError("loop_ with no data names")

    # no value can be a data name or a reserved word, so the next mark -- if
    # there is one -- is exactly where this loop stops
    end = marks[k] if k < n_marks else n_tokens
    values = tokens[expected:end]
    width = len(names)
    left_over = len(values) % width
    if left_over:
        LOG.warning(
            "loop over %s has %d trailing value(s), expected a multiple of %d",
            names[0],
            left_over,
            width,
        )
        values = values[: len(values) - left_over]
    for offset, name in enumerate(names):
        block[name] = _convert_column(values[offset::width])
    block.loops.append(names)
    return k


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

_QUOTE_FIRST = frozenset("_#$[]'\";")
_RESERVED_PREFIXES = ("data_", "save_", "loop_", "global_", "stop_")


def is_scalar(value):
    "check if the value is a string or has no __len__ dunder method"
    return isinstance(value, str) or (not hasattr(value, "__len__"))


def needs_quote(string):
    """check whether a value has to be quoted to survive a round trip

    >>> needs_quote("this string will need quoting")
    True
    >>> needs_quote("thiswon't")
    False
    >>> needs_quote(3.45)
    False
    """
    if not isinstance(string, str):
        return False
    if not string or string in (".", "?"):
        return True
    if string[0] in _QUOTE_FIRST:
        return True
    if any(ws in string for ws in _WHITESPACE):
        return True
    return string.lower().startswith(_RESERVED_PREFIXES)


def quote(string):
    """quote a value so that it reads back as the same string

    Text fields are used for anything a one-line quote cannot hold.

    >>> quote("P 21/c")
    "'P 21/c'"
    >>> quote("it's")
    '"it\\'s"'
    """
    if "\n" not in string and "\r" not in string:
        if "'" not in string:
            return f"'{string}'"
        if '"' not in string:
            return f'"{string}"'
    return f"\n;\n{string}\n;"


def format_field(x):
    """format a value as a CIF field

    >>> len(format_field(5.4))
    20
    >>> format_field("3.4")
    '3.4'
    """
    if isinstance(x, float):
        return f"{x:20.12f}"
    elif isinstance(x, bool):
        return "1" if x else "0"
    elif isinstance(x, int):
        return f"{x:20d}"
    elif x is None:
        return "?"
    elif isinstance(x, str):
        return quote(x) if needs_quote(x) else x
    return str(x)


def _data_block_name(name):
    """A block name that reads back as itself.

    CIF has no way to quote a block name, so whitespace in one would silently
    truncate it on the next read.

    >>> _data_block_name("my structure")
    'my_structure'
    """
    return "_".join(str(name).split()) or "unnamed"


def _loop_groups(block, names):
    """Which data names share a ``loop_``.

    Blocks that were read from a file remember their loops; ones built by
    hand are grouped by category and length, which is the best guess
    available and always produces a valid file.
    """
    remaining = list(names)
    groups = []
    for loop in getattr(block, "loops", ()):
        group = [n for n in loop if n in remaining]
        if group:
            groups.append(group)
            for name in group:
                remaining.remove(name)
    for _, by_category in groupby(remaining, key=lambda x: x.split("_")[0]):
        for _, by_length in groupby(by_category, key=lambda x: len(block[x])):
            groups.append(list(by_length))
    return groups


def _write_block(block, lines):
    loop_names = []
    for name, value in block.items():
        if is_scalar(value):
            field = format_field(value)
            sep = "" if field.startswith("\n") else " "
            lines.append(f"_{name}{sep}{field}")
        else:
            loop_names.append(name)

    for group in _loop_groups(block, loop_names):
        columns = [block[name] for name in group]
        lengths = {len(column) for column in columns}
        if len(lengths) > 1:
            raise ValueError(
                f"cannot write a loop over {', '.join(group)}: its columns "
                f"have differing lengths {sorted(lengths)}"
            )
        lines.append("loop_")
        for name in group:
            lines.append(f"_{name}")
        for row in zip(*columns, strict=True):
            fields = [format_field(x) for x in row]
            if any(f.startswith("\n") for f in fields):
                lines.append(
                    "\n".join(
                        f.lstrip("\n") if f.startswith("\n") else f for f in fields
                    )
                )
            else:
                lines.append(" ".join(fields))


# ---------------------------------------------------------------------------
# single value helpers, kept for callers that work with one field at a time
# ---------------------------------------------------------------------------


def parse_value(string, with_uncertainty=False):
    """parse a single value from a cif file to its appropriate type
    e.g. int, float, str etc. Will handle uncertainty values
    contained in parentheses.

    Args:
        string: the string containing the value to parse
        with_uncertainty: return a tuple including uncertainty if a numeric
            type is expected

    Returns:
        the value coerced into the appropriate type

    >>> parse_value("2.3(1)", with_uncertainty=True)
    (2.3, 1)
    >>> parse_value("string help")
    'string help'
    >>> parse_value("3.1415") * 4
    12.566
    """
    string = string.strip()
    if not string:
        return string
    value = _convert(string)
    if not with_uncertainty:
        return value
    if isinstance(value, str) or string[-1] != ")":
        return value, 0
    return value, int(string[string.find("(") + 1 : -1])


def parse_quote(string, delimiter=";"):
    """extract a value contained within quotes, with an optional change
    of delimiter

    Args:
        string: the string containing the value to parse
        delimiter: the quote delimiter, default ';'

    Returns:
        the string contained inside the quotes

    >>> parse_quote(";quote text;")
    'quote text'
    >>> parse_quote(":'quote text':", delimiter="'")
    ":'quote text':"
    >>> parse_quote(":'quote text':", delimiter=":")
    "'quote text'"
    >>> parse_quote("'-y, x-y, z'", delimiter="'")
    '-y, x-y, z'
    """
    if len(string) > 1 and string[0] == delimiter and string[-1] == delimiter:
        return string[1:-1].strip()
    return string
