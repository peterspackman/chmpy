import logging
from pathlib import Path
import re

LOG = logging.getLogger(__name__)
NUM_ERR_REGEX = re.compile(r"([-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?)(\(\d+\))?")
QUOTE_REGEX = r"{0}\s*([^{0}]*)\s*{0}"
VALUES_REGEX = re.compile(r"""('.*?'|".*?"|;.*?;|\S+)""")


def parse_value(string, with_uncertainty=False):
    """parse a value from a cif file to its appropriate type
    e.g. int, float, str etc. Will handle uncertainty values
    contained in parentheses.

    Parameters
    ----------
    string: str
        the string containing the value to parse

    with_uncertainty: bool, optional
        return a tuple including uncertainty if a numeric type is expected

    Returns
    -------
    value
        the value coerced into the appropriate type

    >>> parse_value("2.3(1)", with_uncertainty=True)
    (2.3, 1)
    >>> parse_value("string help")
    'string help'
    >>> parse_value("3.1415") * 4
    12.566
    """
    match = NUM_ERR_REGEX.match(string)
    if match and match.span()[1] == len(string):
        groups = match.groups()
        number, uncertainty = groups[0], groups[-1]
        number = float(number)
        if number.is_integer():
            number = int(number)
        if with_uncertainty:
            return number, int(uncertainty.strip("()")) if uncertainty else 0
        return number
    else:
        s = string.strip()
        if s[0] == s[-1] and s[0] in ("'", ";", '"'):
            return parse_quote(string, delimiter=s[0])
    return string


def parse_quote(string, delimiter=";"):
    """extract a value contained within quotes, with an optional change
    of delimiter

    Parameters
    ----------
    string: str
        the string containing the value to parse

    delimiter: bool, optional
        the quote delimiter, default ';'

    Returns
    -------
    value
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

    regex = QUOTE_REGEX.format(delimiter)
    match = re.match(regex, string)
    if match:
        return match.groups()[0]
    return string


def needs_quote(string):
    "check if a string needs to be quoted (i.e. it has a space or quotation marks in it"
    if not isinstance(string, str):
        return False
    return " " in string and (not ('"' in string or "'" in string))


def is_scalar(value):
    "check if the value is a string or has no __len__ dunder method"
    return isinstance(value, str) or (not hasattr(value, "__len__"))


def format_field(x):
    "format a field to fixed precision if float otherwise as a string"
    if isinstance(x, float):
        return f"{x:20.12f}"
    elif isinstance(x, int):
        return f"{x:20d}"
    elif isinstance(x, str):
        if needs_quote(x):
            return f"'{x}'"
        else:
            return x
    else:
        return str(x)


class Cif:
    """Class to represent data extracted from a CIF
    standard file format.

    Parameters
    ----------
    cif_data : dict
        dictionary of CIF keys and values
    """

    def __init__(self, cif_data):
        self.data = cif_data
        self.line_dispatch = {
            "#": self.parse_comment_line,
            "loop_": self.parse_loop_block,
        }
        self.current_data_block_name = "unknown"
        self.content_lines = []

    def is_comment_line(self, line):
        "check if the line is a comment i.e. starts with '#'"
        return line.strip().startswith("#")

    def is_data_name_line(self, line):
        "check if the line is a data_name i.e. starts with a single '_'"
        return line.strip().startswith("_")

    def is_empty_line(self, line):
        "check if the line is empty/blank"
        if line and line.strip():
            return False
        return True

    def is_data_line(self, line):
        "check if the line contains a value for the currentt key"
        if self.is_empty_line(line):
            return False
        if self.is_comment_line(line):
            return False
        if self.is_data_name_line(line):
            return False
        if line.split()[0] in self.line_dispatch:
            return False
        return True

    def parse_quoted_block(self, delimiter=";"):
        "parse an entire quoted block, delimited by delimiter"
        LOG.debug("Parsing quoted block at line %d", self.line_index)
        l1 = self.content_lines[self.line_index].strip()
        i = self.line_index + 1
        j = i
        n = 1
        while ";" not in self.content_lines[j]:
            j += 1
            n += 1
            if j >= len(self.content_lines) - 1:
                break
        else:
            section = " ".join(x.strip() for x in self.content_lines[i - 1 : j + 1])
            self.line_index += n
            return parse_quote(section)
        raise ValueError(f"Unmatch quotation on line {self.line_index + 1}")

    def parse_data_name(self):
        "parse a single data name i.e key for the cif_data dictionary"
        tokens = self.content_lines[self.line_index].strip()[1:].split()
        k = tokens[0]
        v = None
        if len(tokens) == 1:
            next_line = self.content_lines[self.line_index + 1]
            while next_line.strip().startswith("#"):
                self.line_index += 1
                next_line = self.content_lines[self.line_index + 1]

            if ";" in next_line:
                self.line_index += 1
                v = self.parse_quoted_block().strip()
            else:
                v = parse_value(next_line)
                self.line_index += 1
        else:
            v = " ".join(tokens[1:])
        if v is None:
            raise ValueError(
                f"Error parsing CIF data_name on line {self.line_index}, context = {k}"
            )
        self.current_data_block[k] = parse_value(v)
        self.line_index += 1
        LOG.debug("Parsed data name: %s = %s", k, v)

    def parse_loop_block(self):
        "parse values contained in a _loop block"
        LOG.debug("Parsing loop block")
        self.line_index += 1
        line = self.content_lines[self.line_index]
        keys = []
        while line.strip().startswith("_"):
            keys.append(line.strip()[1:])
            self.line_index += 1
            line = self.content_lines[self.line_index]

        line = self.content_lines[self.line_index]
        values = []
        while self.is_data_line(line):
            LOG.debug("Parsing data line: %s", line)
            values.append(line.strip())
            self.line_index += 1
            if self.line_index >= len(self.content_lines):
                LOG.debug("Reached end of file parsing loop block")
                break
            line = self.content_lines[self.line_index]
        for k in keys:
            self.current_data_block[k] = []

        for value in values:
            vs = re.findall(VALUES_REGEX, value.strip())
            for k, v in zip(keys, vs):
                self.current_data_block[k].append(parse_value(v))
        LOG.debug("Parsed loop block")

    def parse_comment_line(self):
        "ignore comment lines"
        self.line_index += 1

    def parse_data_block_name(self):
        "parse a data block name"
        LOG.debug("Parsing data block name at line %d", self.line_index)
        line = self.content_lines[self.line_index]
        self.current_data_block_name = line[5:].strip()
        self.line_index += 1
        LOG.debug("Parsed data block name: %s", self.current_data_block_name)

    def parse(self, ignore_uncertainty=True):
        "parse the entire CIF contents"
        if not ignore_uncertainty:
            raise NotImplementedError(
                "Storing uncertainty information has not been implemented"
            )
        self.line_index = 0
        line_count = len(self.content_lines)
        while self.line_index < line_count:
            line = self.content_lines[self.line_index].strip()
            if line:
                token = line.split()[0]
                if token in self.line_dispatch:
                    self.line_dispatch[token]()
                elif token.startswith("_"):
                    self.parse_data_name()
                elif token.startswith("data_"):
                    self.parse_data_block_name()
                else:
                    LOG.debug("Skipping unknown line: %s", line)
                    self.line_index += 1
            else:
                self.line_index += 1
        self.line_index = 0
        return self.data

    @property
    def current_data_block(self):
        "return the current data block, adding the key if necessary"
        if self.current_data_block_name not in self.data:
            self.data[self.current_data_block_name] = {}
        return self.data[self.current_data_block_name]

    @classmethod
    def from_file(cls, filename):
        "initialize a :obj:`Cif` from a file path"
        return cls.from_string(Path(filename).read_text())

    @classmethod
    def from_string(cls, contents):
        "initialize a :obj:`Cif` from string contents"
        c = cls({})
        c.content_lines = contents.split("\n")
        c.parse()
        return c

    def to_string(self):
        "represent the data in this :obj`Cif` textually in the CIF format"
        lines = []
        for data_block_name, data_block_data in self.data.items():
            lines.append(f"data_{data_block_name}")
            vector_data_names = []
            for data_name, data_value in data_block_data.items():
                if is_scalar(data_value):
                    quote = ""
                    if needs_quote(data_value):
                        quote = "'"
                    lines.append(f"_{data_name} {quote}{data_value}{quote}")
                else:
                    vector_data_names.append(data_name)
            from itertools import groupby

            for name_section, names in groupby(
                vector_data_names, key=lambda x: x.split("_")[0]
            ):
                for section, names in groupby(
                    names, key=lambda x: len(data_block_data[x])
                ):
                    lines.append("loop_")
                    loop_values = []
                    for name in names:
                        lines.append(f"_{name}")
                        loop_values.append(data_block_data[name])
                    for row in zip(*loop_values):
                        lines.append(" ".join(format_field(x) for x in row))

        lines.append("#END")
        return "\n".join(lines)

    def to_file(self, filename):
        "write this :obj:`Cif` to file"
        Path(filename).write_text(self.to_string())
