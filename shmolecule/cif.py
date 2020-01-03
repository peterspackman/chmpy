import logging
from .space_group import SymmetryOperation, SpaceGroup
from pathlib import Path
import re

LOG = logging.getLogger(__name__)
NUM_ERR_REGEX = re.compile(r"([-+]?[0-9]+[.]?[0-9]*)(\(\d+\))?")
QUOTE_REGEX = r"{0}\s+([^{0}]*)\s+{0}"


def parse_value(string, with_uncertainty=False):
    match = NUM_ERR_REGEX.match(string)
    if match and match.span()[1] == len(string):
        number, uncertainty = match.groups()
        number = float(number)
        if number.is_integer():
            number = int(number)
        if with_uncertainty:
            return number, int(uncertainty.strip("()")) if uncertainty else 0
        return number
    else:
        return string


def parse_quote(string, delimiter=";"):
    match = re.match(QUOTE_REGEX.format(delimiter), string)
    if match:
        return match.groups()[0]
    return string


def needs_quote(string):
    if not isinstance(string, str):
        return False
    return " " in string and (not ('"' in string or "'" in string))


def is_scalar(value):
    return isinstance(value, str) or (not hasattr(value, "__len__"))


def format_field(x):
    if isinstance(x, float):
        return f"{x:20.12f}"
    else:
        return str(x)


class Cif:
    def __init__(self, cif_data):
        self.data = cif_data
        self.line_dispatch = {
            "#": self.parse_comment_line,
            "loop_": self.parse_loop_block,
        }
        self.current_data_block_name = "unknown"
        self.content_lines = []

    def is_comment_line(self, line):
        return line.strip().startswith("#")

    def is_data_name_line(self, line):
        return line.strip().startswith("_")

    def is_empty_line(self, line):
        if line and line.strip():
            return False
        return True

    def is_data_line(self, line):
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
        l1 = self.content_lines[self.line_index].strip()
        i = self.line_index + 1
        j = i
        while ";" not in self.content_lines[j]:
            j += 1
            if j >= len(self.content_lines) - 1:
                break
        else:
            section = " ".join(x.strip() for x in self.content_lines[i - 1 : j + 1])
            return parse_quote(section)
        raise ValueError(f"Unmatch quotation on line {self.line_index + 1}")

    def parse_data_name(self):
        tokens = self.content_lines[self.line_index].strip()[1:].split()
        k = tokens[0]
        if len(tokens) == 1:
            next_line = self.content_lines[self.line_index + 1]
            if ";" in next_line:
                self.line_index += 1
                v = self.parse_quoted_block().strip()
        else:
            v = " ".join(tokens[1:])
        self.current_data_block[k] = parse_value(v)
        self.line_index += 1
        LOG.debug("Parsed data name: %s = %s", k, v)

    def parse_loop_block(self):
        LOG.debug("Parsing loop block")
        self.line_index += 1
        line = self.content_lines[self.line_index]
        keys = []
        while line.startswith("_"):
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
            for k, v in zip(keys, value.split()):
                self.current_data_block[k].append(parse_value(v))
        LOG.debug("Parsed loop block")

    def parse_comment_line(self):
        self.line_index += 1

    def parse_data_block_name(self):
        LOG.debug("Parsing data block name")
        line = self.content_lines[self.line_index]
        self.current_data_block_name = line[5:].strip()
        self.line_index += 1
        LOG.debug("Parsed data block name: %s", self.current_data_block_name)

    def parse(self, ignore_uncertainty=True):
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
                try:
                    if token in self.line_dispatch:
                        self.line_dispatch[token]()
                    elif token.startswith("_"):
                        self.parse_data_name()
                    elif token.startswith("data_"):
                        self.parse_data_block_name()
                    else:
                        LOG.debug("Skipping line: %s", line)
                        self.line_index += 1
                except Exception as e:
                    LOG.exception("Error in parser: %s", e)
                    raise ValueError(
                        f"Error parsing CIF, line number = "
                        f"{self.line_index + 1}: {e}"
                    ) from e
            else:
                self.line_index += 1
        self.line_index = 0
        return self.data

    @property
    def current_data_block(self):
        if self.current_data_block_name not in self.data:
            self.data[self.current_data_block_name] = {}
        return self.data[self.current_data_block_name]

    @classmethod
    def from_file(cls, filename):
        return cls.from_string(Path(filename).read_text())

    @classmethod
    def from_string(cls, contents):
        c = cls({})
        c.content_lines = contents.split("\n")
        c.parse()
        return c

    def to_string(self):
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
        Path(filename).write_text(self.to_string())
