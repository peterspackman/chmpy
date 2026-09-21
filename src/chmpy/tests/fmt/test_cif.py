import logging
import unittest
from tempfile import TemporaryDirectory

from chmpy.fmt.cif import (
    Cif,
    CifBlock,
    CifOptions,
    CifParseError,
    canonical_data_name,
    format_field,
    is_scalar,
    needs_quote,
    parse_quote,
    parse_value,
    register_aliases,
    tokenize,
)

LOG = logging.getLogger(__name__)

SMALL_MOLECULE = """\
# a comment before anything
data_example
_symmetry_space_group_name_H-M   'P 21/c'
_symmetry_Int_Tables_number      14
_cell_length_a                   5.1234(3)
_cell_length_b                   6.0
_cell_angle_beta                 102.345(12)
_chemical_name_systematic
;
 first line
 second line
;
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'-x, -y, -z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_occupancy
_atom_site_disorder_group
C1 C 0.1234(5) 1.000 .
O1 O 0.5678(6) 0.500 ?
"""


class TokenizerTestCase(unittest.TestCase):
    def test_comments_are_dropped(self):
        self.assertEqual(tokenize("_a 1 # trailing\n_b 2\n"), ["_a", "1", "_b", "2"])

    def test_quoted_string_keeps_delimiters(self):
        self.assertEqual(tokenize("_a 'two words'"), ["_a", "'two words'"])

    def test_quote_closes_only_before_whitespace(self):
        "a CIF quote is only a delimiter when whitespace follows it"
        self.assertEqual(tokenize("_a 'it's here' _b"), ["_a", "'it's here'", "_b"])

    def test_apostrophe_inside_bare_token(self):
        self.assertEqual(tokenize("_a 5'-AMP _b 1"), ["_a", "5'-AMP", "_b", "1"])

    def test_hash_inside_quotes_is_not_a_comment(self):
        self.assertEqual(tokenize("_a 'x # y' _b 1"), ["_a", "'x # y'", "_b", "1"])

    def test_text_field(self):
        self.assertEqual(
            tokenize("_a\n;\none\ntwo\n;\n_b 1"), ["_a", ";\none\ntwo\n;", "_b", "1"]
        )

    def test_semicolon_away_from_column_one(self):
        "only a semicolon in the first column opens a text field"
        self.assertEqual(tokenize("_a x;y\n_b 1"), ["_a", "x;y", "_b", "1"])

    def test_unterminated_quote(self):
        with self.assertRaises(CifParseError):
            tokenize("_a 'oops\n_b 1\n")

    def test_unterminated_text_field(self):
        with self.assertRaises(CifParseError):
            tokenize("_a\n;\noops\n")


class ParserTestCase(unittest.TestCase):
    def setUp(self):
        self.cif = Cif.from_string(SMALL_MOLECULE)
        self.block = self.cif["example"]

    def test_block_names(self):
        self.assertEqual(list(self.cif), ["example"])
        self.assertIs(self.cif.first_block, self.block)

    def test_scalar_types(self):
        self.assertEqual(self.block["cell_length_a"], 5.1234)
        self.assertEqual(self.block["cell_length_b"], 6.0)
        self.assertEqual(self.block["symmetry_Int_Tables_number"], 14)
        self.assertIsInstance(self.block["symmetry_Int_Tables_number"], int)
        self.assertEqual(self.block["symmetry_space_group_name_H-M"], "P 21/c")

    def test_text_field_keeps_its_line_breaks(self):
        self.assertEqual(
            self.block["chemical_name_systematic"], " first line\n second line"
        )

    def test_loop_columns(self):
        self.assertEqual(self.block["atom_site_label"], ["C1", "O1"])
        self.assertEqual(self.block["atom_site_fract_x"], [0.1234, 0.5678])
        self.assertEqual(self.block["atom_site_occupancy"], [1.0, 0.5])
        self.assertEqual(
            self.block["symmetry_equiv_pos_as_xyz"], ["x, y, z", "-x, -y, -z"]
        )

    def test_null_values_stay_as_written(self):
        self.assertEqual(self.block["atom_site_disorder_group"], [".", "?"])

    def test_loops_are_remembered(self):
        self.assertTrue(self.block.is_loop("atom_site_label"))
        self.assertFalse(self.block.is_loop("cell_length_a"))
        self.assertEqual(len(self.block.loops), 2)

    def test_rows_may_wrap_across_lines(self):
        block = Cif.from_string("data_a\nloop_\n_p\n_q\n_r\n1 2\n3\n4 5 6\n")["a"]
        self.assertEqual(block["p"], [1, 4])
        self.assertEqual(block["r"], [3, 6])

    def test_multiple_data_blocks(self):
        cif = Cif.from_string("data_one\n_a 1\ndata_two\n_a 2\n")
        self.assertEqual([cif["one"]["a"], cif["two"]["a"]], [1, 2])

    def test_reserved_words_are_case_insensitive(self):
        cif = Cif.from_string("DATA_a\nLOOP_\n_p\n_q\n1 2\n3 4\n")
        self.assertEqual(cif["a"]["p"], [1, 3])

    def test_data_names_before_any_block(self):
        self.assertEqual(Cif.from_string("_a 1\n")["unknown"]["a"], 1)

    def test_save_frames(self):
        cif = Cif.from_string("data_d\nsave_f\n_x 1\nsave_\n_y 2\n")
        self.assertEqual(cif["d"]["y"], 2)
        self.assertEqual(cif["d"].save_frames["f"]["x"], 1)

    def test_symmetry_codes_are_not_numbers(self):
        "int() reads 1_555 as 1555; a CIF symmetry code is a string"
        block = Cif.from_string("data_a\nloop_\n_s\n1_555\n2_665\n")["a"]
        self.assertEqual(block["s"], ["1_555", "2_665"])

    def test_unquoted_symmetry_operations(self):
        block = Cif.from_string("data_a\nloop_\n_s\n-x,-y,-z\n1/2+x,y,z\n")["a"]
        self.assertEqual(block["s"], ["-x,-y,-z", "1/2+x,y,z"])

    def test_exponents(self):
        block = Cif.from_string("data_a\n_x 1.5e-3\n_y 2E4\n")["a"]
        self.assertEqual([block["x"], block["y"]], [0.0015, 20000.0])

    def test_crlf_line_endings(self):
        block = Cif.from_string("data_a\r\n_x 1\r\nloop_\r\n_p\r\n7\r\n8\r\n")["a"]
        self.assertEqual([block["x"], block["p"]], [1, [7, 8]])

    def test_ragged_loop_drops_partial_row(self):
        with self.assertLogs("chmpy.fmt.cif", level="WARNING"):
            block = Cif.from_string("data_a\nloop_\n_p\n_q\n1 2\n3\n")["a"]
        self.assertEqual(block["p"], [1])

    def test_data_name_without_a_value(self):
        with self.assertRaises(CifParseError):
            Cif.from_string("data_a\n_x\n")
        with self.assertRaises(CifParseError):
            Cif.from_string("data_a\n_x\n_y 1\n")

    def test_loop_without_data_names(self):
        with self.assertRaises(CifParseError):
            Cif.from_string("data_a\nloop_\n1 2\n")


class AliasTestCase(unittest.TestCase):
    def test_exact_key(self):
        block = CifBlock({"cell_length_a": 5.0})
        self.assertEqual(block["cell_length_a"], 5.0)

    def test_leading_underscore_and_case(self):
        block = CifBlock({"Cell_Length_A": 5.0})
        self.assertEqual(block["_cell_length_a"], 5.0)
        self.assertEqual(block.get("CELL_LENGTH_A"), 5.0)
        self.assertIn("cell_length_a", block)

    def test_mmcif_spelling(self):
        block = Cif.from_string("data_a\n_cell.length_a 5.0\n")["a"]
        self.assertEqual(block["cell_length_a"], 5.0)
        self.assertEqual(block["_cell.length_a"], 5.0)

    def test_core_cif_read_through_mmcif_names(self):
        block = Cif.from_string("data_a\n_cell_length_a 5.0\n")["a"]
        self.assertEqual(block["cell.length_a"], 5.0)

    def test_missing_key(self):
        block = CifBlock({"cell_length_a": 5.0})
        self.assertNotIn("cell_volume", block)
        self.assertIsNone(block.get("cell_volume"))
        with self.assertRaises(KeyError):
            block["cell_volume"]

    def test_keys_added_later_are_found(self):
        block = CifBlock({"cell_length_a": 5.0})
        self.assertNotIn("cell_length_b", block)
        block["Cell_Length_B"] = 6.0
        self.assertEqual(block["cell_length_b"], 6.0)

    def test_register_aliases(self):
        register_aliases("cell_volume", "test_program_volume")
        block = Cif.from_string("data_a\n_test_program_volume 100.0\n")["a"]
        self.assertEqual(block["cell_volume"], 100.0)

    def test_canonical_data_name(self):
        self.assertEqual(canonical_data_name("_cell.length_a"), "cell_length_a")
        self.assertEqual(canonical_data_name("unknown_name"), "unknown_name")


class WriterTestCase(unittest.TestCase):
    def test_needs_quote(self):
        self.assertTrue(needs_quote("this string will need quoting"))
        self.assertFalse(needs_quote("thiswon't"))
        self.assertFalse(needs_quote(3.45))
        self.assertTrue(needs_quote(""))
        self.assertTrue(needs_quote("_looks_like_a_tag"))
        self.assertTrue(needs_quote("."))

    def test_is_scalar(self):
        self.assertTrue(is_scalar(3))
        self.assertTrue(is_scalar("test string"))
        self.assertFalse(is_scalar([3, 4, 5]))
        self.assertFalse(is_scalar((3, 4)))

    def test_format_field(self):
        self.assertTrue(len(format_field(5.4)) == 20)
        self.assertTrue(format_field("3.4") == "3.4")
        self.assertEqual(format_field(None), "?")

    def test_round_trip(self):
        cif = Cif.from_string(SMALL_MOLECULE)
        again = Cif.from_string(cif.to_string())
        self.assertEqual(again["example"], cif["example"])

    def test_round_trip_keeps_loops_together(self):
        cif = Cif.from_string(SMALL_MOLECULE)
        again = Cif.from_string(cif.to_string())
        self.assertEqual(again["example"].loops, cif["example"].loops)

    def test_round_trip_of_awkward_strings(self):
        values = {
            "with_space": "two words",
            "with_apostrophe": "it's",
            "with_both": 'it\'s "quoted"',
            "multiline": "one\ntwo",
            "empty": "",
        }
        cif = Cif({"x": values})
        self.assertEqual(dict(Cif.from_string(cif.to_string())["x"]), values)

    def test_to_file(self):
        from pathlib import Path

        cif = Cif.from_string(SMALL_MOLECULE)
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "out.cif")
            cif.to_file(path)
            self.assertEqual(Cif.from_file(path)["example"], cif["example"])

    def test_hand_built_data(self):
        cif = Cif({"crystal": {"loop_block": [1, 2], "quote_block": "test this"}})
        self.assertEqual(
            dict(Cif.from_string(cif.to_string())["crystal"]), cif["crystal"]
        )


class ValueHelperTestCase(unittest.TestCase):
    def test_parse_value(self):
        self.assertEqual(parse_value("2.3(1)", with_uncertainty=True), (2.3, 1))
        self.assertEqual(parse_value("string help"), "string help")
        self.assertEqual(parse_value("3"), 3)
        self.assertEqual(parse_value("3", with_uncertainty=True), (3, 0))

    def test_parse_quote(self):
        self.assertEqual(parse_quote(";quote text;"), "quote text")
        self.assertEqual(parse_quote("'-y, x-y, z'", delimiter="'"), "-y, x-y, z")
        self.assertEqual(parse_quote("no quotes"), "no quotes")


class OptionsTestCase(unittest.TestCase):
    TEXT_FIELD = "data_a\n_x\n;\nfirst line\n  second  line\n\nthird\n;\n_y 1\n"

    def read(self, text, **settings):
        return Cif.from_string(text, options=CifOptions(**settings))["a"]

    def test_text_field_keeps_its_lines_by_default(self):
        self.assertEqual(
            Cif.from_string(self.TEXT_FIELD)["a"]["x"],
            "first line\n  second  line\n\nthird",
        )

    def test_join_text_field_lines(self):
        block = self.read(self.TEXT_FIELD, join_text_field_lines=True)
        self.assertEqual(block["x"], "first line second  line third")
        self.assertEqual(block["y"], 1)

    def test_triple_quotes_are_off_by_default(self):
        "under CIF 1.1 the outer quotes delimit and the inner two are content"
        text = "data_a\n_x '''one line'''\n"
        self.assertEqual(Cif.from_string(text)["a"]["x"], "''one line''")

    def test_triple_quoted_string(self):
        text = "data_a\n_x '''one line'''\n_y 1\n"
        block = self.read(text, cif2_triple_quotes=True)
        self.assertEqual(block["x"], "one line")
        self.assertEqual(block["y"], 1)

    def test_triple_quoted_string_spans_lines(self):
        text = "data_a\n_x '''spans\ntwo lines'''\n_y 1\n"
        self.assertEqual(
            self.read(text, cif2_triple_quotes=True)["x"], "spans\ntwo lines"
        )

    def test_triple_quoted_string_may_hold_quotes(self):
        text = "data_a\n_x '''has 'one' and ''two'' inside'''\n"
        self.assertEqual(
            self.read(text, cif2_triple_quotes=True)["x"],
            "has 'one' and ''two'' inside",
        )

    def test_triple_double_quotes(self):
        text = 'data_a\n_x """double\nquoted"""\n'
        self.assertEqual(
            self.read(text, cif2_triple_quotes=True)["x"], "double\nquoted"
        )

    def test_empty_triple_quoted_string(self):
        text = "data_a\n_x ''''''\n_y 1\n"
        self.assertEqual(self.read(text, cif2_triple_quotes=True)["x"], "")

    def test_triple_quotes_leave_ordinary_quotes_alone(self):
        "CIF 1.1 quoting still applies, so an apostrophe needs no escaping"
        text = "data_a\n_x 'it's here'\n"
        self.assertEqual(self.read(text, cif2_triple_quotes=True)["x"], "it's here")

    def test_triple_quoted_value_in_a_loop(self):
        text = "data_a\nloop_\n_p\n_q\n'''a\nb''' 1\n'plain' 2\n"
        block = self.read(text, cif2_triple_quotes=True)
        self.assertEqual(block["p"], ["a\nb", "plain"])
        self.assertEqual(block["q"], [1, 2])

    def test_unterminated_triple_quoted_string(self):
        with self.assertRaises(CifParseError):
            self.read("data_a\n_x '''never closed\n", cif2_triple_quotes=True)

    def test_block_name_with_whitespace(self):
        "CIF cannot quote a block name, so whitespace in one must not survive"
        text = Cif({"my structure": {"a": 1}}).to_string()
        self.assertEqual(list(Cif.from_string(text)), ["my_structure"])

    def test_ragged_loop_is_refused(self):
        "dropping rows to make a loop rectangular would lose data silently"
        block = Cif.from_string("data_a\nloop_\n_p\n_q\n1 2\n3 4\n")["a"]
        block["p"] = [1, 2, 3]
        with self.assertRaises(ValueError):
            Cif({"a": block}).to_string()
