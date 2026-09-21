import unittest

import numpy as np

from chmpy.crystal import Crystal
from chmpy.crystal.disorder import Disorder, analyse_disorder, disorder_components

HEADER = """data_{name}
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
"""

LABELLED_HEADER = HEADER + "_atom_site_disorder_assembly\n_atom_site_disorder_group\n"


def crystal_from(name, rows, labelled=False):
    header = LABELLED_HEADER if labelled else HEADER
    return Crystal.from_cif_string(header.format(name=name) + "\n".join(rows) + "\n")


class OrderedTestCase(unittest.TestCase):
    def setUp(self):
        self.crystal = crystal_from(
            "ordered", ["C1 C 0.0 0.0 0.0 1.0", "O1 O 0.1 0.0 0.0 1.0"]
        )

    def test_not_disordered(self):
        self.assertFalse(self.crystal.is_disordered)
        self.assertEqual(self.crystal.disorder.source, "none")
        self.assertEqual(self.crystal.disorder.n_components, 1)

    def test_components_is_the_crystal_itself(self):
        components = self.crystal.disorder_components()
        self.assertEqual(len(components), 1)
        self.assertIs(components[0], self.crystal)


class OccupancyDisorderTestCase(unittest.TestCase):
    """Disorder given only as partial occupancies, as most deposited CIFs do."""

    def setUp(self):
        self.crystal = crystal_from(
            "occ",
            [
                "C1  C 0.00 0.00 0.00 1.00",
                "O1A O 0.10 0.00 0.00 0.70",
                "O1B O 0.12 0.02 0.00 0.30",
                "N1A N 0.00 0.20 0.00 0.55",
                "N1B N 0.02 0.22 0.00 0.45",
            ],
        )

    def test_assemblies_are_inferred_from_occupancy(self):
        disorder = self.crystal.disorder
        self.assertEqual(disorder.source, "occupancy")
        self.assertTrue(disorder.is_resolvable)
        self.assertEqual(len(disorder.assemblies), 2)
        self.assertEqual(disorder.n_components, 2)

    def test_site_groups(self):
        groups = self.crystal.disorder.site_groups
        self.assertEqual(groups[0], ".")
        self.assertNotEqual(groups[1], groups[2])
        self.assertNotEqual(groups[3], groups[4])

    def test_components_take_the_major_alternative_first(self):
        major, minor = self.crystal.disorder_components()
        self.assertEqual(list(major.asymmetric_unit.labels), ["C1", "O1A", "N1A"])
        self.assertEqual(list(minor.asymmetric_unit.labels), ["C1", "O1B", "N1B"])

    def test_components_are_ordered(self):
        for component in self.crystal.disorder_components():
            self.assertFalse(component.is_disordered)
            np.testing.assert_allclose(
                component.asymmetric_unit.properties["occupation"], 1.0
            )

    def test_components_keep_cell_and_symmetry(self):
        for component in self.crystal.disorder_components():
            np.testing.assert_allclose(
                component.unit_cell.parameters, self.crystal.unit_cell.parameters
            )
            self.assertEqual(
                component.space_group.international_tables_number,
                self.crystal.space_group.international_tables_number,
            )

    def test_components_survive_a_cif_round_trip(self):
        for component in self.crystal.disorder_components():
            again = Crystal.from_cif_string(component.to_cif_string())
            self.assertEqual(
                list(again.asymmetric_unit.labels),
                list(component.asymmetric_unit.labels),
            )
            self.assertFalse(again.is_disordered)

    def test_the_parent_is_left_alone(self):
        before = len(self.crystal.asymmetric_unit)
        self.crystal.disorder_components()
        self.assertEqual(len(self.crystal.asymmetric_unit), before)


class LabelledDisorderTestCase(unittest.TestCase):
    """Disorder the file labels itself, which is believed over the occupancies."""

    def setUp(self):
        self.crystal = crystal_from(
            "labelled",
            [
                "C1  C 0.00 0.00 0.00 1.00 .  .",
                "O1A O 0.10 0.00 0.00 0.50 A  1",
                "O1B O 0.12 0.02 0.00 0.50 A  2",
                "N1A N 0.00 0.20 0.00 0.50 B  1",
                "N1B N 0.02 0.22 0.00 0.50 B  2",
            ],
            labelled=True,
        )

    def test_labels_are_used(self):
        "equal occupancies say nothing, but the labels do"
        disorder = self.crystal.disorder
        self.assertEqual(disorder.source, "disorder_group")
        self.assertEqual({a.name for a in disorder.assemblies}, {"A", "B"})
        self.assertEqual(disorder.n_components, 2)

    def test_components(self):
        major, minor = self.crystal.disorder_components()
        self.assertEqual(list(major.asymmetric_unit.labels), ["C1", "O1A", "N1A"])
        self.assertEqual(list(minor.asymmetric_unit.labels), ["C1", "O1B", "N1B"])

    def test_equal_occupancies_fall_back_to_the_labels(self):
        "0.5/0.5 says nothing, so the A/B naming has to carry it"
        bare = crystal_from(
            "bare",
            [
                "C1  C 0.00 0.00 0.00 1.00",
                "O1A O 0.10 0.00 0.00 0.50",
                "O1B O 0.12 0.02 0.00 0.50",
            ],
        )
        self.assertEqual(bare.disorder.source, "label")
        major, minor = bare.disorder_components()
        self.assertEqual(list(major.asymmetric_unit.labels), ["C1", "O1A"])
        self.assertEqual(list(minor.asymmetric_unit.labels), ["C1", "O1B"])

    def test_assemblies_with_unequal_group_counts(self):
        crystal = crystal_from(
            "three",
            [
                "C1  C 0.00 0.00 0.00 1.00 .  .",
                "O1A O 0.10 0.00 0.00 0.70 A  1",
                "O1B O 0.12 0.02 0.00 0.30 A  2",
                "N1A N 0.00 0.20 0.00 0.50 B  1",
                "N1B N 0.02 0.22 0.00 0.30 B  2",
                "N1C N 0.04 0.24 0.00 0.20 B  3",
            ],
            labelled=True,
        )
        self.assertEqual(crystal.disorder.n_components, 3)
        components = crystal.disorder_components()
        # assembly A runs out of alternatives, so keeps its last one
        self.assertEqual(
            list(components[2].asymmetric_unit.labels), ["C1", "O1B", "N1C"]
        )


class UnresolvableTestCase(unittest.TestCase):
    def setUp(self):
        "occupancies that do not pair up to 1.0 cannot be read as alternatives"
        self.crystal = crystal_from(
            "odd",
            [
                "C1  C 0.00 0.00 0.00 1.00",
                "H1A H 0.10 0.00 0.00 0.25",
                "H1B H 0.12 0.02 0.00 0.63",
                "H1C H 0.14 0.04 0.00 0.63",
            ],
        )

    def test_reported_rather_than_guessed(self):
        disorder = self.crystal.disorder
        self.assertTrue(disorder.is_disordered)
        self.assertFalse(disorder.is_resolvable)
        self.assertEqual(len(disorder.unresolved), 3)

    def test_strict_raises_and_names_the_sites(self):
        with self.assertRaises(ValueError) as caught:
            self.crystal.disorder_components()
        self.assertIn("H1A", str(caught.exception))

    def test_the_refusal_says_what_is_wrong(self):
        "0.25 + 0.63 + 0.63 on general positions is not a whole number of atoms"
        reason = self.crystal.disorder.reason
        self.assertIn("whole number", reason)

    def test_symmetry_imposed_disorder_is_named_as_such(self):
        """Three H over a site on a mirror: 0.25 + 0.63 + 0.63 times the
        multiplicity is a whole number of atoms, just not one per position."""
        crystal = Crystal.from_cif_string(
            """data_imposed
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_Int_Tables_number 2
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
-x,-y,-z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
N1  N 0.00 0.00 0.00 1.00
H1A H 0.10 0.00 0.00 0.50
H1B H 0.00 0.10 0.00 0.50
H1C H 0.00 0.00 0.10 0.50
"""
        )
        disorder = crystal.disorder
        self.assertFalse(disorder.is_resolvable)
        # 3 sites x 0.5 occupancy x multiplicity 2 = 3 whole hydrogens
        self.assertIn("3 atom(s) per unit cell", disorder.reason)
        self.assertIn("subgroup", disorder.reason)

    def test_not_strict_keeps_them(self):
        components = self.crystal.disorder_components(strict=False)
        self.assertEqual(len(components), 1)
        self.assertEqual(len(components[0].asymmetric_unit), 4)

    def test_a_disorder_worked_out_by_hand_can_be_supplied(self):
        from chmpy.crystal.disorder import DisorderAssembly

        disorder = Disorder(
            n_sites=4,
            assemblies=(
                DisorderAssembly(
                    name="A",
                    groups={"1": np.array([1]), "2": np.array([2, 3])},
                    occupancies={"1": 0.25, "2": 0.63},
                ),
            ),
            source="manual",
        )
        major, minor = disorder_components(self.crystal, disorder=disorder)
        self.assertEqual(list(major.asymmetric_unit.labels), ["C1", "H1B", "H1C"])
        self.assertEqual(list(minor.asymmetric_unit.labels), ["C1", "H1A"])


class RealFileTestCase(unittest.TestCase):
    """A deposited structure with two methyl orientations and no group labels."""

    CIF = HEADER.format(name="lutidine") + "\n".join(
        [
            "N1A  N  0.25742 0.81270 0.37067 1.000",
            "C2A  C  0.40630 0.73500 0.32155 1.000",
            "C7A  C  0.56340 0.71550 0.40036 1.000",
            "H7AA H  0.51180 0.64060 0.44600 0.768",
            "H7AB H  0.61880 0.83740 0.45100 0.768",
            "H7AC H  0.66200 0.65580 0.35550 0.768",
            "H7AD H  0.68330 0.78200 0.38900 0.232",
            "H7AE H  0.57620 0.58510 0.38390 0.232",
            "H7AF H  0.53310 0.76670 0.47950 0.232",
        ]
    )

    def test_methyl_disorder_resolves(self):
        crystal = Crystal.from_cif_string(self.CIF)
        disorder = crystal.disorder
        self.assertEqual(disorder.source, "occupancy")
        self.assertEqual(disorder.n_components, 2)
        major, minor = crystal.disorder_components()
        self.assertEqual(
            [x for x in major.asymmetric_unit.labels if x.startswith("H")],
            ["H7AA", "H7AB", "H7AC"],
        )
        self.assertEqual(
            [x for x in minor.asymmetric_unit.labels if x.startswith("H")],
            ["H7AD", "H7AE", "H7AF"],
        )

    def test_analyse_disorder_function(self):
        crystal = Crystal.from_cif_string(self.CIF)
        self.assertEqual(analyse_disorder(crystal).n_components, 2)


class LabelInferenceTestCase(unittest.TestCase):
    """The last resort: alternatives named by a trailing letter."""

    def test_a_whole_fragment(self):
        crystal = crystal_from(
            "fragment",
            [
                "C1  C 0.00 0.00 0.00 1.00",
                "C2A C 0.10 0.00 0.00 0.50",
                "C3A C 0.20 0.00 0.00 0.50",
                "C2B C 0.12 0.02 0.00 0.50",
                "C3B C 0.22 0.02 0.00 0.50",
            ],
        )
        major, minor = crystal.disorder_components()
        self.assertEqual(list(major.asymmetric_unit.labels), ["C1", "C2A", "C3A"])
        self.assertEqual(list(minor.asymmetric_unit.labels), ["C1", "C2B", "C3B"])

    def test_a_bare_stem_against_a_suffixed_one(self):
        "C1 and C1A are also a pair"
        crystal = crystal_from(
            "bare_stem",
            [
                "N1 N 0.00 0.00 0.00 1.00",
                "C1  C 0.10 0.00 0.00 0.50",
                "C1A C 0.12 0.02 0.00 0.50",
            ],
        )
        self.assertEqual(crystal.disorder.source, "label")
        major, minor = crystal.disorder_components()
        self.assertEqual(list(major.asymmetric_unit.labels), ["N1", "C1"])
        self.assertEqual(list(minor.asymmetric_unit.labels), ["N1", "C1A"])

    def test_three_alternatives(self):
        crystal = crystal_from(
            "three_way",
            [
                "C1  C 0.00 0.00 0.00 1.00",
                "O1A O 0.10 0.00 0.00 0.50",
                "O1B O 0.12 0.02 0.00 0.30",
                "O1C O 0.14 0.04 0.00 0.20",
            ],
        )
        self.assertEqual(crystal.disorder.n_components, 3)
        components = crystal.disorder_components()
        self.assertEqual(list(components[0].asymmetric_unit.labels), ["C1", "O1A"])
        self.assertEqual(list(components[2].asymmetric_unit.labels), ["C1", "O1C"])

    def test_labels_are_not_trusted_when_the_occupancies_disagree(self):
        "the naming may say A/B, but if they do not add to 1.0 it means nothing"
        crystal = crystal_from(
            "mismatch",
            [
                "C1  C 0.00 0.00 0.00 1.00",
                "O1A O 0.10 0.00 0.00 0.50",
                "O1B O 0.12 0.02 0.00 0.30",
            ],
        )
        self.assertFalse(crystal.disorder.is_resolvable)
        with self.assertRaises(ValueError):
            crystal.disorder_components()

    def test_no_suffix_convention_to_go_on(self):
        crystal = crystal_from(
            "no_suffix",
            [
                "C1 C 0.00 0.00 0.00 1.00",
                "O1 O 0.10 0.00 0.00 0.50",
                "O2 O 0.12 0.02 0.00 0.50",
            ],
        )
        self.assertFalse(crystal.disorder.is_resolvable)

    def test_occupancy_pairing_wins_over_labels(self):
        """H7AA..H7AF share a stem but are two methyl orientations, not six.

        The occupancies settle it, and must be consulted first.
        """
        crystal = crystal_from(
            "methyl",
            [
                "C7A  C 0.5634 0.7155 0.4004 1.000",
                "H7AA H 0.5118 0.6406 0.4460 0.768",
                "H7AB H 0.6188 0.8374 0.4510 0.768",
                "H7AC H 0.6620 0.6558 0.3555 0.768",
                "H7AD H 0.6833 0.7820 0.3890 0.232",
                "H7AE H 0.5762 0.5851 0.3839 0.232",
                "H7AF H 0.5331 0.7667 0.4795 0.232",
            ],
        )
        self.assertEqual(crystal.disorder.source, "occupancy")
        self.assertEqual(crystal.disorder.n_components, 2)


class DescendTestCase(unittest.TestCase):
    """Alternatives related by symmetry, which only a subgroup can tell apart."""

    # An oxygen disordered across the inversion centre: the two alternatives
    # are one orbit in P-1, so nothing can choose between them there.
    ACROSS_INVERSION = """data_across_inversion
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_Int_Tables_number 2
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
-x,-y,-z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0.30 0.40 0.10 1.00
O1 O 0.02 0.00 0.00 0.50
"""

    def setUp(self):
        self.crystal = Crystal.from_cif_string(self.ACROSS_INVERSION)

    def test_not_resolvable_in_the_parent_group(self):
        disorder = self.crystal.disorder
        self.assertFalse(disorder.is_resolvable)
        self.assertIn("symmetry equivalent", disorder.reason)
        with self.assertRaises(ValueError):
            self.crystal.disorder_components()

    def test_descending_resolves_it(self):
        components = self.crystal.disorder_components(descend=True)
        self.assertEqual(len(components), 2)
        for component in components:
            self.assertFalse(component.is_disordered)
            np.testing.assert_allclose(
                component.asymmetric_unit.properties["occupation"], 1.0
            )

    def test_the_components_are_in_a_subgroup(self):
        components = self.crystal.disorder_components(descend=True)
        parent = len(self.crystal.space_group.symmetry_operations)
        for component in components:
            self.assertLess(len(component.space_group.symmetry_operations), parent)

    def test_the_atom_count_per_unit_cell_is_unchanged(self):
        for component in self.crystal.disorder_components(descend=True):
            self.assertAlmostEqual(component.density, self.crystal.density)

    def test_the_components_differ(self):
        major, minor = self.crystal.disorder_components(descend=True)
        self.assertNotEqual(
            list(major.asymmetric_unit.labels), list(minor.asymmetric_unit.labels)
        )

    def test_descending_is_off_by_default(self):
        with self.assertRaises(ValueError):
            self.crystal.disorder_components(descend=False)

    def test_an_ordered_crystal_is_untouched(self):
        ordered = crystal_from("ordered", ["C1 C 0.0 0.0 0.0 1.0"])
        self.assertEqual(len(ordered.disorder_components(descend=True)), 1)

    def test_descending_does_not_disturb_resolvable_disorder(self):
        crystal = crystal_from(
            "occ",
            [
                "C1  C 0.00 0.00 0.00 1.00",
                "O1A O 0.10 0.00 0.00 0.70",
                "O1B O 0.12 0.02 0.00 0.30",
            ],
        )
        with_descend = crystal.disorder_components(descend=True)
        without = crystal.disorder_components()
        self.assertEqual(len(with_descend), len(without))
        for a, b in zip(with_descend, without, strict=True):
            self.assertEqual(
                list(a.asymmetric_unit.labels), list(b.asymmetric_unit.labels)
            )

    def test_a_smeared_model_is_refused_rather_than_ordered(self):
        """Two H straddling the mirror their nitrogen sits on, too close together.

        Lowering the symmetry lets a choice be made, but every choice puts two
        hydrogens a bond length apart, which proves the alternatives were never
        a discrete set of arrangements.
        """
        crystal = Crystal.from_cif_string(
            """data_smeared
_cell_length_a 8.0
_cell_length_b 8.0
_cell_length_c 8.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_Int_Tables_number 6
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
-x,y,z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
N1 N 0.000 0.000 0.000 1.00
H1 H 0.020 0.120 0.000 0.50
H2 H 0.030 0.100 0.050 0.50
"""
        )
        self.assertFalse(crystal.disorder.is_resolvable)
        with self.assertRaises(ValueError) as caught:
            crystal.disorder_components(descend=True)
        self.assertIn("bonded neighbour", str(caught.exception))
        # it can still be forced, for a look at what the descent produced
        self.assertEqual(
            len(crystal.disorder_components(descend=True, strict=False)), 2
        )
