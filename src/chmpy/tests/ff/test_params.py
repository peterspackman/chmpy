import logging
import unittest

from chmpy.crystal import Crystal
from chmpy.ff.params import (
    assign_uff_type_from_coordination,
    crystal_uff_params,
    get_uff_parameters,
    load_lj_params,
    molecule_uff_params,
)

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class UFFParamsTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test crystals from available test files."""
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.ice_ii = Crystal.load(TEST_FILES["iceII.cif"])

    def test_load_lj_params(self):
        """Test loading of LJ parameters from JSON file."""
        params = load_lj_params()

        # Check that both force fields are present
        self.assertIn("uff", params)
        self.assertIn("uff4mof", params)

        # Check some standard UFF types exist
        uff_params = params["uff"]
        self.assertIn("H_", uff_params)
        self.assertIn("C_3", uff_params)
        self.assertIn("O_3", uff_params)

    def test_coordination_based_typing(self):
        """Test UFF type assignment from coordination numbers."""

        # Test carbon coordination thresholds
        self.assertEqual(assign_uff_type_from_coordination(6, 3.8), "C_3")  # sp3
        self.assertEqual(assign_uff_type_from_coordination(6, 2.7), "C_2")  # sp2
        self.assertEqual(assign_uff_type_from_coordination(6, 1.8), "C_1")  # sp

        # Test oxygen coordination thresholds
        self.assertEqual(assign_uff_type_from_coordination(8, 1.8), "O_3")  # bent
        self.assertEqual(assign_uff_type_from_coordination(8, 1.2), "O_2")  # linear

        # Test nitrogen
        self.assertEqual(assign_uff_type_from_coordination(7, 2.8), "N_3")  # pyramidal
        self.assertEqual(assign_uff_type_from_coordination(7, 1.8), "N_2")  # trigonal

    def test_crystal_uff_parameters(self):
        """Test UFF parameter assignment for crystal structures."""

        atom_types, parameters = crystal_uff_params(self.acetic, force_field="uff")

        uc_atoms = self.acetic.unit_cell_atoms()
        n_atoms = len(uc_atoms["element"])

        # Check we get results for all atoms
        self.assertEqual(len(atom_types), n_atoms)
        self.assertEqual(len(parameters), n_atoms)

        # Check parameter format
        for i in range(n_atoms):
            uff_type = atom_types[i]
            params = parameters[i]

            self.assertIsInstance(uff_type, str)
            self.assertIn("sigma", params)
            self.assertIn("epsilon", params)
            self.assertGreater(params["sigma"], 0)
            self.assertGreater(params["epsilon"], 0)

    def test_acetic_acid_molecule_typing(self):
        """Test UFF typing for acetic acid molecules using EEQ coordination."""

        molecules = self.acetic.unit_cell_molecules()
        self.assertGreater(len(molecules), 0, "Should have acetic acid molecules")

        mol = molecules[0]

        # Print debug info for acetic acid
        print(f"\nAcetic acid molecule: {mol.molecular_formula}")
        coord_nums = mol.coordination_numbers

        for i, (atomic_num, coord_num) in enumerate(
            zip(mol.atomic_numbers, coord_nums, strict=False)
        ):
            from chmpy.core.element import Element

            symbol = Element.from_atomic_number(atomic_num).symbol
            uff_type = assign_uff_type_from_coordination(atomic_num, coord_num)
            print(f"  Atom {i + 1}: {symbol} coord={coord_num:.2f} → {uff_type}")

        atom_types, parameters = molecule_uff_params(mol, force_field="uff")

        # Get types by element
        carbon_types = [
            atom_types[i]
            for i, atomic_num in enumerate(mol.atomic_numbers)
            if atomic_num == 6
        ]
        oxygen_types = [
            atom_types[i]
            for i, atomic_num in enumerate(mol.atomic_numbers)
            if atomic_num == 8
        ]

        print(f"  Carbon types: {carbon_types}")
        print(f"  Oxygen types: {oxygen_types}")

        # Should have both sp2 and sp3 carbons in acetic acid
        self.assertIn(
            "C_2", carbon_types, f"Should have sp2 carbon, got: {carbon_types}"
        )
        self.assertIn(
            "C_3", carbon_types, f"Should have sp3 carbon, got: {carbon_types}"
        )

        # Should have both carbonyl and hydroxyl oxygens
        self.assertIn(
            "O_2", oxygen_types, f"Should have carbonyl oxygen, got: {oxygen_types}"
        )
        self.assertIn(
            "O_3", oxygen_types, f"Should have hydroxyl oxygen, got: {oxygen_types}"
        )

    def test_ice_structure_typing(self):
        """Test UFF typing for ice structure (only H and O)."""

        atom_types, parameters = get_uff_parameters(self.ice_ii, force_field="uff")

        # Ice should only have hydrogen and oxygen
        unique_types = set(atom_types.values())

        for uff_type in unique_types:
            self.assertTrue(
                uff_type.startswith("H_") or uff_type.startswith("O_"),
                f"Ice should only have H and O atoms, found: {uff_type}",
            )
