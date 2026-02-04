"""
Space group multiplication table for algebraic symmetry operations.

This module provides the SpaceGroupTable class which precomputes the Cayley
table (multiplication table) of a space group. This enables purely algebraic
dimer equivalence determination without numerical tolerances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .space_group import SpaceGroup

LOG = logging.getLogger(__name__)


@dataclass
class SpaceGroupTable:
    """
    Precomputed group algebra (Cayley table) for a space group.

    This class stores the multiplication table and inverse table for a space
    group, enabling efficient algebraic operations on symmetry-related objects
    without numerical tolerance comparisons.

    Attributes:
        n_ops: Number of symmetry operations in the group
        mult_table: (n_ops, n_ops) int array where mult_table[i, j] = k means
            g_k = g_j ∘ g_i (composition: apply g_i first, then g_j)
        inverse_table: (n_ops,) int array where inverse_table[i] = j means
            g_j = g_i^{-1}
        rotations: (n_ops, 3, 3) rotation matrices for each symop
        translations: (n_ops, 3) translation vectors for each symop
        symop_to_idx: Dict mapping symop integer_code to index in the table
        idx_to_symop: (n_ops,) symop integer codes for each index
    """

    n_ops: int
    mult_table: np.ndarray
    inverse_table: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    symop_to_idx: dict[int, int]
    idx_to_symop: np.ndarray

    @classmethod
    def from_space_group(cls, sg: SpaceGroup) -> SpaceGroupTable:
        """
        Build a SpaceGroupTable from a SpaceGroup.

        This constructs the complete Cayley table by composing all pairs of
        symmetry operations and finding their result in the group.

        Args:
            sg: The space group to build the table from

        Returns:
            SpaceGroupTable with precomputed multiplication and inverse tables
        """
        symops = sg.symmetry_operations
        n_ops = len(symops)

        # Build index mappings
        # Use integer_code as the canonical identifier
        symop_to_idx = {s.integer_code: i for i, s in enumerate(symops)}
        idx_to_symop = np.array([s.integer_code for s in symops], dtype=np.int32)

        # Extract rotations and translations
        rotations = np.array([s.rotation for s in symops])
        translations = np.array([s.translation for s in symops])

        # Build multiplication table: mult[i, j] = index of (g_j ∘ g_i)
        # Note: g_j ∘ g_i means apply g_i first, then g_j
        mult_table = np.zeros((n_ops, n_ops), dtype=np.int32)

        for i, g_i in enumerate(symops):
            for j, g_j in enumerate(symops):
                # Compose: g_result = g_j ∘ g_i
                g_result = g_j.compose(g_i)
                result_code = g_result.integer_code

                if result_code not in symop_to_idx:
                    # This can happen if translations differ by a lattice vector
                    # Find the matching symop by rotation
                    result_code = _find_matching_symop(
                        g_result, symops, symop_to_idx
                    )

                mult_table[i, j] = symop_to_idx[result_code]

        # Build inverse table: inv[i] = index of g_i^{-1}
        inverse_table = np.zeros(n_ops, dtype=np.int32)

        for i, g_i in enumerate(symops):
            g_inv = g_i.inverse()
            inv_code = g_inv.integer_code

            if inv_code not in symop_to_idx:
                inv_code = _find_matching_symop(g_inv, symops, symop_to_idx)

            inverse_table[i] = symop_to_idx[inv_code]

        return cls(
            n_ops=n_ops,
            mult_table=mult_table,
            inverse_table=inverse_table,
            rotations=rotations,
            translations=translations,
            symop_to_idx=symop_to_idx,
            idx_to_symop=idx_to_symop,
        )

    def compose(self, i: int, j: int) -> int:
        """
        Return the index of the composition g_j ∘ g_i.

        Args:
            i: Index of first symop (applied first)
            j: Index of second symop (applied second)

        Returns:
            Index k such that g_k = g_j ∘ g_i
        """
        return self.mult_table[i, j]

    def inv(self, i: int) -> int:
        """
        Return the index of the inverse of g_i.

        Args:
            i: Index of symop

        Returns:
            Index j such that g_j = g_i^{-1}
        """
        return self.inverse_table[i]

    def identity_idx(self) -> int:
        """Return the index of the identity symop."""
        # The identity has integer_code 16484
        return self.symop_to_idx.get(16484, 0)

    def verify_group_axioms(self) -> dict[str, bool]:
        """
        Verify that the multiplication table satisfies group axioms.

        Returns:
            Dictionary with verification results:
            - 'closure': All compositions are in the group
            - 'identity': There exists an identity element
            - 'inverses': Every element has an inverse
            - 'associativity': Composition is associative
        """
        n = self.n_ops
        results = {}

        # Closure: mult_table values should all be valid indices
        results["closure"] = np.all((self.mult_table >= 0) & (self.mult_table < n))

        # Identity: There should be an identity element e such that
        # e * g = g * e = g for all g
        identity_found = False
        for e in range(n):
            is_identity = True
            for g in range(n):
                if self.mult_table[g, e] != g or self.mult_table[e, g] != g:
                    is_identity = False
                    break
            if is_identity:
                identity_found = True
                break
        results["identity"] = identity_found

        # Inverses: For each g, there exists h such that g * h = h * g = e
        e = self.identity_idx()
        all_have_inverses = True
        for g in range(n):
            has_inverse = False
            for h in range(n):
                if self.mult_table[g, h] == e and self.mult_table[h, g] == e:
                    has_inverse = True
                    break
            if not has_inverse:
                all_have_inverses = False
                break
        results["inverses"] = all_have_inverses

        # Associativity: (a * b) * c = a * (b * c) for all a, b, c
        # This is expensive O(n^3), so sample if n is large
        associative = True
        if n <= 48:  # Check all for reasonable sizes
            for a in range(n):
                for b in range(n):
                    ab = self.mult_table[a, b]
                    for c in range(n):
                        # (a * b) * c
                        ab_c = self.mult_table[ab, c]
                        # a * (b * c)
                        bc = self.mult_table[b, c]
                        a_bc = self.mult_table[a, bc]
                        if ab_c != a_bc:
                            associative = False
                            LOG.error(
                                f"Associativity failed: ({a}*{b})*{c}={ab_c} "
                                f"!= {a}*({b}*{c})={a_bc}"
                            )
                            break
                    if not associative:
                        break
                if not associative:
                    break
        else:
            # Sample 1000 random triples
            rng = np.random.default_rng(42)
            for _ in range(1000):
                a, b, c = rng.integers(0, n, size=3)
                ab = self.mult_table[a, b]
                ab_c = self.mult_table[ab, c]
                bc = self.mult_table[b, c]
                a_bc = self.mult_table[a, bc]
                if ab_c != a_bc:
                    associative = False
                    break
        results["associativity"] = associative

        return results


def _find_matching_symop(
    target: "SymmetryOperation",
    symops: list,
    symop_to_idx: dict[int, int],
) -> int:
    """
    Find a symop in the list that matches the target up to lattice translation.

    This handles the case where a composed symmetry operation differs from
    an existing one only by a lattice translation (which is equivalent under
    periodic boundary conditions).

    Args:
        target: The symmetry operation to find
        symops: List of symmetry operations to search
        symop_to_idx: Mapping from integer_code to index

    Returns:
        The integer_code of the matching symop
    """
    target_rot = target.rotation
    target_trans = target.translation % 1  # Wrap to [0, 1)

    for s in symops:
        # Check if rotations match
        if not np.allclose(s.rotation, target_rot, atol=1e-10):
            continue

        # Check if translations match modulo 1
        s_trans = s.translation % 1
        if np.allclose(s_trans, target_trans, atol=1e-10):
            return s.integer_code

    # Fallback: shouldn't happen for valid space groups
    LOG.warning(
        f"Could not find matching symop for {target}, using first match by rotation"
    )
    for s in symops:
        if np.allclose(s.rotation, target_rot, atol=1e-10):
            return s.integer_code

    raise ValueError(f"No matching symop found for {target}")
