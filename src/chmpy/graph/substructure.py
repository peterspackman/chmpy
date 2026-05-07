"""
Substructure matching using the VF2++ algorithm.

Provides subgraph isomorphism detection for finding molecular substructures
without requiring external dependencies like graph_tool.

VF2++ improves on VF2 with:
- Optimal matching order based on node rarity and connectivity
- Improved feasibility rules with better lookahead
- More aggressive pruning
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .adjacency import MolecularGraph

LOG = logging.getLogger(__name__)


def find_substructure(
    target: "MolecularGraph",
    query: "MolecularGraph",
    match_element: bool = True,
) -> list[dict[int, int]]:
    """
    Find all occurrences of a query substructure in a target graph.

    Uses the VF2++ algorithm for subgraph isomorphism.

    Args:
        target: The target molecular graph to search in.
        query: The query pattern to search for.
        match_element: If True, only match atoms with same atomic number.

    Returns:
        List of mappings from query atom indices to target atom indices.
        Each mapping is a dict {query_idx: target_idx}.
    """
    if query.n_atoms == 0:
        return [{}]

    if query.n_atoms > target.n_atoms:
        return []

    matcher = _VF2PlusPlusMatcher(target, query, match_element)
    return matcher.find_all()


def has_substructure(
    target: "MolecularGraph",
    query: "MolecularGraph",
    match_element: bool = True,
) -> bool:
    """
    Check if a target graph contains a query substructure.

    More efficient than find_substructure when only existence is needed.

    Args:
        target: The target molecular graph to search in.
        query: The query pattern to search for.
        match_element: If True, only match atoms with same atomic number.

    Returns:
        True if the query is found in the target.
    """
    if query.n_atoms == 0:
        return True

    if query.n_atoms > target.n_atoms:
        return False

    matcher = _VF2PlusPlusMatcher(target, query, match_element)
    return matcher.find_one() is not None


def count_substructures(
    target: "MolecularGraph",
    query: "MolecularGraph",
    match_element: bool = True,
) -> int:
    """
    Count occurrences of a query substructure in a target graph.

    Args:
        target: The target molecular graph to search in.
        query: The query pattern to search for.
        match_element: If True, only match atoms with same atomic number.

    Returns:
        Number of times the query appears in the target.
    """
    return len(find_substructure(target, query, match_element))


class _VF2PlusPlusMatcher:
    """
    VF2++ algorithm implementation for subgraph isomorphism.

    VF2++ (Jüttner & Madarasi, 2018) improves on VF2 with:
    1. Optimal matching order computed during preprocessing
    2. Improved candidate selection using node labels and connectivity
    3. Better feasibility rules with stronger lookahead
    """

    def __init__(
        self,
        target: "MolecularGraph",
        query: "MolecularGraph",
        match_element: bool = True,
    ):
        self.target = target
        self.query = query
        self.match_element = match_element

        # Precompute neighbor lists
        self.target_neighbors = target.all_neighbors()
        self.query_neighbors = query.all_neighbors()

        # Precompute degrees
        self.target_degree = np.array([len(n) for n in self.target_neighbors])
        self.query_degree = np.array([len(n) for n in self.query_neighbors])

        # Compute matching order for query nodes (VF2++ preprocessing)
        self.query_order = self._compute_matching_order()

        # Precompute candidate lists for each query node
        self.candidates = self._compute_candidates()

        # Results storage
        self.mappings = []

    def _compute_matching_order(self) -> list[int]:
        """
        Compute optimal matching order for query nodes.

        VF2++ orders nodes by:
        1. Rarity of their label in the target graph
        2. Degree (higher degree = more constraints = better)
        3. Connectivity to already-matched nodes
        """
        n_query = self.query.n_atoms
        if n_query == 0:
            return []

        # Count label frequencies in target
        if self.match_element:
            target_label_count = {}
            for i in range(self.target.n_atoms):
                label = self.target.atomic_numbers[i]
                target_label_count[label] = target_label_count.get(label, 0) + 1
        else:
            target_label_count = {0: self.target.n_atoms}

        # Score each query node: lower frequency = higher priority
        # Also consider degree (higher = better)
        scores = []
        for i in range(n_query):
            if self.match_element:
                label = self.query.atomic_numbers[i]
                freq = target_label_count.get(label, 0)
            else:
                freq = self.target.n_atoms

            degree = self.query_degree[i]
            # Lower frequency and higher degree = better
            # Use negative freq so lower is better, add degree contribution
            score = (freq, -degree, i)
            scores.append((score, i))

        # Sort by score (lower is better)
        scores.sort()

        # Build order using BFS from best-scored node to maintain connectivity
        order = []
        ordered_set = set()

        # Start with the rarest/highest-degree node
        start = scores[0][1]
        queue = [start]

        while len(order) < n_query:
            if queue:
                node = queue.pop(0)
                if node in ordered_set:
                    continue
                order.append(node)
                ordered_set.add(node)

                # Add unordered neighbors to queue, sorted by score
                neighbors = [n for n in self.query_neighbors[node] if n not in ordered_set]
                neighbors.sort(key=lambda x: next(s for s, idx in scores if idx == x))
                queue.extend(neighbors)
            else:
                # Disconnected component - find next unordered node with best score
                for _, node in scores:
                    if node not in ordered_set:
                        queue.append(node)
                        break

        return order

    def _compute_candidates(self) -> list[list[int]]:
        """
        Precompute candidate target nodes for each query node.

        A target node is a candidate if:
        1. It has the same label (if match_element is True)
        2. Its degree >= query node's degree
        """
        candidates = []
        for q in range(self.query.n_atoms):
            q_label = self.query.atomic_numbers[q]
            q_degree = self.query_degree[q]

            cands = []
            for t in range(self.target.n_atoms):
                # Check label
                if self.match_element:
                    if self.target.atomic_numbers[t] != q_label:
                        continue
                # Check degree
                if self.target_degree[t] < q_degree:
                    continue
                cands.append(t)
            candidates.append(cands)

        return candidates

    def find_all(self) -> list[dict[int, int]]:
        """Find all subgraph isomorphisms."""
        self.mappings = []
        if not self.query_order:
            return [{}]
        self._search(0, {}, set())
        return self.mappings

    def find_one(self) -> dict[int, int] | None:
        """Find first subgraph isomorphism (early exit)."""
        self.mappings = []
        if not self.query_order:
            return {}
        try:
            self._search(0, {}, set(), find_one=True)
        except _FoundMatch:
            pass

        if self.mappings:
            return self.mappings[0]
        return None

    def _search(
        self,
        depth: int,
        mapping: dict[int, int],
        target_matched: set[int],
        find_one: bool = False,
    ):
        """Recursive VF2++ search."""
        # Check if complete mapping found
        if depth == len(self.query_order):
            self.mappings.append(mapping.copy())
            if find_one:
                raise _FoundMatch()
            return

        # Get the query node to match at this depth
        query_node = self.query_order[depth]

        # Get candidates for this query node
        candidates = self._get_refined_candidates(
            query_node, mapping, target_matched
        )

        for target_node in candidates:
            # Check feasibility
            if not self._is_feasible(
                query_node, target_node, mapping, target_matched
            ):
                continue

            # Extend mapping
            mapping[query_node] = target_node
            target_matched.add(target_node)

            # Recurse
            self._search(depth + 1, mapping, target_matched, find_one)

            # Backtrack
            del mapping[query_node]
            target_matched.remove(target_node)

    def _get_refined_candidates(
        self,
        query_node: int,
        mapping: dict[int, int],
        target_matched: set[int],
    ) -> list[int]:
        """
        Get refined candidate list for a query node.

        Uses the mapping to filter candidates that are adjacent to
        already-matched target nodes (when the query node is adjacent
        to already-matched query nodes).
        """
        base_candidates = self.candidates[query_node]

        # Filter out already-matched target nodes
        candidates = [c for c in base_candidates if c not in target_matched]

        # If query node has matched neighbors, filter by adjacency
        matched_query_neighbors = [
            n for n in self.query_neighbors[query_node] if n in mapping
        ]

        if matched_query_neighbors:
            # Candidate must be adjacent to all matched neighbors' mappings
            filtered = []
            for cand in candidates:
                valid = True
                for q_neighbor in matched_query_neighbors:
                    t_neighbor = mapping[q_neighbor]
                    if not self.target.has_bond(cand, t_neighbor):
                        valid = False
                        break
                if valid:
                    filtered.append(cand)
            candidates = filtered

        return candidates

    def _is_feasible(
        self,
        query_node: int,
        target_node: int,
        mapping: dict[int, int],
        target_matched: set[int],
    ) -> bool:
        """
        Check if extending mapping with (query_node, target_node) is feasible.

        VF2++ feasibility rules:
        1. Semantic compatibility (label matching)
        2. 1-lookahead: check consistency with current mapping
        3. 2-lookahead: check neighbor counts in/out of mapped regions
        """
        # Rule 1: Semantic compatibility (already checked in candidate filtering)
        # But double-check for safety
        if self.match_element:
            if (
                self.query.atomic_numbers[query_node]
                != self.target.atomic_numbers[target_node]
            ):
                return False

        # Rule 2: Consistency with current mapping
        # All mapped neighbors of query_node must be bonded to target_node
        for q_neighbor in self.query_neighbors[query_node]:
            if q_neighbor in mapping:
                t_neighbor = mapping[q_neighbor]
                if not self.target.has_bond(target_node, t_neighbor):
                    return False

        # Rule 3: Lookahead - count neighbors in different regions
        # Query neighbors in mapping (T1_q)
        q_in_mapped = sum(
            1 for n in self.query_neighbors[query_node] if n in mapping
        )
        # Target neighbors in mapping (T1_t)
        t_in_mapped = sum(
            1 for n in self.target_neighbors[target_node] if n in target_matched
        )

        # Target must have at least as many mapped neighbors
        if t_in_mapped < q_in_mapped:
            return False

        # Query neighbors not in mapping but adjacent to mapping (T2_q)
        q_adj_to_mapped = sum(
            1 for n in self.query_neighbors[query_node]
            if n not in mapping and any(
                nn in mapping for nn in self.query_neighbors[n]
            )
        )
        # Target neighbors not in mapping but adjacent to mapping (T2_t)
        t_adj_to_mapped = sum(
            1 for n in self.target_neighbors[target_node]
            if n not in target_matched and any(
                nn in target_matched for nn in self.target_neighbors[n]
            )
        )

        if t_adj_to_mapped < q_adj_to_mapped:
            return False

        # Query neighbors completely outside mapping (T3_q)
        q_outside = sum(
            1 for n in self.query_neighbors[query_node]
            if n not in mapping and not any(
                nn in mapping for nn in self.query_neighbors[n]
            )
        )
        # Target neighbors completely outside mapping (T3_t)
        t_outside = sum(
            1 for n in self.target_neighbors[target_node]
            if n not in target_matched and not any(
                nn in target_matched for nn in self.target_neighbors[n]
            )
        )

        if t_outside < q_outside:
            return False

        return True


class _FoundMatch(Exception):
    """Exception used for early exit when finding first match."""

    pass


def functional_group_matches(
    target: "MolecularGraph",
    pattern_atoms: list[int],
    pattern_bonds: list[tuple[int, int]],
    match_element: bool = True,
) -> list[dict[int, int]]:
    """
    Find matches of a functional group pattern in a target molecule.

    Convenience function for matching simple patterns specified as atom
    lists and bond lists.

    Args:
        target: The target molecular graph.
        pattern_atoms: List of atomic numbers for pattern atoms.
        pattern_bonds: List of (i, j) tuples for pattern bonds.
        match_element: If True, match atomic numbers.

    Returns:
        List of mappings from pattern indices to target indices.
    """
    from .adjacency import MolecularGraph

    # Create pattern graph
    n = len(pattern_atoms)
    atomic_numbers = np.array(pattern_atoms, dtype=np.int32)
    positions = np.zeros((n, 3))  # Positions don't matter for matching
    pattern = MolecularGraph.from_edge_list(atomic_numbers, positions, pattern_bonds)

    return find_substructure(target, pattern, match_element)


# Predefined functional group patterns
FUNCTIONAL_GROUPS = {
    "hydroxyl": (
        [8, 1],  # O-H
        [(0, 1)],
    ),
    "carbonyl": (
        [6, 8],  # C=O (need bond order for proper matching)
        [(0, 1)],
    ),
    "carboxyl": (
        [6, 8, 8],  # C with two O
        [(0, 1), (0, 2)],
    ),
    "amine": (
        [7, 1],  # N-H
        [(0, 1)],
    ),
    "primary_amine": (
        [7, 1, 1],  # N with 2 H
        [(0, 1), (0, 2)],
    ),
    "nitro": (
        [7, 8, 8],  # N with 2 O
        [(0, 1), (0, 2)],
    ),
}


def find_functional_groups(
    target: "MolecularGraph",
    group_name: str,
) -> list[dict[int, int]]:
    """
    Find instances of a named functional group in a molecule.

    Args:
        target: The target molecular graph.
        group_name: Name of the functional group (e.g., "hydroxyl", "carbonyl").

    Returns:
        List of mappings from pattern indices to target indices.

    Raises:
        KeyError: If group_name is not recognized.
    """
    if group_name not in FUNCTIONAL_GROUPS:
        raise KeyError(f"Unknown functional group: {group_name}")

    atoms, bonds = FUNCTIONAL_GROUPS[group_name]
    return functional_group_matches(target, atoms, bonds)


def list_functional_groups() -> list[str]:
    """List available functional group names."""
    return list(FUNCTIONAL_GROUPS.keys())
