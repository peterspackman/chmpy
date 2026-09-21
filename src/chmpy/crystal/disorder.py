"""Describe and resolve the substitutional disorder in a crystal structure.

A disordered crystal is really several structures superimposed: a site that
is only 0.63 occupied is there in 63% of unit cells and something else is
there in the rest.  chmpy models one set of sites, so until the disorder is
resolved a methyl group modelled in two orientations has six hydrogens on
one carbon, and every molecule, surface and energy derived from it is wrong.

The CIF dictionary describes disorder with two labels per site:

``_atom_site_disorder_assembly``
    which independent set of alternatives a site takes part in
``_atom_site_disorder_group``
    which alternative, within that assembly, the site belongs to

A **component** of the structure is every ordered site plus one group from
each assembly, so taking group 1 everywhere gives one ordered structure,
group 2 everywhere gives the next, and so on.

Plenty of files -- most of the ones deposited without those labels -- only
give partial occupancies, so :func:`analyse_disorder` falls back to reading
the occupancies -- values that pair up to 1.0 are two alternatives of one
assembly -- and then to the site labels, where ``O1A`` and ``O1B`` name one
oxygen in two places.  What none of those can work out is reported rather
than guessed at, because a wrong guess here quietly produces a plausible and
incorrect structure.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

LOG = logging.getLogger(__name__)

#: the CIF marker for "this site is not disordered"
ORDERED = "."
#: per site property recording which parent orbit a site came out of
_PARENT_SITE = "_disorder_parent_site"


@dataclass(frozen=True)
class DisorderAssembly:
    """One independent set of alternatives, and the sites in each."""

    name: str
    #: group label -> indices of the asymmetric unit sites in that group
    groups: dict
    #: group label -> the occupancy its sites carry
    occupancies: dict

    @property
    def group_labels(self):
        "group labels, most occupied first, ties broken by label"
        return sorted(self.groups, key=lambda g: (-self.occupancies.get(g, 0.0), g))

    def __repr__(self):
        parts = ", ".join(
            f"{g}:{len(self.groups[g])} sites @ {self.occupancies.get(g, 1.0):.3g}"
            for g in self.group_labels
        )
        return f"<DisorderAssembly {self.name}: {parts}>"


@dataclass(frozen=True)
class Disorder:
    """How the sites of an asymmetric unit are shared between alternatives.

    Attributes:
        assemblies: the independent sets of alternatives
        unresolved: indices of partially occupied sites that could not be
            placed in any assembly
        reason: what is going on with those sites, in words
        source: where the grouping came from, one of ``"disorder_group"``,
            ``"occupancy"``, ``"label"``, ``"occupancy+label"`` or ``"none"``
        n_sites: how many sites the asymmetric unit has
    """

    n_sites: int
    assemblies: tuple = ()
    unresolved: tuple = ()
    source: str = "none"
    occupancies: np.ndarray = field(default=None, repr=False)
    #: why the unresolved sites could not be grouped, if there are any
    reason: str = field(default="", repr=False)

    @property
    def is_disordered(self):
        "whether any site is shared between alternatives"
        return bool(self.assemblies) or bool(self.unresolved)

    @property
    def is_resolvable(self):
        "whether every partially occupied site was placed in an assembly"
        return not self.unresolved

    @property
    def n_components(self):
        """How many ordered structures this disorder describes.

        One, if there is nothing to resolve; otherwise the largest number of
        alternatives any one assembly has.
        """
        if not self.assemblies:
            return 1
        return max(len(a.groups) for a in self.assemblies)

    @property
    def site_groups(self):
        """An ``(n_sites,)`` array naming the alternative each site belongs to.

        Ordered sites are ``"."``; the rest read ``"assembly:group"``.  Handy
        for looking at what was inferred before trusting it.
        """
        labels = np.full(self.n_sites, ORDERED, dtype=object)
        for assembly in self.assemblies:
            for group, sites in assembly.groups.items():
                labels[sites] = f"{assembly.name}:{group}"
        return labels

    def component_mask(self, index):
        """A boolean mask over the asymmetric unit selecting component ``index``.

        Component 0 takes the most occupied alternative of every assembly,
        component 1 the next, and so on.  An assembly with fewer alternatives
        than ``index`` contributes its least occupied one.
        """
        if index < 0 or index >= self.n_components:
            raise IndexError(
                f"component {index} out of range for {self.n_components} components"
            )
        mask = np.ones(self.n_sites, dtype=bool)
        for assembly in self.assemblies:
            labels = assembly.group_labels
            chosen = labels[min(index, len(labels) - 1)]
            for label, sites in assembly.groups.items():
                if label != chosen:
                    mask[sites] = False
        return mask

    def __repr__(self):
        if not self.is_disordered:
            return "<Disorder: ordered>"
        state = "" if self.is_resolvable else f", {len(self.unresolved)} unresolved"
        return (
            f"<Disorder from {self.source}: {len(self.assemblies)} "
            f"assemblies, {self.n_components} components{state}>"
        )


def _labelled_assemblies(assembly_labels, group_labels, occupancies):
    """Assemblies as the file itself labelled them."""
    assemblies = []
    unnamed = 0
    by_assembly = {}
    for site, (assembly, group) in enumerate(
        zip(assembly_labels, group_labels, strict=True)
    ):
        if group == ORDERED:
            continue
        by_assembly.setdefault(assembly, {}).setdefault(group, []).append(site)

    for name, groups in by_assembly.items():
        if name == ORDERED:
            # a file that gives groups but no assemblies: the groups are the
            # only thing distinguishing alternatives, so they stand alone
            name = f"A{unnamed}"
            unnamed += 1
        assemblies.append(
            DisorderAssembly(
                name=str(name),
                groups={g: np.asarray(s) for g, s in groups.items()},
                occupancies={
                    g: float(np.mean(occupancies[s])) for g, s in groups.items()
                },
            )
        )
    return assemblies


def _occupancy_assemblies(occupancies, tolerance):
    """Assemblies inferred from occupancies alone.

    Two occupancies that add up to 1.0 are read as the two alternatives of
    one assembly.  Anything left over is returned as unresolved: three-way
    disorder, alternatives that share an occupancy, and sites whose
    occupancy is reduced by site symmetry rather than by disorder all land
    here, and none of them can be told apart from the occupancies alone.
    """
    partial = np.flatnonzero(occupancies < 1.0 - tolerance)
    if not len(partial):
        return [], ()

    values = sorted({round(float(occupancies[i]), 6) for i in partial}, reverse=True)
    assemblies, paired = [], set()
    for major in values:
        if major in paired:
            continue
        minor = next(
            (
                v
                for v in values
                if v not in paired and v != major and abs(major + v - 1.0) < tolerance
            ),
            None,
        )
        if minor is None:
            continue
        paired.update((major, minor))
        groups = {
            "1": np.flatnonzero(np.abs(occupancies - major) < tolerance),
            "2": np.flatnonzero(np.abs(occupancies - minor) < tolerance),
        }
        assemblies.append(
            DisorderAssembly(
                name=f"A{len(assemblies) + 1}",
                groups=groups,
                occupancies={"1": major, "2": minor},
            )
        )

    placed = {int(i) for a in assemblies for s in a.groups.values() for i in s}
    unresolved = tuple(int(i) for i in partial if int(i) not in placed)
    return assemblies, unresolved


def _label_split(label):
    """A site label split into its stem and the letters that follow it.

    >>> _label_split("O1A")
    ('O1', 'A')
    >>> _label_split("C12")
    ('C12', '')
    """
    i = len(label)
    while i and label[i - 1].isalpha():
        i -= 1
    return (label[:i], label[i:]) if i else (label, "")


def _label_assemblies(labels, occupancies, sites, tolerance):
    """Assemblies inferred from the labels of sites nothing else could place.

    Refinement programs name the alternatives of a disordered group by adding
    a letter, so ``O1A`` and ``O1B`` are one oxygen in two places.  That is
    only a convention and nothing enforces it, so it is the last thing tried
    and only where a stem's alternatives have occupancies adding up to one --
    which is the part the labels cannot be wrong about.
    """
    by_stem = {}
    for site in sites:
        stem, suffix = _label_split(str(labels[site]))
        by_stem.setdefault(stem, {}).setdefault(suffix, []).append(site)

    assemblies, placed = [], set()
    for stem, groups in by_stem.items():
        if len(groups) < 2:
            continue  # a stem with one spelling names no alternatives
        occupancy = {
            suffix: float(np.mean(occupancies[group]))
            for suffix, group in groups.items()
        }
        if abs(sum(occupancy.values()) - 1.0) > tolerance:
            continue
        assemblies.append(
            DisorderAssembly(
                name=stem,
                groups={s: np.asarray(g) for s, g in groups.items()},
                occupancies=occupancy,
            )
        )
        placed.update(int(i) for group in groups.values() for i in group)
    return assemblies, placed


def _diagnose(crystal, unresolved, occupancies):
    """Why some sites could not be grouped, in words.

    The rule the grouping uses -- alternatives whose occupancies add to 1.0 --
    is the special case of a more general one for a site on a general
    position.  What actually has to come out whole is the number of atoms per
    unit cell, which is the occupancy times the site multiplicity, and a site
    sitting on a symmetry element has a multiplicity lower than the general
    one.  Telling the two apart says whether the file is inconsistent or
    whether the disorder simply cannot be resolved in this space group.
    """
    sites = list(unresolved)
    multiplicity = np.bincount(
        crystal.unit_cell_atoms()["asym_atom"], minlength=len(occupancies)
    )
    per_cell = float(np.sum(occupancies[sites] * multiplicity[sites]))
    whole = round(per_cell)
    if whole and abs(per_cell - whole) < 0.05 * whole:
        return (
            f"their occupancies account for {whole} atom(s) per unit cell spread "
            "over symmetry equivalent positions, so this is disorder imposed by "
            "the site symmetry rather than a choice between alternatives. No "
            "selection of these sites is ordered in this space group; resolving "
            "it means descending to a subgroup in which those positions are no "
            "longer equivalent"
        )
    return (
        "their occupancies neither pair up to 1.0 nor account for a whole number "
        f"of atoms per unit cell ({per_cell:.2f}), so what they describe is unclear"
    )


def analyse_disorder(crystal, tolerance=1e-3) -> Disorder:
    """Work out how the sites of ``crystal`` are shared between alternatives.

    The CIF's own ``_atom_site_disorder_group`` labels are used when the file
    carries them.  Otherwise the occupancies are read, and then the site
    labels for whatever is left -- which is what resolves the common case of
    two alternatives at 0.5 each, where the occupancies say nothing.  See the
    module docstring for what each can and cannot say.

    Args:
        crystal: the crystal structure to examine
        tolerance: how far an occupancy may stray from a whole number, or
            from summing to 1.0 with its partner, and still count

    Returns:
        Disorder: the grouping, and what could not be grouped
    """
    asym = crystal.asymmetric_unit
    n_sites = len(asym)
    occupancies = np.asarray(
        asym.properties.get("occupation", np.ones(n_sites)), dtype=float
    )

    cif_data = crystal.properties.get("cif_data") or {}
    group_labels = cif_data.get("atom_site_disorder_group")
    if group_labels is not None and len(group_labels) == n_sites:
        labels = [str(g) for g in group_labels]
        if any(g != ORDERED for g in labels):
            assembly_labels = cif_data.get("atom_site_disorder_assembly")
            if assembly_labels is None or len(assembly_labels) != n_sites:
                assembly_labels = [ORDERED] * n_sites
            return Disorder(
                n_sites=n_sites,
                assemblies=tuple(
                    _labelled_assemblies(
                        [str(a) for a in assembly_labels], labels, occupancies
                    )
                ),
                source="disorder_group",
                occupancies=occupancies,
            )

    assemblies, unresolved = _occupancy_assemblies(occupancies, tolerance)
    source = "occupancy" if assemblies else "none"
    if unresolved:
        source = source if assemblies else "none"
        by_label, placed = _label_assemblies(
            asym.labels, occupancies, unresolved, tolerance
        )
        if by_label:
            assemblies = assemblies + by_label
            unresolved = tuple(i for i in unresolved if i not in placed)
            source = "occupancy+label" if source == "occupancy" else "label"
    if unresolved and source == "none":
        source = "occupancy"
    return Disorder(
        n_sites=n_sites,
        assemblies=tuple(assemblies),
        unresolved=unresolved,
        source=source,
        occupancies=occupancies,
        reason=_diagnose(crystal, unresolved, occupancies) if unresolved else "",
    )


#: how close two sites have to be before they cannot both hold an atom
CLASH_DISTANCE = 1.2


def _straddled_operations(crystal, sites, clash_distance):
    """Which symmetry operations map one alternative onto another.

    No atom sits within a bond length of itself, so a partially occupied site
    whose image under some operation lands that close to a partially occupied
    site is not one atom in one place: the operation is relating two
    alternatives.  While it remains in the group no selection of sites is
    ordered, because choosing one alternative means choosing its image too.
    """
    positions = crystal.asymmetric_unit.positions[list(sites)]
    straddled = []
    for index, operation in enumerate(crystal.space_group.symmetry_operations):
        if operation.is_identity():
            continue
        difference = operation.apply(positions)[:, None, :] - positions[None, :, :]
        difference -= np.round(difference)  # nearest lattice image
        distance = np.linalg.norm(
            crystal.to_cartesian(difference.reshape(-1, 3)), axis=1
        )
        if np.any((distance > 1e-3) & (distance < clash_distance)):
            straddled.append(index)
    return straddled


def _subgroup_space_group(operations, result):
    """A SpaceGroup carrying exactly ``operations``.

    The standard setting of the subgroup's type is often a different origin
    or axis choice, so it cannot simply be looked up by number.
    """
    from .space_group import SpaceGroup

    try:
        return SpaceGroup.from_symmetry_operations(list(operations))
    except (ValueError, KeyError):
        pass
    space_group = SpaceGroup(1)
    space_group.symmetry_operations = list(operations)
    if result.space_group_number:
        space_group.international_tables_number = result.space_group_number
    symbol = result.space_group_symbol or space_group.symbol
    space_group.symbol = symbol
    space_group.full_symbol = symbol
    LOG.debug("subgroup %s is a non-standard setting", symbol)
    return space_group


def _overbonded_hydrogens(crystal, bond_tolerance=0.4):
    """Hydrogens with more than one neighbour close enough to be bonded.

    Hydrogen forms one bond.  Two of them a bond length apart is the clearest
    sign that a set of sites chosen as an ordered structure is not one: the
    alternatives were never a discrete arrangement, so no choice among them
    produces a real structure.
    """
    from scipy.spatial import KDTree

    from chmpy.core.element import Element

    atoms = crystal.unit_cell_atoms()
    fractional, cartesian = atoms["frac_pos"], atoms["cart_pos"]
    numbers = atoms["element"]
    hydrogens = np.flatnonzero(numbers == 1)
    if not len(hydrogens):
        return 0

    radii = np.fromiter(
        (Element[int(z)].covalent_radius for z in numbers),
        dtype=float,
        count=len(numbers),
    )
    shifts = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    )
    images = crystal.to_cartesian(
        (fractional[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    )
    image_radii = np.tile(radii, len(shifts))
    reach = Element["H"].covalent_radius + radii.max() + bond_tolerance
    tree = KDTree(images)

    overbonded = 0
    for h in hydrogens:
        neighbours = tree.query_ball_point(cartesian[h], reach)
        bonded = 0
        for n in neighbours:
            distance = np.linalg.norm(images[n] - cartesian[h])
            if distance < 1e-6:
                continue  # the atom itself
            if distance < radii[h] + image_radii[n] + bond_tolerance:
                bonded += 1
        if bonded > 1:
            overbonded += 1
    return overbonded


def descend_for_disorder(
    crystal, disorder=None, max_index=4, clash_distance=CLASH_DISTANCE
):
    """Lower the symmetry until the disorder is a choice rather than a fact.

    Where alternatives are related by a symmetry operation -- a group
    straddling the mirror it sits on, say -- no selection of sites is ordered
    in the parent group, because picking one alternative picks its image too.
    Dropping that operation splits the orbit, and the alternatives become
    separate sites that can be chosen between.

    The descent is to a translationengleiche subgroup, so the cell is
    unchanged and only the point operations are reduced.  Giving up a factor
    of ``n`` in symmetry multiplies the size of the asymmetric unit by ``n``:
    Ima2 with 8 operations and 6 sites becomes, at index 2, one of Pna2_1,
    Pnc2, Pma2, Pmc2_1, Pm, Cc or C2 with 4 operations and 11 sites.

    Args:
        crystal: the structure to lower
        disorder: a :class:`Disorder` to use instead of analysing the crystal
        max_index: how much symmetry may be given up.  The index of a
            subgroup is ``|G|/|H|``, how many times smaller it is than the
            parent; since the lattice is unchanged here that is just the
            ratio of the number of symmetry operations, and the asymmetric
            unit grows by the same factor (as does Z').  So ``max_index=4``
            allows dropping down to a quarter of the operations -- for a
            group of 8, a subgroup of 4 or 2.  The smallest index that works
            is used, so as little symmetry as possible is given up, and only
            indices dividing the order of the group exist at all.
        clash_distance: how close two sites must be to be alternatives

    Returns:
        Crystal: the same structure in a subgroup, its asymmetric unit
        expanded, or None if no subgroup within ``max_index`` helps

    Note:
        The subgroup is one of several symmetry equivalent choices, so the
        structures that come out are one ordering of many rather than the
        ordering.  Disorder that is genuinely smeared -- alternatives that do
        not correspond to any discrete arrangement -- is not made resolvable
        by this, and is still reported as unresolved afterwards.
    """
    from .asymmetric_unit import AsymmetricUnit
    from .crystal import Crystal
    from .space_group_table import SpaceGroupTable
    from .subgroup import SubgroupEnumerator, expand_asymmetric_unit

    if disorder is None:
        disorder = analyse_disorder(crystal)
    if disorder.is_resolvable:
        return None

    straddled = set(_straddled_operations(crystal, disorder.unresolved, clash_distance))
    if not straddled:
        LOG.debug("no symmetry operation relates the unresolved sites")
        return None

    space_group = crystal.space_group
    symops = space_group.symmetry_operations
    enumerator = SubgroupEnumerator.from_space_group(space_group)
    chosen = None
    for index in range(2, max_index + 1):
        candidates = [
            result
            for result in enumerator.find_by_index(index)
            if straddled.isdisjoint(result.symop_indices)
        ]
        if candidates:
            # prefer one whose operations are a standard setting, so the
            # result is a fully described space group rather than a bare list
            candidates.sort(key=lambda r: (r.space_group_number is None,))
            chosen = candidates[0]
            break
    if chosen is None:
        LOG.debug("no subgroup up to index %d drops the operations", max_index)
        return None

    asym = crystal.asymmetric_unit
    # carry the parent site index along, so the expansion says which sites
    # came out of which orbit and the alternatives need not be guessed again
    tagged = AsymmetricUnit(
        asym.elements,
        asym.positions,
        labels=asym.labels,
        **{**asym.properties, _PARENT_SITE: np.arange(len(asym))},
    )
    expanded = expand_asymmetric_unit(
        tagged,
        symops,
        chosen.symop_indices,
        SpaceGroupTable.from_space_group(space_group),
    )
    new_space_group = _subgroup_space_group(
        [symops[i] for i in chosen.symop_indices], chosen
    )
    properties = dict(crystal.properties)
    properties.pop("unit_cell_atoms", None)
    properties["descended_from"] = space_group.symbol
    LOG.debug(
        "descended %s -> %s (index %d), %d -> %d sites",
        space_group.symbol,
        new_space_group.symbol,
        chosen.index,
        len(asym),
        len(expanded),
    )
    return Crystal(crystal.unit_cell, new_space_group, expanded, **properties)


def _disorder_from_split_orbits(crystal, tolerance):
    """The disorder of a crystal just produced by :func:`descend_for_disorder`.

    The descent already knows which sites came out of which parent orbit, so
    the alternatives are recorded rather than inferred a second time.
    """
    asym = crystal.asymmetric_unit
    parents = asym.properties.get(_PARENT_SITE)
    if parents is None:
        return None
    occupancies = np.asarray(
        asym.properties.get("occupation", np.ones(len(asym))), dtype=float
    )
    by_parent = {}
    for site, parent in enumerate(np.asarray(parents)):
        by_parent.setdefault(int(parent), []).append(site)

    assemblies = []
    for sites in by_parent.values():
        if len(sites) < 2 or occupancies[sites[0]] >= 1.0 - tolerance:
            continue  # a split orbit of a full site is Z' doubling, not disorder
        assemblies.append(
            DisorderAssembly(
                name=f"{asym.labels[sites[0]]}",
                groups={str(k + 1): np.asarray([site]) for k, site in enumerate(sites)},
                occupancies={
                    str(k + 1): float(occupancies[site]) for k, site in enumerate(sites)
                },
            )
        )
    placed = {i for a in assemblies for s in a.groups.values() for i in s}
    unresolved = tuple(
        int(i)
        for i in np.flatnonzero(occupancies < 1.0 - tolerance)
        if int(i) not in placed
    )
    return Disorder(
        n_sites=len(asym),
        assemblies=tuple(assemblies),
        unresolved=unresolved,
        source="subgroup",
        occupancies=occupancies,
        reason=_diagnose(crystal, unresolved, occupancies) if unresolved else "",
    )


def disorder_components(
    crystal, disorder=None, strict=True, descend=False, max_index=4
) -> list:
    """Split a disordered crystal into ordered ones.

    Each returned crystal keeps every fully occupied site and one alternative
    from each disorder assembly, with the surviving occupancies set to 1.0.
    An ordered crystal comes back as a single-item list, so callers need no
    special case for it.

    Args:
        crystal: the structure to resolve
        disorder: a :class:`Disorder` to use instead of analysing the crystal,
            for when you have worked the grouping out yourself
        strict: raise if some partially occupied site could not be placed in
            an assembly.  Pass ``False`` to build the components anyway, which
            keeps those sites in every one of them.
        descend: where alternatives are related by symmetry, lower the
            symmetry to a subgroup in which they are not, and resolve there.
            The components then have the subgroup's symmetry, and are one of
            several symmetry equivalent orderings -- see
            :func:`descend_for_disorder`.
        max_index: how much symmetry ``descend`` may give up, as the index
            ``|G|/|H|`` of the subgroup it may drop to -- see
            :func:`descend_for_disorder`

    Returns:
        list: one ordered :class:`~chmpy.crystal.crystal.Crystal` per component,
        most occupied alternatives first
    """
    if disorder is None:
        disorder = analyse_disorder(crystal)
    if descend and not disorder.is_resolvable:
        lowered = descend_for_disorder(crystal, disorder, max_index=max_index)
        if lowered is not None:
            components = disorder_components(
                lowered, _disorder_from_split_orbits(lowered, 1e-3), strict=strict
            )
            broken = max(_overbonded_hydrogens(c) for c in components)
            if broken and strict:
                raise ValueError(
                    f"lowering the symmetry of {crystal.titl} to "
                    f"{lowered.space_group.symbol} did not resolve its disorder: "
                    f"{broken} hydrogen(s) in the result have more than one bonded "
                    "neighbour, so the alternatives are a smeared model rather "
                    "than a discrete set of arrangements and no choice among "
                    "them is a real structure. Use strict=False to get them anyway."
                )
            if broken:
                LOG.warning(
                    "%d hydrogen(s) in the descended components are overbonded; "
                    "the disorder is not a discrete set of arrangements",
                    broken,
                )
            return components
    if strict and not disorder.is_resolvable:
        labels = [str(crystal.asymmetric_unit.labels[i]) for i in disorder.unresolved]
        raise ValueError(
            f"cannot resolve the disorder in {crystal.titl}: "
            f"{len(labels)} partially occupied site(s) ({', '.join(labels[:6])}"
            f"{', ...' if len(labels) > 6 else ''}) could not be grouped into "
            f"alternatives -- {disorder.reason}. Pass a Disorder built by hand, "
            "or strict=False to keep those sites in every component."
        )
    if not disorder.is_disordered:
        return [crystal]
    return [
        _component(crystal, disorder.component_mask(i), i)
        for i in range(disorder.n_components)
    ]


def _component(crystal, mask, index):
    """One ordered crystal, from the sites ``mask`` selects."""
    from .asymmetric_unit import AsymmetricUnit
    from .crystal import Crystal

    asym = crystal.asymmetric_unit
    keep = np.flatnonzero(mask)
    properties = {}
    for name, value in asym.properties.items():
        if name == _PARENT_SITE:
            continue  # bookkeeping for the descent, not part of the structure
        value = np.asarray(value)
        properties[name] = value[keep] if len(value) == len(asym) else value
    # the sites that remain are all there is, so they are fully occupied
    properties["occupation"] = np.ones(len(keep))

    component = AsymmetricUnit(
        elements=[asym.elements[i] for i in keep],
        positions=asym.positions[keep],
        labels=asym.labels[keep],
        **properties,
    )
    crystal_properties = dict(crystal.properties)
    crystal_properties.pop("unit_cell_atoms", None)
    crystal_properties["disorder_component"] = index
    titl = crystal_properties.get("titl")
    if titl:
        crystal_properties["titl"] = f"{titl}_{index + 1}"
    return Crystal(
        crystal.unit_cell, crystal.space_group, component, **crystal_properties
    )
