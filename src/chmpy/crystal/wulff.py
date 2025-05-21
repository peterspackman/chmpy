import numpy as np
from scipy.spatial import ConvexHull


def expand_symmetry_related_planes(hkl, energies, crystal):
    hkl.shape[0]
    sg = crystal.sg
    len(sg)
    expanded_facets = []

    R = crystal.uc.reciprocal_lattice
    RI = crystal.uc.direct.T
    frac = crystal.to_fractional(hkl @ R)
    print(frac)
    print(np.round(crystal.to_cartesian(frac) @ RI, 2).astype(int))

    for s in sg.symmetry_operations:
        if len(np.nonzero(s.translation)[0]) > 0:
            continue
        print("Symop", s)
        rot = s.rotation.T
        print(rot)
        transformed = crystal.to_cartesian(frac @ rot) @ RI
        print(transformed)
        transformed = np.round(transformed, 2).astype(int)
        print(transformed)
        expanded_facets.append(transformed)

    expanded_facets = np.vstack(expanded_facets)
    tiled_energies = np.tile(energies, expanded_facets.shape[0] // hkl.shape[0])

    # identify the unique directions
    # we may not need to include the inversion as it should probably
    # be provided...
    unique = {}
    for i, x in enumerate(expanded_facets):
        reduced = tuple(x / np.gcd.reduce(x).max())
        if reduced not in unique or tiled_energies[unique[reduced]] > tiled_energies[i]:
            unique[reduced] = i

    unique_facets = []
    unique_energies = []
    for k, v in unique.items():
        unique_facets.append(k)
        unique_energies.append(tiled_energies[v])
    return np.array(unique_facets), np.array(unique_energies)


def project_to_plane(points, plane_normal):
    projected_points = points - np.outer(np.dot(points, plane_normal), plane_normal)

    a_vector = projected_points[1] - projected_points[0]
    b_vector = np.cross(plane_normal, a_vector)

    u = np.dot(projected_points, a_vector)
    v = np.dot(projected_points, b_vector)

    return np.column_stack((u, v))


def winding_order_ccw(points):
    "return the indices to reorder the provided 2D points into CCW order"
    centroid = points[0]
    directions = points[1:] - centroid
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    idxs = list(range(1, points.shape[0]))
    return [0] + sorted(
        idxs, key=lambda x: np.arctan2(directions[x - 1, 1], directions[x - 1, 0])
    )


def prune_degenerate_points(points, threshold=1e-5):
    dist_sq = np.sum((points[:, np.newaxis] - points) ** 2, axis=2)
    dist_sq[np.tril_indices_from(dist_sq)] = 1
    keep_mask = np.all(dist_sq >= threshold**2, axis=1)
    idxs = np.nonzero(keep_mask)[0]
    return points[keep_mask], idxs


def ordered_facets(points, facets, facet_normals):
    result = []
    for i, facet in enumerate(facets):
        if len(facet) == 0:
            result.append([])
            continue
        pts, idxs = prune_degenerate_points(points[facet])
        points_2d = project_to_plane(pts, facet_normals[i])
        ccw_order = winding_order_ccw(points_2d)
        result.append([facet[idxs[x]] for x in ccw_order])
    return result


def order_and_triangulate_polygons(points, facets, facet_normals):
    # probably the bottleneck
    ordered = ordered_facets(points, facets, facet_normals)

    triangles = []
    facet_indices = []
    for i, facet in enumerate(ordered):
        N = len(facet)
        if N == 0:
            continue
        t = np.column_stack((np.repeat(facet[0], N - 2), facet[1 : N - 1], facet[2:N]))
        triangles.append(t)
        facet_indices += [i] * t.shape[0]

    return ordered, np.vstack(triangles), facet_indices


def perpendicular_vector(p, q, r):
    "a unit vector perpendicular to the triangle p q r"
    perp_vector = np.cross(q - p, r - p)
    dp = np.dot(perp_vector, p)
    if dp > 0:
        return perp_vector / np.linalg.norm(perp_vector)
    else:
        return -perp_vector / np.linalg.norm(perp_vector)


class WulffSHT:
    def __init__(
        self,
        facet_normals,
        facet_energies,
        labels=None,
        l_max=10,
        sht_object=None,
        scale=1,
    ):
        from chmpy.shape.sht import SHT

        self.facet_normals = facet_normals
        self.facet_energies = facet_energies
        self.labels = labels
        self.l_max = l_max
        if self.labels is None:
            self.labels = np.arange(len(facet_normals))

        if sht_object is None:
            sht = SHT(l_max)
        else:
            sht = sht_object

        self.grid = np.array(sht.grid_cartesian)
        dps = np.einsum("ijk,li->jkl", self.grid, self.facet_normals)
        mask = dps > 0
        intersections = np.inf * np.ones_like(dps)
        intersections[mask] = 1.0 / dps[mask]
        intersections *= scale * self.facet_energies

        facet_idx = np.argmin(intersections, axis=2, keepdims=True)
        self.distances = np.take_along_axis(intersections, facet_idx, axis=2).squeeze()
        self.facet_idx = facet_idx.squeeze()
        self.coeffs = sht.analysis(self.distances)

    def invariants(self, **kwargs):
        from chmpy.shape.shape_descriptors import expand_coeffs_to_full, make_invariants

        coeff4inv = expand_coeffs_to_full(self.l_max, self.coeffs)
        invariants = make_invariants(
            self.l_max, coeff4inv, kinds=kwargs.get("kinds", "NP")
        )
        return invariants

    def invariants_kazhdan(self, sht_object=None, **kwargs):
        from chmpy.shape.sht import SHT

        if sht_object is None:
            sht = SHT(self.l_max)
        else:
            sht = sht_object
        return sht.invariants_pure_python(self.coeffs)

    def power_spectrum(self, sht_object=None, **kwargs):
        from chmpy.shape.sht import SHT

        if sht_object is None:
            sht = SHT(self.l_max)
        else:
            sht = sht_object
        return sht.power_spectrum(self.coeffs)

    @classmethod
    def from_gmf_and_crystal(cls, gmf, crystal, **kwargs):
        hkl, energies = expand_symmetry_related_planes(
            gmf.hkl, gmf.energies, crystal.space_group
        )
        facet_normals = hkl @ crystal.uc.reciprocal_lattice
        facet_normals /= np.linalg.norm(facet_normals, axis=1)[:, None]
        return cls(facet_normals, energies, labels=energies, **kwargs)

    def __repr__(self):
        return f"WulffSHT<l={self.l_max}>"


class WulffConstruction:
    def __init__(self, facet_normals, facet_energies, labels=None):
        self.facet_normals = np.array(facet_normals)
        self.facet_energies = np.array(facet_energies)
        self.labels = labels
        if self.labels is None:
            self.labels = np.arange(len(facet_normals))

        self._populate_duals()
        self._construct_dual_space_hull()
        self._extract_wulff_from_dual_mesh()
        self._fix_wulff_mesh()

    def _populate_duals(self):
        self.facet_vectors = self.facet_normals * self.facet_energies[:, np.newaxis]
        self.facet_dual_vectors = (
            self.facet_vectors / (self.facet_vectors**2).sum(axis=1)[:, np.newaxis]
        )

    def _construct_dual_space_hull(self):
        self.dual_hull = ConvexHull(self.facet_dual_vectors)

    def _extract_wulff_from_dual_mesh(self):
        # Get the simplices and reshape them for indexing
        simplices = self.dual_hull.simplices
        self.facet_dual_vectors[simplices]
        a, b, c = np.rollaxis(self.facet_dual_vectors[simplices], 1)

        normals = np.cross(b - a, c - a)

        # Get the corresponding facet indices and normals
        facet_indices = simplices[:, 0]  # Take first index from each simplex

        corresponding_facet_normals = self.facet_normals[facet_indices]
        corresponding_facet_energies = self.facet_energies[facet_indices]
        inv_factors = np.einsum("ij,ij->i", normals, corresponding_facet_normals)
        scaling_factors = corresponding_facet_energies / inv_factors
        vertices = normals * scaling_factors[:, np.newaxis]

        num_dual_vectors = len(self.facet_dual_vectors)
        facets = [[] for _ in range(num_dual_vectors)]

        for idx, facet in enumerate(simplices):
            for f in facet:
                facets[f].append(idx)

        self.wulff_vertices = vertices
        self.wulff_facets = facets

    def triangle_labels(self):
        return self.labels[self.wulff_triangle_indices]

    def _fix_wulff_mesh(self):
        result = order_and_triangulate_polygons(
            self.wulff_vertices, self.wulff_facets, self.facet_normals
        )
        self.wulff_facets = result[0]
        self.wulff_triangles = np.array(result[1])
        self.wulff_triangle_indices = np.array(result[2])

    def to_trimesh(self):
        from trimesh import Trimesh

        return Trimesh(self.wulff_vertices, self.wulff_triangles)

    def dual_trimesh(self):
        from trimesh import Trimesh

        return Trimesh(self.facet_dual_vectors, self.dual_hull.simplices)

    def sht(self, **kwargs):
        return WulffSHT(
            self.facet_normals, self.facet_energies, labels=self.labels, **kwargs
        )

    @classmethod
    def from_gmf_and_crystal(cls, gmf, crystal):
        hkl, energies = expand_symmetry_related_planes(
            gmf.hkl, gmf.energies, crystal.space_group
        )
        facet_normals = hkl @ crystal.uc.reciprocal_lattice
        facet_normals /= np.linalg.norm(facet_normals, axis=1)[:, None]

        return cls(facet_normals, energies, labels=energies)

    def __repr__(self):
        return (
            f"WulffConstruction<verts={len(self.wulff_vertices)},"
            f"facets={len(self.wulff_facets)}>"
        )
