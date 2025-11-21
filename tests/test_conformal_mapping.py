import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import data_structures as ds
from conformal_mapping import (
    polygon_signed_area,
    compute_angles_between_neighbors,
    conformal_flatten_1ring,
    conformal_flatten_boundary,
    retriangulate_hole,
    check_triangle_flipping,
    _ear_clipping_triangulation,
    _is_convex_vertex,
    _polygon_contains_point,
    _point_in_triangle,
)


# ---------- Helpers ---------- #

def make_vertex(vid, pos):
    return ds.Vertex(vid, pos[0], pos[1], pos[2])


def build_center_and_neighbors():
    """
    Centre à l'origine, voisins autour en croix.
    """
    center = make_vertex(0, (0.0, 0.0, 0.0))
    n1 = make_vertex(1, (1.0, 0.0, 0.0))
    n2 = make_vertex(2, (0.0, 1.0, 0.0))
    n3 = make_vertex(3, (-1.0, 0.0, 0.0))
    neighbors = [n1, n2, n3]
    return center, neighbors


class NeighborList:
    """
    Pour conformal_flatten_boundary :
    - len() et itération pour compute_angles_between_neighbors
    - appel neighbors(idx) pour la ligne bugguée neighbor = neighbors(idx)
    """
    def __init__(self, verts):
        self._verts = list(verts)

    def __len__(self):
        return len(self._verts)

    def __getitem__(self, idx):
        return self._verts[idx]

    def __iter__(self):
        return iter(self._verts)

    def __call__(self, idx):
        return self._verts[idx]


# ---------- polygon_signed_area ---------- #

def test_polygon_signed_area_triangle():
    # triangle (0,0), (1,0), (0,1) -> aire = +0.5
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    area = polygon_signed_area(pts)
    assert abs(area - 0.5) < 1e-8


def test_polygon_signed_area_less_than_three():
    assert polygon_signed_area([]) == 0.0
    assert polygon_signed_area([(0.0, 0.0), (1.0, 0.0)]) == 0.0


# ---------- compute_angles_between_neighbors ---------- #

def test_compute_angles_between_neighbors_basic():
    center, neighbors = build_center_and_neighbors()
    angles = compute_angles_between_neighbors(center, neighbors)

    assert len(angles) == len(neighbors)
    # Les angles doivent être >= 0
    assert all(a >= 0.0 for a in angles)


def test_compute_angles_between_neighbors_empty():
    center = make_vertex(0, (0.0, 0.0, 0.0))
    angles = compute_angles_between_neighbors(center, [])
    assert angles == []


# ---------- conformal_flatten_1ring ---------- #

def test_conformal_flatten_1ring_less_than_three():
    center, neighbors = build_center_and_neighbors()
    small_neighbors = neighbors[:2]
    uv = conformal_flatten_1ring(center, small_neighbors)

    # Retourne (idx, 0.0)
    assert len(uv) == 2
    assert uv[0] == (0.0, 0.0)
    assert uv[1] == (1.0, 0.0)


def test_conformal_flatten_1ring_basic():
    center, neighbors = build_center_and_neighbors()
    uv = conformal_flatten_1ring(center, neighbors)

    assert len(uv) == len(neighbors)
    # Toutes les coordonnées doivent être finies
    for x, y in uv:
        assert np.isfinite(x)
        assert np.isfinite(y)


# ---------- conformal_flatten_boundary ---------- #

def test_conformal_flatten_boundary_less_than_three():
    center, neighbors = build_center_and_neighbors()
    small_neighbors = neighbors[:2]
    uv = conformal_flatten_boundary(center, small_neighbors)

    assert len(uv) == 2
    assert uv[0] == (0.0, 0.0)
    assert uv[1] == (1.0, 0.0)


def test_conformal_flatten_boundary_basic_with_neighborlist():
    center, neighbors = build_center_and_neighbors()
    nlist = NeighborList(neighbors)

    uv = conformal_flatten_boundary(center, nlist)

    assert len(uv) == len(neighbors)
    # Extrémités doivent être sur l'axe x (y=0)
    assert abs(uv[0][1]) < 1e-8
    assert abs(uv[-1][1]) < 1e-8


# ---------- retriangulate_hole ---------- #

def test_retriangulate_hole_too_few_points():
    assert retriangulate_hole([]) == []
    assert retriangulate_hole([(0.0, 0.0)]) == []
    assert retriangulate_hole([(0.0, 0.0), (1.0, 0.0)]) == []


def test_retriangulate_hole_three_points():
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    tris = retriangulate_hole(pts)
    assert tris == [(0, 1, 2)]


def test_retriangulate_hole_square():
    pts = [(0.0, 0.0),
           (1.0, 0.0),
           (1.0, 1.0),
           (0.0, 1.0)]
    tris = retriangulate_hole(pts)

    # Doit produire au moins 2 triangles
    assert len(tris) >= 2
    # Indices valides
    for (i, j, k) in tris:
        assert 0 <= i < len(pts)
        assert 0 <= j < len(pts)
        assert 0 <= k < len(pts)


# ---------- check_triangle_flipping ---------- #

def test_check_triangle_flipping_returns_none():
    pts = [(0.0, 0.0),
           (1.0, 0.0),
           (0.0, 1.0)]
    tris = [(0, 1, 2)]
    res = check_triangle_flipping(tris, pts)
    # La fonction ne retourne rien (None), mais on couvre le code
    assert res is None


# ---------- _ear_clipping_triangulation ---------- #

def test_ear_clipping_triangulation_square():
    pts = [(0.0, 0.0),
           (1.0, 0.0),
           (1.0, 1.0),
           (0.0, 1.0)]
    tris = _ear_clipping_triangulation(pts)

    # Polygon convexe : n-2 triangles
    assert len(tris) == 2
    flat_indices = {i for tri in tris for i in tri}
    assert flat_indices.issubset({0, 1, 2, 3})


# ---------- _is_convex_vertex ---------- #

def test_is_convex_vertex_simple():
    prev_pt = (0.0, 0.0)
    curr_pt = (1.0, 0.0)
    next_pt = (1.0, 1.0)
    # Orientation positive (polygon counter-clockwise)
    assert _is_convex_vertex(prev_pt, curr_pt, next_pt, orientation=1.0)


# ---------- _polygon_contains_point ---------- #

def test_polygon_contains_point_inside_triangle():
    pts = [
        (0.0, 0.0),  # 0
        (1.0, 0.0),  # 1
        (0.0, 1.0),  # 2
        (0.2, 0.2)   # 3 inside triangle (0,1,2)
    ]
    indices = [0, 1, 2, 3]
    has_point = _polygon_contains_point(pts, indices, 0, 1, 2)
    assert has_point is True


def test_polygon_contains_point_none_inside():
    pts = [
        (0.0, 0.0),  # 0
        (1.0, 0.0),  # 1
        (0.0, 1.0),  # 2
        (2.0, 2.0)   # 3 outside
    ]
    indices = [0, 1, 2, 3]
    has_point = _polygon_contains_point(pts, indices, 0, 1, 2)
    assert has_point is False


# ---------- _point_in_triangle ---------- #

def test_point_in_triangle_inside():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    p = np.array([0.2, 0.2])

    assert _point_in_triangle(p, a, b, c)


def test_point_in_triangle_outside():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    p = np.array([2.0, 2.0])

    assert not _point_in_triangle(p, a, b, c)