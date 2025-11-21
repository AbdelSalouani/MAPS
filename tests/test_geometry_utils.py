import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from types import SimpleNamespace
from geometry_utils import GeometryUtils


# ---------- Helpers minimaux pour les fonctions qui utilisent la topologie ----------

def make_vertex(vid, pos):
    """Crée un 'vertex' minimal avec .id et .position()"""
    pos_arr = np.array(pos, dtype=np.float64)
    return SimpleNamespace(
        id=vid,
        position=lambda p=pos_arr: p
    )


def make_face(v1, v2, v3):
    """Face minimale avec attributs v1, v2, v3"""
    return SimpleNamespace(v1=v1, v2=v2, v3=v3)


def make_topology(vertices, faces):
    """
    Topologie minimale :
    - vertices : dict[id] -> vertex avec .position()
    - get_star(vid) : renvoie toutes les faces incidentes à vid
    """
    def get_star(vid):
        return [f for f in faces if f.v1 == vid or f.v2 == vid or f.v3 == vid]

    return SimpleNamespace(vertices=vertices, faces=faces, get_star=get_star)


def build_square_patch():
    """
    Carré dans le plan z=0 :
        v0 = (0,0,0)
        v1 = (1,0,0)
        v2 = (1,1,0)
        v3 = (0,1,0)

    Deux triangles : (0,1,2) et (0,2,3)
    On renvoie : topology, center_vertex, neighbors_list
    """
    v0 = make_vertex(0, [0.0, 0.0, 0.0])
    v1 = make_vertex(1, [1.0, 0.0, 0.0])
    v2 = make_vertex(2, [1.0, 1.0, 0.0])
    v3 = make_vertex(3, [0.0, 1.0, 0.0])

    vertices = {0: v0, 1: v1, 2: v2, 3: v3}
    f0 = make_face(0, 1, 2)
    f1 = make_face(0, 2, 3)
    faces = [f0, f1]

    topo = make_topology(vertices, faces)
    center = v0
    neighbors = [v1, v2, v3]

    return topo, center, neighbors


# -------------------- TESTS DES FONCTIONS PURÉMENT GÉOMÉTRIQUES -------------------- #

def test_compute_triangle_area_simple():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])

    area = GeometryUtils.compute_triangle_area(v1, v2, v3)

    assert abs(area - 0.5) < 1e-8


def test_compute_face_normal_planar_up():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])

    n = GeometryUtils.compute_face_normal(v1, v2, v3)

    assert np.allclose(n, np.array([0.0, 0.0, 1.0]), atol=1e-8)


def test_compute_face_normal_degenerate_triangle():
    v1 = np.zeros(3)
    v2 = np.zeros(3)
    v3 = np.zeros(3)

    n = GeometryUtils.compute_face_normal(v1, v2, v3)

    # Cas dégénéré -> normal par défaut [0,0,1]
    assert np.allclose(n, np.array([0.0, 0.0, 1.0]), atol=1e-8)


def test_build_tangent_basis_unit_z():
    normal = np.array([0.0, 0.0, 1.0])
    basis = GeometryUtils._build_tangent_basis(normal)

    assert basis.shape == (3, 3)

    tan_u, tan_v, n = basis[0], basis[1], basis[2]

    # Norme 1
    assert abs(np.linalg.norm(tan_u) - 1.0) < 1e-8
    assert abs(np.linalg.norm(tan_v) - 1.0) < 1e-8
    assert abs(np.linalg.norm(n) - 1.0) < 1e-8

    # Orthogonalité
    assert abs(np.dot(tan_u, tan_v)) < 1e-8
    assert abs(np.dot(tan_u, n)) < 1e-8
    assert abs(np.dot(tan_v, n)) < 1e-8

    # La normale restituée doit être la normale d’entrée
    assert np.allclose(n, normal, atol=1e-8)


def test_compute_dihedral_angle_90_degrees():
    # Deux triangles partageant l'arête (v1,v2) avec un angle de 90°
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])  # plan z=0
    v4 = np.array([0.0, 0.0, 1.0])  # plan y=0

    angle = GeometryUtils.compute_dihedral_angle(v1, v2, v3, v4)

    assert abs(angle - (np.pi / 2.0)) < 1e-6


# -------------------- TESTS DES FONCTIONS AVEC TOPOLOGIE (1-RING) -------------------- #

def test_compute_area_1ring_square_patch():
    topo, center, neighbors = build_square_patch()
    geom = GeometryUtils()

    total_area = geom.compute_area_1ring(center, neighbors, topo)

    # 2 triangles de 0.5 -> aire totale = 1.0
    assert abs(total_area - 1.0) < 1e-8


def test_estimate_vertex_normal_planar_patch():
    topo, center, neighbors = build_square_patch()

    normal = GeometryUtils.estimate_vertex_normal(center, neighbors, topo)

    # Tout est dans le plan z=0 -> normale (0,0,1)
    assert np.allclose(normal, np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_estimate_vertex_normal_empty_star_returns_default():
    # Topologie vide pour ce centre
    center = make_vertex(0, [0.0, 0.0, 0.0])
    topo = make_topology(vertices={}, faces=[])
    neighbors = []

    normal = GeometryUtils.estimate_vertex_normal(center, neighbors, topo)

    assert np.allclose(normal, np.array([0.0, 0.0, 1.0]), atol=1e-8)


def test_estimate_curvature_planar_patch():
    topo, center, neighbors = build_square_patch()
    geom = GeometryUtils()

    curvature = geom.estimate_curvature(center, neighbors, topo)

    # Patch parfaitement plan → courbure très proche de 0
    assert abs(curvature) < 1e-4


def test_estimate_curvature_not_enough_neighbors():
    topo, center, neighbors = build_square_patch()
    geom = GeometryUtils()

    # Moins de 3 voisins -> la fonction doit retourner 0.0
    small_neighbors = neighbors[:2]
    curvature = geom.estimate_curvature(center, small_neighbors, topo)

    assert curvature == 0.0
