import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from data_structures import (
    Vertex,
    Face,
    BarycentricCoordinates,
    MeshLevel,
    MeshHierarchy,
    obj2mesh
)


# ------------------------- TESTS VERTEX ------------------------- #

def test_vertex_position_and_distance():
    v1 = Vertex(0, 0.0, 0.0, 0.0)
    v2 = Vertex(1, 1.0, 0.0, 0.0)

    assert np.allclose(v1.position(), np.array([0, 0, 0]))
    assert np.allclose(v2.position(), np.array([1, 0, 0]))

    dist = v1.distance_to(v2)
    assert abs(dist - 1.0) < 1e-8


# ------------------------- TESTS FACE ------------------------- #

def test_face_vertices_and_contains():
    f = Face(1, 2, 3)

    assert f.vertices() == [1, 2, 3]
    assert f.contains_vertex(1)
    assert f.contains_vertex(3)
    assert not f.contains_vertex(4)

def test_face_clone():
    f = Face(1, 2, 3)
    c = f.clone()

    assert isinstance(c, Face)
    assert c.v1 == 1 and c.v2 == 2 and c.v3 == 3
    assert c is not f  # clone must be a different object


# ------------------------- TEST BARYCENTRIC ------------------------- #

def test_barycentric_valid():
    b = BarycentricCoordinates()
    b.a, b.b, b.g = 0.2, 0.3, 0.5
    assert b.is_valid()

def test_barycentric_invalid():
    b = BarycentricCoordinates()
    b.a, b.b, b.g = 1.5, -0.1, 0.0
    assert not b.is_valid()


# ------------------------- TESTS MESH LEVEL ------------------------- #

def test_mesh_level_add_and_count():
    ml = MeshLevel(0)

    v0 = Vertex(0, 0, 0, 0)
    v1 = Vertex(1, 1, 0, 0)

    ml.add_vertex(v0)
    ml.add_vertex(v1)

    assert ml.num_vertices() == 2
    assert ml.get_vertex(1) is v1

    f = Face(0, 1, 0)
    ml.add_face(f)

    assert ml.num_faces() == 1
    assert ml.get_active_faces() == [f]

def test_mesh_level_removed_vertices_list_exists():
    ml = MeshLevel(0)
    assert isinstance(ml.removed_vertices, list)


# ------------------------- TESTS MESH HIERARCHY ------------------------- #

def test_mesh_hierarchy_add_and_get_levels():
    h = MeshHierarchy()

    # build 2 levels
    ml2 = MeshLevel(2)
    ml1 = MeshLevel(1)

    h.add_level(ml2)
    h.add_level(ml1)

    assert h.finest_level == 2
    assert h.coarsest_level == 1
    assert h.num_levels() == 2

    # get level by logical ID
    assert h.get_level(2) is ml2
    assert h.get_level(1) is ml1

def test_mesh_hierarchy_compression_ratio():
    h = MeshHierarchy()

    ml2 = MeshLevel(2)
    ml1 = MeshLevel(1)

    # 10 vertices at fine level
    for i in range(10):
        ml2.add_vertex(Vertex(i, 0, 0, 0))

    # 2 vertices at coarse level
    for i in range(2):
        ml1.add_vertex(Vertex(i, 0, 0, 0))

    h.add_level(ml2)
    h.add_level(ml1)

    ratio = h.compression_ratio()
    assert abs(ratio - (10/2)) < 1e-8


# ------------------------- TEST obj2mesh (mocking parse_file) ------------------------- #

class DummyOBJ:
    def __init__(self):
        # 3 vertices
        self.vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        # 1 triangle face → attributes a, b, c
        self.faces = [
            type("F", (), {"a": 0, "b": 1, "c": 2, "visible": True}),
        ]


def test_obj2mesh_monkeypatch(monkeypatch):
    def fake_parse(path):
        return DummyOBJ()

    # monkeypatch parse_file
    monkeypatch.setattr("data_structures.parse_file", fake_parse)

    ml = obj2mesh("dummy.obj", mlevel=0)

    assert ml.num_vertices() == 3
    assert ml.num_faces() == 1

    v0 = ml.get_vertex(0)
    assert np.allclose(v0.position(), np.array([0, 0, 0]))