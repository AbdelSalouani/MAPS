import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import data_structures as ds
from mesh_topology import MeshTopology
from priority_queue import (
    VertexPriorityQueue,
    select_independent_set_with_priorities,
    compute_removel_statistics,
)


# ---------- Maillage de base : carré en 2 triangles ---------- #
#
# v0 = (0,0,0)
# v1 = (1,0,0)
# v2 = (1,1,0)
# v3 = (0,1,0)
#
# faces :
#   f0 : (0,1,2)
#   f1 : (0,2,3)
#
# Patch ouvert → bord tout autour.


def build_square_mesh_level():
    ml = ds.MeshLevel(0)

    v0 = ds.Vertex(0, 0.0, 0.0, 0.0)
    v1 = ds.Vertex(1, 1.0, 0.0, 0.0)
    v2 = ds.Vertex(2, 1.0, 1.0, 0.0)
    v3 = ds.Vertex(3, 0.0, 1.0, 0.0)

    ml.add_vertex(v0)
    ml.add_vertex(v1)
    ml.add_vertex(v2)
    ml.add_vertex(v3)

    f0 = ds.Face(0, 1, 2)
    f1 = ds.Face(0, 2, 3)
    ml.add_face(f0)
    ml.add_face(f1)

    topo = MeshTopology(ml.vertices, ml.faces)

    return ml, topo


# ---------- 1) Test build/pop/is_empty/size sur un vrai maillage ---------- #

def test_vertex_priority_queue_build_and_pop():
    mesh_level, topology = build_square_mesh_level()

    pq = VertexPriorityQueue(lambda_weight=0.5, max_degree=10)

    # On ne veut pas exclure les sommets de bord (sinon tout peut être filtré)
    pq.build(mesh_level, topology, exclude_boundary=False)

    # Sur ce maillage, seuls les sommets avec au moins 3 voisins sont candidats : 0 et 2
    assert pq.size() == 2
    assert pq.valid_vertices == {0, 2}

    popped = []
    vid1 = pq.pop()
    popped.append(vid1)
    vid2 = pq.pop()
    popped.append(vid2)
    vid3 = pq.pop()

    # On doit récupérer les deux sommets 0 et 2 (dans un ordre quelconque)
    assert set(popped) == {0, 2}
    assert vid3 is None
    assert pq.is_empty()


# ---------- 2) Test remove() et is_valid() ---------- #

def test_vertex_priority_queue_remove_and_is_valid():
    mesh_level, topology = build_square_mesh_level()
    pq = VertexPriorityQueue(lambda_weight=0.5, max_degree=10)
    pq.build(mesh_level, topology, exclude_boundary=False)

    # On sait que les candidats sont {0, 2}
    assert pq.is_valid(0)
    assert pq.is_valid(2)

    pq.remove(0)

    assert not pq.is_valid(0)
    assert pq.is_valid(2)
    assert pq.size() == 1


# ---------- 3) Test _compute_priority (valeurs normalisées, dans [0,1]) ---------- #

def test_compute_priority_range():
    pq = VertexPriorityQueue(lambda_weight=0.3, max_degree=10)

    p1 = pq._compute_priority(area=1.0, curvature=0.0, max_area=2.0, max_curvature=1.0)
    p2 = pq._compute_priority(area=2.0, curvature=1.0, max_area=2.0, max_curvature=1.0)

    # Les priorités doivent être dans [0,1]
    assert 0.0 <= p1 <= 1.0
    assert 0.0 <= p2 <= 1.0

    # En général, plus d'aire et plus de courbure => priorité plus grande
    assert p2 >= p1


# ---------- 4) Test select_independent_set_with_priorities ---------- #

def test_select_independent_set_with_priorities():
    mesh_level, topology = build_square_mesh_level()

    indep = select_independent_set_with_priorities(
        mesh_level,
        topology,
        lambda_weight=0.5,
        max_degree=10,
        exclude_boundary=False,
    )

    # L'ensemble indépendant est un sous-ensemble des sommets {0,1,2,3}
    assert isinstance(indep, set)
    assert all(vid in mesh_level.vertices for vid in indep)

    # Sur ce petit graphe, les candidats sont {0,2}, mais comme ils sont voisins,
    # l'ensemble indépendant ne doit en contenir au plus qu'un.
    assert len(indep) <= 1
    # S'il y a un sommet choisi, il est parmi {0,2}
    if len(indep) == 1:
        assert next(iter(indep)) in {0, 2}


# ---------- 5) Test build sans candidat (aucune insertion dans le tas) ---------- #

def test_vertex_priority_queue_build_no_candidates():
    class DummyVertex:
        def __init__(self, vid):
            self.id = vid

    class DummyMeshLevel:
        def __init__(self):
            self.vertices = {0: DummyVertex(0)}

    class DummyTopology:
        def get_vertex_degree(self, vid):
            return 1  # <= max_degree

        def is_boundary_vertex(self, vid):
            return False

        def get_neighbors(self, vid):
            return []  # < 3 voisins => rejeté comme candidat

    ml = DummyMeshLevel()
    topo = DummyTopology()

    pq = VertexPriorityQueue(lambda_weight=0.5, max_degree=10)
    pq.build(ml, topo, exclude_boundary=True)

    # Aucun candidat ne doit être inséré
    assert pq.size() == 0
    assert pq.is_empty()


# ---------- 6) Test compute_removel_statistics (fonction définie mais vide) ---------- #

def test_compute_removel_statistics_returns_none():
    result = compute_removel_statistics(set(), None)
    assert result is None
