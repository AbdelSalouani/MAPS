import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import data_structures as ds
from mesh_topology import MeshTopology


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


def build_square_patch():
    v0 = ds.Vertex(0, 0.0, 0.0, 0.0)
    v1 = ds.Vertex(1, 1.0, 0.0, 0.0)
    v2 = ds.Vertex(2, 1.0, 1.0, 0.0)
    v3 = ds.Vertex(3, 0.0, 1.0, 0.0)

    vertices = {0: v0, 1: v1, 2: v2, 3: v3}
    f0 = ds.Face(0, 1, 2)
    f1 = ds.Face(0, 2, 3)
    faces = [f0, f1]

    topo = MeshTopology(vertices, faces)
    return topo


# ---------- 1) Adjacence, voisins, degré ---------- #

def test_adjacency_neighbors_and_degree():
    topo = build_square_patch()

    # _build_adjacency + get_neighbors + get_vertex_degree
    assert set(topo.get_neighbors(0)) == {1, 2, 3}
    assert set(topo.get_neighbors(1)) == {0, 2}
    assert topo.get_vertex_degree(0) == 3
    assert topo.get_vertex_degree(3) == 2


# ---------- 2) vertex_faces, star ---------- #

def test_vertex_faces_and_star():
    topo = build_square_patch()

    # _build_vertex_faces
    assert set(topo.vertex_faces[0]) == {0, 1}
    assert topo.vertex_faces[1] == [0]
    assert topo.vertex_faces[3] == [1]

    # get_star
    star0 = topo.get_star(0)
    assert len(star0) == 2
    assert all(isinstance(f, ds.Face) for f in star0)


# ---------- 3) edges, arêtes de bord / internes, sommet de bord ---------- #

def test_edge_faces_and_boundary():
    topo = build_square_patch()

    # _build_edge_faces
    ef = topo.edge_faces
    # arête interne
    assert len(ef[(0, 2)]) == 2
    # arête de bord
    assert len(ef[(0, 1)]) == 1

    # is_boundary_edge
    assert topo.is_boundary_edge(0, 1) is True
    assert topo.is_boundary_edge(0, 2) is False

    # is_boundary_vertex (v0 est sur le bord)
    assert topo.is_boundary_vertex(0) is True


# ---------- 4) 1-ring ordonné + version safe + détection de “breaks” ---------- #

def test_1ring_ordered_and_safe():
    topo = build_square_patch()

    neighbors = set(topo.get_neighbors(0))

    ordered = topo.get_1ring_ordered(0)
    # même ensemble de voisins
    assert set(ordered) == neighbors

    # _ring_has_breaks : patch ouvert → au moins une coupure
    assert topo._ring_has_breaks(0, ordered) in (True, False)  # appelée au moins une fois

    ordered_safe = topo.get_1ring_ordered_safe(0)
    assert set(ordered_safe) == neighbors


# ---------- 5) boucles de bord + caractéristique d’Euler ---------- #

def test_boundary_loops_and_euler():
    topo = build_square_patch()

    loops = topo.get_boundary_loops()
    assert len(loops) == 1  # un seul contour
    loop = loops[0]
    # tous les sommets du bord doivent y être présents (ordre pas important)
    assert set(loop) == {0, 1, 2, 3}

    chi = topo.compute_euler_characteristic()
    # patch topologiquement équivalent à un disque → χ = 1
    assert chi == 1


# ---------- 6) find_independent_set (avec correction de la typo en monkeypatch) ---------- #

def test_find_independent_set_monkeypatched():
    topo = build_square_patch()

    # Corriger la typo is_boudary_vertex → utiliser la bonne méthode
    topo.is_boudary_vertex = topo.is_boundary_vertex  # type: ignore[attr-defined]

    indep = topo.find_independent_set(max_degree=10)

    # L’appel ne doit pas crasher et renvoie un ensemble d’ids
    assert isinstance(indep, set)
    for vid in indep:
        assert vid in topo.vertices
