import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import data_structures as ds
import dk


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
# Patch ouvert


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

    return ml


# ---------- estimate_num_levels ---------- #

def test_estimate_num_levels_small_and_large():
    # <= TARGET_BASE_SIZE => 1
    assert dk.estimate_num_levels(10) == 1
    assert dk.estimate_num_levels(dk.TARGET_BASE_SIZE) == 1

    # Exemple pour un grand nombre de sommets
    # num_vertices = 200, TARGET_BASE_SIZE = 50 -> ratio = 4
    # log2(4) = 2 ; 3.5 * 2 = 7 ; ceil(7) = 7
    levels = dk.estimate_num_levels(200)
    assert levels == 7


# ---------- clone_mesh_level ---------- #

def test_clone_mesh_level_basic():
    ml = build_square_mesh_level()

    cloned = dk.clone_mesh_level(ml)

    # Même nombre de sommets et de faces
    assert cloned.num_vertices() == ml.num_vertices()
    assert cloned.num_faces() == ml.num_faces()

    # Les objets Vertex sont différents (deep copy)
    orig_ids = sorted(ml.vertices.keys())
    clone_ids = sorted(cloned.vertices.keys())
    assert orig_ids == clone_ids
    for vid in orig_ids:
        assert cloned.vertices[vid] is not ml.vertices[vid]


# ---------- select_independent_set_with_fallback ---------- #

def test_select_independent_set_with_fallback(monkeypatch):
    ml = build_square_mesh_level()
    topo = dk.MeshTopology(ml.vertices, ml.faces)

    # Monkeypatch : aucun sommet n'est considéré comme "boundary"
    monkeypatch.setattr(dk.MeshTopology, "is_boundary_vertex", lambda self, vid: False)

    indep_set, fraction = dk.select_independent_set_with_fallback(ml, topo)

    assert isinstance(indep_set, set)
    assert len(indep_set) >= 1
    assert 0.0 < fraction <= 1.0
    for vid in indep_set:
        assert vid in ml.vertices


# ---------- simplify_level (+ remove_vertices_and_retrianguate indirectement) ---------- #

def test_simplify_level_reduces_vertices(monkeypatch):
    ml = build_square_mesh_level()

    # Monkeypatch boundary pour éviter conformal_flatten_boundary
    monkeypatch.setattr(dk.MeshTopology, "is_boundary_vertex", lambda self, vid: False)

    coarser = dk.simplify_level(ml, new_level_idx=0)

    # Doit produire un niveau plus grossier
    assert coarser is not None
    assert isinstance(coarser, ds.MeshLevel)

    # On doit avoir supprimé au moins un sommet
    assert coarser.num_vertices() < ml.num_vertices()

    # Il doit rester des faces
    assert coarser.num_faces() > 0

    # Les sommets supprimés doivent être enregistrés dans removed_vertices
    assert hasattr(ml, "removed_vertices")
    assert len(ml.removed_vertices) >= 1


# ---------- build_hierarchy ---------- #

def test_build_hierarchy_single_level():
    ml = build_square_mesh_level()

    hierarchy = dk.build_hierarchy(ml)

    # Comme le maillage est petit (< TARGET_BASE_SIZE), on ne doit avoir qu'un seul niveau
    assert hierarchy.num_levels() == 1
    assert hierarchy.finest_level == hierarchy.coarsest_level

    lvl_id = hierarchy.finest_level
    lvl = hierarchy.get_level(lvl_id)

    assert lvl.num_vertices() == ml.num_vertices()
    assert lvl.num_faces() == ml.num_faces()


# ---------- print_hierarchy_summary ---------- #

def test_print_hierarchy_summary_smoke(capsys):
    ml = build_square_mesh_level()
    hierarchy = dk.build_hierarchy(ml)

    dk.print_hierarchy_summary(hierarchy)
    captured = capsys.readouterr()

    # Vérifier simplement que quelque chose a été imprimé
    assert "RUNNING print_hierarchy_summary" in captured.out
