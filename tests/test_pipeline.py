import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import data_structures as ds
import pipeline
import dk


# ----------------------------------------------------------
# Helpers : petit maillage carré (comme d'habitude)
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# compute_barycenter
# ----------------------------------------------------------

def test_compute_barycenter_triangle():
    p = np.array([0.2, 0.2, 0.0])
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])

    bary = pipeline.compute_barycenter(p, v1, v2, v3)
    assert bary is not None
    assert np.all(bary >= -1e-6)


# ----------------------------------------------------------
# project_vertex_to_base
# ----------------------------------------------------------

def test_project_vertex_to_base_simple():
    p = np.array([0.3, 0.3, 0.0])
    base_tri = [
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
    ]

    result = pipeline.project_vertex_to_base(p, base_tri)
    assert result is not None
    idx, bary = result
    assert idx == 0
    assert abs(sum(bary) - 1.0) < 1e-6


# ----------------------------------------------------------
# build_parametrization
# ----------------------------------------------------------

def test_build_parametrization_single_level():
    ml = build_square_mesh_level()
    # DK build_hierarchy renvoie 1 seul niveau
    hierarchy = dk.MeshHierarchy()
    hierarchy.add_level(ml)
    hierarchy.finest_level = 0
    hierarchy.coarsest_level = 0

    param = pipeline.build_parametrization(hierarchy)
    # Aucun sommet supprimé, donc param doit être vide
    assert isinstance(param, dict)
    assert len(param) == 0


# ----------------------------------------------------------
# face_in_level
# ----------------------------------------------------------

def test_face_in_level_basic():
    ml = build_square_mesh_level()
    f = ds.Face(0, 1, 2)
    assert pipeline.face_in_level(ml, f) is True

    f2 = ds.Face(0, 1, 3)
    assert pipeline.face_in_level(ml, f2) is False


# ----------------------------------------------------------
# generate_obja — test avec fichier temporaire
# ----------------------------------------------------------

def test_generate_obja(tmp_path, monkeypatch):
    obja_file = tmp_path / "out.obja"

    # 1 niveau hiérarchie
    ml = build_square_mesh_level()
    hierarchy = dk.MeshHierarchy()
    hierarchy.add_level(ml)
    hierarchy.finest_level = 0
    hierarchy.coarsest_level = 0

    param = {}  # aucun vertex supprimé

    # Monkeypatch output printer to avoid depending on obja.Output logic
    class DummyOutput:
        def __init__(self, handle, random_color=False):
            self.handle = handle

        def add_vertex(self, vid, pos):
            self.handle.write(f"v {vid} {pos[0]} {pos[1]} {pos[2]}\n")
        def add_face(self, idx, face):
            v1 = getattr(face, "v1", getattr(face, "a", None))
            v2 = getattr(face, "v2", getattr(face, "b", None))
            v3 = getattr(face, "v3", getattr(face, "c", None))
            self.handle.write(f"f {v1} {v2} {v3}\n")


    monkeypatch.setattr("pipeline.Output", DummyOutput)

    pipeline.generate_obja(hierarchy, param, obja_file)

    content = obja_file.read_text()
    assert "v 0" in content
    assert "f" in content
    assert "Base domain" in content


# ----------------------------------------------------------
# process_obj2obja — smoke test (sans charger un vrai .obj)
# ----------------------------------------------------------
def test_process_obj2obja_smoke(monkeypatch, tmp_path):
    # Fake obj2mesh: on renvoie juste un MeshLevel simple
    def fake_obj2mesh(path):
        ml = ds.MeshLevel(0)
        v0 = ds.Vertex(0, 0.0, 0.0, 0.0)
        v1 = ds.Vertex(1, 1.0, 0.0, 0.0)
        v2 = ds.Vertex(2, 0.0, 1.0, 0.0)
        ml.add_vertex(v0)
        ml.add_vertex(v1)
        ml.add_vertex(v2)
        ml.add_face(ds.Face(0, 1, 2))
        return ml

    # Monkeypatch obj2mesh utilisé dans pipeline
    monkeypatch.setattr("pipeline.obj2mesh", fake_obj2mesh)

    # Monkeypatch DK hierarchy builder → hiérarchie triviale
    def fake_build_hierarchy(level):
        h = dk.MeshHierarchy()
        h.add_level(level)
        h.finest_level = level.level
        h.coarsest_level = level.level
        return h

    monkeypatch.setattr("pipeline.dk.build_hierarchy", fake_build_hierarchy)

    # Monkeypatch parametrization
    monkeypatch.setattr("pipeline.build_parametrization", lambda h: {})

    # Monkeypatch generator (on ne veut pas vraiment écrire de fichier ici)
    monkeypatch.setattr("pipeline.generate_obja", lambda h, p, path: None)

    out_path = tmp_path / "test.obja"
    pipeline.process_obj2obja("whatever.obj", out_path)

    # Smoke test : le code s'exécute sans lever d'exception
    assert True
