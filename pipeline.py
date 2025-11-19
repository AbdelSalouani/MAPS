
import time
import numpy as np
import dk

from obja import Output, parse_file
from obja import Face as ObjaFace
from data_structures import *
from mesh_topology import MeshTopology

LAMBDA = 0.5
MAX_DEGREE = 12
TARGET_BASE_SIZE = 80
MAX_LEVELS = 30

def process_obj2obja(obj_path, obja_path):
    print("\n" + "="*70)
    print("MAPS PROGRESSIVE ENCODING PIPELINE")
    print("="*70)

    print("\n Step 1/4 Loading input...")
    finest_level = obj2mesh(obj_path)
    print(f"  Loaded mesh: {finest_level.num_vertices()} vertices, ")
    print(f"{finest_level.num_faces()} faces")

    print("\n Step 2/4 Building DK hierarchy...")
    hierarchy = dk.build_hierarchy(finest_level)
    dk.print_hierarchy_summary(hierarchy)

    print("\n Step 3/4 Building parametrization...")
    parameterization = build_parametrization(hierarchy)
    print(f"  Stored barycentric coords for {len(parameterization)} vertices")

    print("\n Step 4/4 Generating OBJA...")
    generate_obja(hierarchy, parameterization, obja_path)

    print("\n"+"="*70)
    print(f"SUCCESS: Progressive OBJA written to {obja_path}")
    print("="*70)


def build_parametrization(hierarchy):
    parameterization = dict()
    base_level = hierarchy.get_level(hierarchy.coarsest_level)
    base_faces = list(base_level.faces)

    face_vertices = list()
    for f in base_faces:
        v1 = base_level.vertices[f.v1].position()
        v2 = base_level.vertices[f.v2].position()
        v3 = base_level.vertices[f.v3].position()
        face_vertices.append((v1, v2, v3))

    for level_idx in range(hierarchy.finest_level, hierarchy.coarsest_level, -1):
        level = hierarchy.get_level(level_idx)
        for vid in level.removed_vertices:
            if vid in parameterization:
                continue
            vertex = level.vertices.get(vid)
            if vertex is None:
                continue
            bary = project_vertex_to_base(vertex.position(), face_vertices)
            if bary is not None:
                face_idx, coords = bary
                parameterization[vid] = BarycentricCoordinates(
                    triangle_id = face_idx,
                    a=float(coords[0]), b=float(coords[1]), g=float(coords[2])
                )

    return parameterization

def project_vertex_to_base(point, base_triangles):
    best_idx, best_coords, best_dist = None, None, float("inf")

    for idx, (v1, v2, v3) in enumerate(base_triangles):
        bary = compute_barycenter(point, v1, v2, v3)
        if bary is None:
            continue
        if np.any(bary < -0.05) or np.any(bary > 1.05):
            continue

        reconstructed = bary[0]*v1 + bary[1]*v2 + bary[2]*v3
        dist = np.linalg.norm(point-reconstructed)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
            best_coords = bary

    if best_idx is not None and best_coords is not None:
        bary = np.clip(best_coords, 0.0, 1.0)
        bary = bary / np.sum(bary)
        return best_idx, bary
    
    return None

def compute_barycenter(point, v1, v2, v3):
    v0_vec = v2 - v1
    v1_vec = v2 - v1
    v2_vec = point - v1

    d00 = np.dot(v0_vec, v0_vec)
    d01 = np.dot(v0_vec, v1_vec)
    d11 = np.dot(v1_vec, v1_vec)
    d20 = np.dot(v2_vec, v0_vec)
    d21 = np.dot(v2_vec, v1_vec)

    denom = d00*d11 - d01*d01
    if abs(denom) < 1e-12:
        return None
    
    inv_denom = 1.0 / denom
    beta = (d11*d20 - d01*d21) * inv_denom
    gamma = (d00*d21 - d01*d20) * inv_denom
    alpha = 1.0 - beta - gamma
    return np.array([alpha, beta, gamma], dtype=np.float64)

def generate_obja(hierarchy, param, obja_path):
    with open(obja_path, "w") as handle:
        output = Output(handle, random_color=True)
        base_level = hierarchy.get_level(hierarchy.coarsest_level)
        vertex_present = dict()

        for vid in sorted(base_level.vertices.keys()):
            vertex = base_level.vertices[vid]
            output.add_vertex(vid, vertex.position())
            vertex_present[vid] = True

        face_index = 0
        for f in base_level.faces:
            if (vertex_present.get(f.v1) and vertex_present.get(f.v2) and vertex_present.get(f.v3)):
                obja_face = ObjaFace(f.v1, f.v2, f.v3, f.visible)
                output.add_face(face_index, obja_face)
                face_index += 1

        handle.write(
            f"# Base domain: {base_level.num_vertices()} vertices,"
            f"{base_level.num_faces()} faces\n"
        )

        for level_idx in range(hierarchy.coarsest_level+1, hierarchy.finest_level+1):
            prev_level = hierarchy.get_level(level_idx-1)
            current_level = hierarchy.get_level(level_idx)
            removed_vertices = current_level.removed_vertices

            if not removed_vertices:
                continue

            handle.write(f"# Level {level_idx}: {len(removed_vertices)} new vertices\n")
            for vid in removed_vertices:
                vertex = current_level.vertices.get(vid)
                if vertex is None:
                    continue

                output.add_vertex(vid, vertex.position())
                vertex_present[vid] = True

            for f in current_level.faces:
                if not (vertex_present.get(f.v1) 
                        and vertex_present.get(f.v2)
                        and vertex_present.get(f.v3)):
                    continue

                if not face_in_level(prev_level, f):
                    obja_face = ObjaFace(f.v1, f.v2, f.v3, f.visible)
                    output.add_face(face_index, obja_face)
                    face_index += 1


def face_in_level(level, face):
    for existing in level.faces:
        if not existing.visible:
            continue

        if {existing.v1, existing.v2, existing.v3} == {face.v1, face.v2, face.v3}:
            return True
        
    return False