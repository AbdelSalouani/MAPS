
import math
import numpy as np
import conformal_mapping as cmp

from data_structures import *
from mesh_topology import MeshTopology
from priority_queue import select_independent_set_with_priorities

TARGET_BASE_SIZE = 50
MAX_LEVELS = 30
LAMBDA = 0.5
MAX_DEGREE = 12
MIN_REMOVAL_FRACTION = 0.04

def build_hierarchy(finest_level):
    cloned_finest = clone_mesh_level(finest_level)
    hierarchy = MeshHierarchy()
    estimated_levels = estimate_num_levels(cloned_finest.num_vertices())
    cloned_finest.level = estimated_levels
    hierarchy.add_level(cloned_finest)

    current_level = cloned_finest
    level_idx = cloned_finest.level

    while (current_level.num_vertices() > TARGET_BASE_SIZE
           and level_idx > 0 and hierarchy.num_levels() < MAX_LEVELS):
        coarser_level = simplify_level(current_level, level_idx-1)
        if coarser_level is None:
            break

        removal_fraction = len(current_level.removed_vertices) / max(current_level.num_vertices(), 1)
        if removal_fraction < 0.01:
            break

        hierarchy.add_level(coarser_level)
        current_level = coarser_level
        level_idx = coarser_level.level
    
    hierarchy.coarsest_level = current_level.level
    return hierarchy

def clone_mesh_level(level):
    cloned = MeshLevel(level)
    for v in level.vertices.values():
        cloned.add_vertex(Vertex(v.id, v.x, v.y, v.z))
        
    for face in level.faces:
        cloned.add_face(face.clone())
    
    return cloned

def estimate_num_levels(num_vertices):
    if num_vertices <= TARGET_BASE_SIZE:
        return 1
    ratio = max(num_vertices / float(TARGET_BASE_SIZE), 1.01)
    return int(math.ceil(3.5 * math.log2(ratio)))


def simplify_level(level, new_level_idx):
    topology = MeshTopology(level.vertices, level.faces)
    indep_set, removal_fraction = select_independent_set_with_fallback(level, topology)
    if not indep_set:
        return None
    
    coarser_level = MeshLevel(new_level_idx)
    for vid, v in level.vertices.items():
        if vid in indep_set:
            continue
        coarser_level.add_vertex(Vertex(v.id, v.x, v.y, v.z))

    remove_vertices_and_retrianguate(level, coarser_level, indep_set, topology)

    level.removed_vertices = sorted(indep_set)

    level.removal_fraction = removal_fraction
    coarser_level.removal_fraction = removal_fraction
    ## explain these 2 lines????????
    
    return coarser_level
    

def select_independent_set_with_fallback(level, topology):
    total_vertices = max(level.num_vertices(), 1)
    attempts = [(MAX_DEGREE, True), (MAX_DEGREE, False),
                (MAX_DEGREE+4, False), (MAX_DEGREE+8, False), (1e6, False)]
    
    last_set = set()
    last_fraction = 0.0

    for degree, exclude_boundary in attempts:
        candidate = select_independent_set_with_priorities(level, topology, exclude_boundary=exclude_boundary)
        fraction = len(candidate) / total_vertices
        if fraction >= MIN_REMOVAL_FRACTION:
            return candidate, fraction
        if candidate:
            last_set = candidate
            last_fraction = fraction

    return last_set, last_fraction

def remove_vertices_and_retrianguate(current, coarser, indep_set, topology):
    processed_faces, added_faces = set(), set()

    
    for center_vid in indep_set:
        center_vertex = current.vertices.get(center_vid)
        if center_vid is None:
            continue

        neighbor_ids = [nid for nid in topology.get_1ring_ordered_safe(center_vid) if nid in coarser.vertices]

        if len(neighbor_ids) < 3:
            continue

        neighbors = [current.vertices[nid] for nid in neighbor_ids]
        is_boundary = topology.is_boundary_vertex(center_vid)

        if is_boundary:
            flattened = cmp.conformal_flatten_boundary(center_vertex, neighbors)
        else:
            flattened = cmp.conformal_flatten_1ring(center_vertex, neighbors)

        if len(flattened) != len(neighbor_ids):
            continue

        polygon_area = cmp.polygon_signed_area(flattened)
        if polygon_area < 0.0:
            neighbor_ids = list(reversed(neighbor_ids))
            neighbors = list(reversed(neighbors))
            flattened = list(reversed(flattened))

        triangles = cmp.retriangulate_hole(flattened)
        if cmp.check_triangle_flipping(triangles, flattened):
            triangles = [(0, i, i+1) for i in range(1, len(neighbor_ids)-1)]

        for t in triangles:
            vids = (neighbor_ids[t[0]],neighbor_ids[t[1]], neighbor_ids[t[2]])
            if len({vids[0], vids[1], vids[2]}) < 3:
                continue
            if any(vid not in coarser.vertices for vid in vids):
                continue

            face_key = tuple(sorted(vids))
            if face_key in added_faces:
                continue

            coarser.add_face(Face(vids[0], vids[1], vids[2]))
            added_faces.add(face_key)

        for face in topology.get_star(center_vid):
            processed_faces.add(tuple(sorted(face.vertices())))

    for face in current.faces:
        if not face.visible:
            continue

        if (face.v1 in indep_set or face.v2 in indep_set or face.v3 in indep_set):
            continue

        face_vertices = (face.v1, face.v2, face.v3)
        if any(v not in coarser.vertices for v in face_vertices):
            continue

        face_key = tuple(sorted(face_vertices))
        if face_key in added_faces:
            continue

        coarser.add_face(face.clone())
        added_faces.add(face_key)


def print_hierarchy_summary(hierarchy):
    print("RUNNING print_hierarchy_summary(hierarchy)")