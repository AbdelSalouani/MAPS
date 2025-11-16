#!/usr/bin/env python3
"""
MAPS Implementation - Chunk 6: DK Hierarchy Construction
Complete pipeline for building mesh hierarchy.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)

from chunk1_data_structures import Face, MeshHierarchy, MeshLevel, Vertex, create_mesh_level_from_obj
from chunk2_mesh_topology import MeshTopology
from chunk4_conformal_mapping import ConformalMapper
from chunk5_priority_queue import select_independent_set_with_priorities


class DKHierarchyBuilder:
    """
    Builds a Dobkin-Kirkpatrick hierarchy through iterative simplification.
    """

    def __init__(
        self,
        lambda_weight: float = 0.5,
        max_degree: int = 12,
        target_base_size: int = 50,
        max_levels: int = 30,
        min_removal_fraction: float = 0.04,
    ):
        self.lambda_weight = float(np.clip(lambda_weight, 0.0, 1.0))
        self.max_degree = max_degree
        self.target_base_size = max(1, target_base_size)
        self.max_levels = max(1, max_levels)
        self.min_removal_fraction = max(0.0, min_removal_fraction)
        self.conformal_mapper = ConformalMapper()

    def build_hierarchy(self, finest_level: MeshLevel) -> MeshHierarchy:
        """
        Build complete DK hierarchy from a finest mesh level.
        """
        cloned_finest = self._clone_mesh_level(finest_level)
        hierarchy = MeshHierarchy()
        estimated_levels = self._estimate_num_levels(cloned_finest.num_vertices())
        cloned_finest.level = estimated_levels
        hierarchy.add_level(cloned_finest)

        current_level = cloned_finest
        level_idx = cloned_finest.level

        while (
            current_level.num_vertices() > self.target_base_size
            and level_idx > 0
            and hierarchy.num_levels() < self.max_levels
        ):
            coarser_level = self.simplify_one_level(current_level, level_idx - 1)
            if coarser_level is None:
                break

            removal_fraction = len(current_level.removed_vertices) / max(
                current_level.num_vertices(), 1
            )
            if removal_fraction < 0.01:
                # Too little progress, abort to avoid infinite loops.
                break

            hierarchy.add_level(coarser_level)
            current_level = coarser_level
            level_idx = coarser_level.level

        hierarchy.coarsest_level = current_level.level
        return hierarchy

    def _clone_mesh_level(self, level: MeshLevel) -> MeshLevel:
        cloned = MeshLevel(level.level)
        for vertex in level.vertices.values():
            cloned.add_vertex(Vertex(vertex.id, vertex.x, vertex.y, vertex.z))
        for face in level.faces:
            cloned.add_face(face.clone())
        return cloned

    def _estimate_num_levels(self, num_vertices: int) -> int:
        if num_vertices <= self.target_base_size:
            return 1
        ratio = max(num_vertices / float(self.target_base_size), 1.01)
        return int(math.ceil(3.5 * math.log2(ratio)))

    def simplify_one_level(
        self, current_level: MeshLevel, new_level_idx: int
    ) -> Optional[MeshLevel]:
        """
        Perform one simplification step to generate the next coarser level.
        """
        topology = MeshTopology(current_level.vertices, current_level.faces)
        independent_set, removal_fraction = self._select_independent_set_with_fallback(
            current_level, topology
        )

        if not independent_set:
            return None

        coarser_level = MeshLevel(new_level_idx)

        for vid, vertex in current_level.vertices.items():
            if vid in independent_set:
                continue
            coarser_level.add_vertex(Vertex(vertex.id, vertex.x, vertex.y, vertex.z))

        self._remove_vertices_and_retriangulate(
            current_level=current_level,
            coarser_level=coarser_level,
            independent_set=independent_set,
            topology=topology,
        )

        current_level.removed_vertices = sorted(independent_set)
        current_level.removal_fraction = removal_fraction  # type: ignore[attr-defined]
        coarser_level.removal_fraction = removal_fraction  # type: ignore[attr-defined]
        return coarser_level

    def _select_independent_set_with_fallback(
        self, current_level: MeshLevel, topology: MeshTopology
    ) -> Tuple[Set[int], float]:
        total_vertices = max(current_level.num_vertices(), 1)
        attempts = [
            (self.max_degree, True),
            (self.max_degree, False),
            (self.max_degree + 4, False),
            (self.max_degree + 8, False),
            (10**6, False),
        ]

        last_set: Set[int] = set()
        last_fraction = 0.0

        for degree, exclude_boundary in attempts:
            candidate = select_independent_set_with_priorities(
                current_level,
                topology,
                lambda_weight=self.lambda_weight,
                max_degree=degree,
                exclude_boundary=exclude_boundary,
            )
            fraction = len(candidate) / total_vertices
            if fraction >= self.min_removal_fraction:
                return candidate, fraction
            if candidate:
                last_set = candidate
                last_fraction = fraction

        return last_set, last_fraction

    def _remove_vertices_and_retriangulate(
        self,
        current_level: MeshLevel,
        coarser_level: MeshLevel,
        independent_set: Set[int],
        topology: MeshTopology,
    ) -> None:
        processed_faces: Set[Tuple[int, int, int]] = set()
        added_faces: Set[Tuple[int, int, int]] = set()

        for center_vid in independent_set:
            center_vertex = current_level.vertices.get(center_vid)
            if center_vertex is None:
                continue

            neighbor_ids = [
                nid
                for nid in topology.get_1ring_ordered(center_vid)
                if nid in coarser_level.vertices
            ]

            if len(neighbor_ids) < 3:
                continue

            neighbors = [current_level.vertices[nid] for nid in neighbor_ids]
            is_boundary = topology.is_boundary_vertex(center_vid)

            if is_boundary:
                flattened = self.conformal_mapper.conformal_flatten_boundary(
                    center_vertex, neighbors
                )
            else:
                flattened = self.conformal_mapper.conformal_flatten_1ring(
                    center_vertex, neighbors
                )

            if len(flattened) != len(neighbor_ids):
                continue

            polygon_area = self.conformal_mapper.polygon_signed_area(flattened)
            if polygon_area < 0.0:
                neighbor_ids = list(reversed(neighbor_ids))
                neighbors = list(reversed(neighbors))
                flattened = list(reversed(flattened))

            triangles = self.conformal_mapper.retriangulate_hole(flattened)

            if self.conformal_mapper.check_triangle_flipping(triangles, flattened):
                triangles = [
                    (0, i, i + 1) for i in range(1, len(neighbor_ids) - 1)
                ]

            for tri in triangles:
                vids = (
                    neighbor_ids[tri[0]],
                    neighbor_ids[tri[1]],
                    neighbor_ids[tri[2]],
                )
                if len({vids[0], vids[1], vids[2]}) < 3:
                    continue
                if any(vid not in coarser_level.vertices for vid in vids):
                    continue

                face_key = tuple(sorted(vids))
                if face_key in added_faces:
                    continue

                coarser_level.add_face(Face(vids[0], vids[1], vids[2]))
                added_faces.add(face_key)

            for face in topology.get_star(center_vid):
                processed_faces.add(tuple(sorted(face.vertices())))

        for face in current_level.faces:
            if not face.visible:
                continue

            if (
                face.v1 in independent_set
                or face.v2 in independent_set
                or face.v3 in independent_set
            ):
                continue

            face_vertices = (face.v1, face.v2, face.v3)
            if any(v not in coarser_level.vertices for v in face_vertices):
                continue

            face_key = tuple(sorted(face_vertices))
            if face_key in added_faces:
                continue

            coarser_level.add_face(face.clone())
            added_faces.add(face_key)


def print_hierarchy_summary(hierarchy: MeshHierarchy) -> None:
    """
    Print a human-readable summary of the hierarchy.
    """
    print("\n" + "=" * 70)
    print("HIERARCHY SUMMARY")
    print("=" * 70)
    print(f"Total levels: {hierarchy.num_levels()}")
    print(f"Finest level: {hierarchy.finest_level}")
    print(f"Coarsest level: {hierarchy.coarsest_level}")
    print(f"Compression ratio: {hierarchy.compression_ratio():.2f}x\n")
    print(f"{'Level':<8}{'Vertices':<12}{'Faces':<12}{'Removed':<12}{'% Removed':<12}")
    print("-" * 70)

    for level_idx in range(
        hierarchy.finest_level, hierarchy.coarsest_level - 1, -1
    ):
        level = hierarchy.get_level(level_idx)
        num_vertices = level.num_vertices()
        num_faces = level.num_faces()
        removed = len(level.removed_vertices)
        removal_pct = (100.0 * removed / num_vertices) if num_vertices else 0.0

        print(
            f"{level_idx:<8}{num_vertices:<12}{num_faces:<12}"
            f"{removed:<12}{removal_pct:<12.1f}"
        )

    print("=" * 70)


