#!/usr/bin/env python3
"""
MAPS Implementation - Chunk 5: Vertex Priority Queue
Priority-based vertex selection for DK hierarchy construction.
"""

from __future__ import annotations

import heapq
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)

from chunk1_data_structures import MeshLevel, Vertex
from chunk2_mesh_topology import MeshTopology
from chunk3_geometry_utils import GeometryUtils


class VertexPriorityQueue:
    """
    Priority queue for vertex removal in DK hierarchy.

    Vertices are prioritized based on geometric criteria:
    - Small 1-ring area (flat regions)
    - Low curvature (smooth regions)
    """

    def __init__(self, lambda_weight: float = 0.5, max_degree: int = 12):
        """
        Initialize priority queue.

        Args:
            lambda_weight: Weight between area and curvature [0, 1].
            max_degree: Maximum vertex degree allowed for removal.
        """
        self.lambda_weight = float(np.clip(lambda_weight, 0.0, 1.0))
        self.max_degree = max_degree
        self.heap: List[Tuple[float, int]] = []
        self.valid_vertices: Set[int] = set()
        self.removed_vertices: Set[int] = set()

    def build(
        self,
        mesh_level: MeshLevel,
        topology: MeshTopology,
        exclude_boundary: bool = True,
    ) -> None:
        """
        Build priority queue from mesh.

        Args:
            mesh_level: Current mesh level.
            topology: Mesh topology manager.
            exclude_boundary: Skip boundary vertices if True.
        """
        self.heap.clear()
        self.valid_vertices.clear()
        self.removed_vertices.clear()

        geom_utils = GeometryUtils()

        candidate_data: List[Tuple[int, float, float]] = []
        max_area = 0.0
        max_curvature = 0.0

        for vid, vertex in mesh_level.vertices.items():
            degree = topology.get_vertex_degree(vid)
            if degree > self.max_degree:
                continue
            if exclude_boundary and topology.is_boundary_vertex(vid):
                continue

            neighbor_ids = topology.get_neighbors(vid)
            if len(neighbor_ids) < 3:
                continue

            neighbors = [mesh_level.vertices[nid] for nid in neighbor_ids]
            area = geom_utils.compute_area_1ring(vertex, neighbors, topology)
            curvature = geom_utils.estimate_curvature(vertex, neighbors, topology)

            max_area = max(max_area, area)
            max_curvature = max(max_curvature, curvature)
            candidate_data.append((vid, area, curvature))

        if max_area < 1e-12:
            max_area = 1.0
        if max_curvature < 1e-12:
            max_curvature = 1.0

        for vid, area, curvature in candidate_data:
            priority = self._compute_priority(area, curvature, max_area, max_curvature)
            heapq.heappush(self.heap, (priority, vid))
            self.valid_vertices.add(vid)

    def _compute_priority(
        self,
        area: float,
        curvature: float,
        max_area: float,
        max_curvature: float,
    ) -> float:
        """
        Compute vertex removal priority.

        Args:
            area: 1-ring area.
            curvature: Curvature estimate.
            max_area: Maximum area for normalization.
            max_curvature: Maximum curvature for normalization.

        Returns:
            Priority value (lower = higher removal priority).
        """
        norm_area = area / max_area
        norm_curvature = curvature / max_curvature
        priority = (
            self.lambda_weight * norm_area
            + (1.0 - self.lambda_weight) * norm_curvature
        )
        return float(priority)

    def pop(self) -> Optional[int]:
        """
        Pop vertex with lowest priority (highest removal priority).

        Returns:
            Vertex ID, or None if queue is empty.
        """
        while self.heap:
            priority, vid = heapq.heappop(self.heap)
            if vid in self.valid_vertices and vid not in self.removed_vertices:
                self.removed_vertices.add(vid)
                return vid
        return None

    def remove(self, vertex_id: int) -> None:
        """
        Mark vertex as removed (lazy deletion).

        Args:
            vertex_id: ID of vertex to remove from consideration.
        """
        self.removed_vertices.add(vertex_id)
        self.valid_vertices.discard(vertex_id)

    def is_valid(self, vertex_id: int) -> bool:
        """
        Check if vertex is still considered valid and not removed.
        """
        return vertex_id in self.valid_vertices and vertex_id not in self.removed_vertices

    def size(self) -> int:
        """Return the number of valid vertices remaining."""
        return len(self.valid_vertices - self.removed_vertices)

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self.size() == 0


def select_independent_set_with_priorities(
    mesh_level: MeshLevel,
    topology: MeshTopology,
    lambda_weight: float = 0.5,
    max_degree: int = 12,
    exclude_boundary: bool = True,
) -> Set[int]:
    """
    Select maximally independent set using priority queue.

    Args:
        mesh_level: Current mesh level.
        topology: Mesh topology.
        lambda_weight: Priority weight parameter.
        max_degree: Maximum vertex degree allowed.

    Returns:
        Set of vertex IDs to remove (independent set).
    """
    pq = VertexPriorityQueue(lambda_weight=lambda_weight, max_degree=max_degree)
    pq.build(mesh_level, topology, exclude_boundary=exclude_boundary)

    independent_set: Set[int] = set()
    blocked: Set[int] = set()

    while not pq.is_empty():
        vid = pq.pop()
        if vid is None:
            break
        if vid in blocked:
            continue

        independent_set.add(vid)
        blocked.add(vid)

        for neighbor in topology.get_neighbors(vid):
            blocked.add(neighbor)
            pq.remove(neighbor)

    return independent_set


def compute_removal_statistics(
    independent_set: Set[int], mesh_level: MeshLevel
) -> Dict[str, float]:
    """
    Compute statistics about vertex removal.

    Args:
        independent_set: Set of vertices to remove.
        mesh_level: Current mesh level.

    Returns:
        Dictionary with removal statistics.
    """
    total_vertices = mesh_level.num_vertices()
    removed_vertices = len(independent_set)
    remaining_vertices = total_vertices - removed_vertices
    removal_fraction = removed_vertices / total_vertices if total_vertices else 0.0

    return {
        "total_vertices": float(total_vertices),
        "removed_vertices": float(removed_vertices),
        "remaining_vertices": float(remaining_vertices),
        "removal_fraction": float(removal_fraction),
        "removal_percentage": float(100.0 * removal_fraction),
    }


