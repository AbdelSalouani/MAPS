#!/usr/bin/env python3
"""
MAPS Implementation - Chunk 2: Mesh Topology and Adjacency
Provides connectivity and topological query operations.
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)

from chunk1_data_structures import Face, MeshLevel, Vertex


class MeshTopology:
    """
    Manages mesh connectivity and adjacency information.

    Provides efficient queries for:
    - 1-ring neighborhoods
    - Vertex stars (incident faces)
    - Boundary detection
    - Independent set selection
    """

    def __init__(self, vertices: Dict[int, Vertex], faces: List[Face]):
        """
        Initialize topology from vertices and faces.

        Args:
            vertices: Dictionary mapping vertex ID to Vertex object.
            faces: List of Face objects.
        """
        self.vertices = vertices
        self.faces = faces
        self.adjacency = self._build_adjacency()
        self.vertex_faces = self._build_vertex_faces()
        self.edge_faces = self._build_edge_faces()

    def _build_adjacency(self) -> Dict[int, Set[int]]:
        """
        Build vertex-to-vertex adjacency graph.

        Returns:
            Dictionary mapping vertex ID to set of neighbor IDs.
        """
        adj: Dict[int, Set[int]] = defaultdict(set)

        for face in self.faces:
            if not face.visible:
                continue

            v1, v2, v3 = face.v1, face.v2, face.v3

            adj[v1].add(v2)
            adj[v1].add(v3)
            adj[v2].add(v1)
            adj[v2].add(v3)
            adj[v3].add(v1)
            adj[v3].add(v2)

        return dict(adj)

    def _build_vertex_faces(self) -> Dict[int, List[int]]:
        """
        Map each vertex to faces containing it.

        Returns:
            Dictionary mapping vertex ID to list of face indices.
        """
        vf: Dict[int, List[int]] = defaultdict(list)

        for face_idx, face in enumerate(self.faces):
            if not face.visible:
                continue
            for vid in face.vertices():
                vf[vid].append(face_idx)

        return dict(vf)

    def _build_edge_faces(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Map each edge to faces containing it.

        Returns:
            Dictionary mapping edge (as sorted tuple) to face indices.
        """
        ef: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        for face_idx, face in enumerate(self.faces):
            if not face.visible:
                continue

            v1, v2, v3 = face.v1, face.v2, face.v3
            edges = [
                tuple(sorted((v1, v2))),
                tuple(sorted((v2, v3))),
                tuple(sorted((v3, v1))),
            ]

            for edge in edges:
                ef[edge].append(face_idx)

        return dict(ef)

    def get_neighbors(self, vertex_id: int) -> List[int]:
        """
        Get 1-ring neighbors of a vertex.

        Args:
            vertex_id: ID of query vertex.

        Returns:
            List of neighbor vertex IDs.
        """
        neighbors = self.adjacency.get(vertex_id, set())
        return list(neighbors)

    def get_vertex_degree(self, vertex_id: int) -> int:
        """
        Get degree (number of neighbors) of a vertex.

        Args:
            vertex_id: ID of query vertex.

        Returns:
            Number of neighbors.
        """
        return len(self.adjacency.get(vertex_id, set()))

    def get_star(self, vertex_id: int) -> List[Face]:
        """
        Get all faces incident to a vertex (its star).

        Args:
            vertex_id: ID of query vertex.

        Returns:
            List of Face objects containing the vertex.
        """
        face_indices = self.vertex_faces.get(vertex_id, [])
        return [self.faces[i] for i in face_indices if self.faces[i].visible]

    def get_1ring_ordered(self, vertex_id: int) -> List[int]:
        """
        Get 1-ring neighbors in cyclic order around vertex.

        Args:
            vertex_id: ID of query vertex.

        Returns:
            List of neighbor IDs in cyclic order.
        """
        neighbors = self.get_neighbors(vertex_id)
        if len(neighbors) <= 2:
            return neighbors

        star_faces = self.get_star(vertex_id)
        if not star_faces:
            return neighbors

        ordered: List[int] = []
        visited_faces: Set[int] = set()

        first_face = star_faces[0]
        verts = first_face.vertices()
        center_idx = verts.index(vertex_id)
        current = verts[(center_idx + 1) % 3]
        ordered.append(current)
        visited_faces.add(id(first_face))

        while len(ordered) < len(neighbors):
            found_next = False

            for face in star_faces:
                if id(face) in visited_faces:
                    continue
                if not face.contains_vertex(vertex_id):
                    continue
                if not face.contains_vertex(current):
                    continue

                verts = face.vertices()
                for idx, vid in enumerate(verts):
                    if vid != vertex_id:
                        continue

                    prev_v = verts[(idx - 1) % 3]
                    next_v = verts[(idx + 1) % 3]

                    if prev_v == current:
                        current = next_v
                    elif next_v == current:
                        current = prev_v
                    else:
                        continue

                    ordered.append(current)
                    visited_faces.add(id(face))
                    found_next = True
                    break

                if found_next:
                    break

            if not found_next:
                break

        for neighbor in neighbors:
            if neighbor not in ordered:
                ordered.append(neighbor)

        return ordered

    def is_boundary_vertex(self, vertex_id: int) -> bool:
        """
        Check if vertex is on mesh boundary.

        Args:
            vertex_id: ID of query vertex.

        Returns:
            True if vertex has at least one boundary edge.
        """
        for neighbor in self.get_neighbors(vertex_id):
            if self.is_boundary_edge(vertex_id, neighbor):
                return True
        return False

    def is_boundary_edge(self, v1: int, v2: int) -> bool:
        """
        Check if edge is on mesh boundary.

        An edge is on the boundary if it belongs to only one face.

        Args:
            v1: First vertex ID.
            v2: Second vertex ID.

        Returns:
            True if edge is on boundary.
        """
        edge = tuple(sorted((v1, v2)))
        face_count = len(self.edge_faces.get(edge, []))
        return face_count == 1

    def find_independent_set(self, max_degree: int = 12) -> Set[int]:
        """
        Find maximally independent set with degree constraint.

        Args:
            max_degree: Maximum allowed vertex degree.

        Returns:
            Set of vertex IDs forming an independent set.
        """
        independent: Set[int] = set()
        marked: Set[int] = set()

        candidates = []
        for vid in self.vertices:
            degree = self.get_vertex_degree(vid)
            if degree <= max_degree and not self.is_boundary_vertex(vid):
                candidates.append(vid)

        candidates.sort(key=self.get_vertex_degree)

        for vid in candidates:
            if vid in marked:
                continue
            independent.add(vid)
            marked.add(vid)
            for neighbor in self.get_neighbors(vid):
                marked.add(neighbor)

        return independent

    def get_boundary_loops(self) -> List[List[int]]:
        """
        Extract boundary loops (sequences of boundary edges).

        Returns:
            List of boundary loops, each as list of vertex IDs.
        """
        boundary_edges = [
            edge for edge, faces in self.edge_faces.items() if len(faces) == 1
        ]

        if not boundary_edges:
            return []

        loops: List[List[int]] = []
        visited: Set[Tuple[int, int]] = set()

        for start_edge in boundary_edges:
            if start_edge in visited:
                continue

            loop = [start_edge[0], start_edge[1]]
            visited.add(start_edge)
            current = start_edge[1]

            while True:
                found_next = False

                for edge in boundary_edges:
                    if edge in visited:
                        continue

                    if current in edge:
                        visited.add(edge)
                        next_vertex = edge[0] if edge[1] == current else edge[1]
                        loop.append(next_vertex)
                        current = next_vertex
                        found_next = True
                        break

                if not found_next or current == start_edge[0]:
                    break

            loops.append(loop)

        return loops

    def compute_euler_characteristic(self) -> int:
        """
        Compute Euler characteristic: χ = V - E + F.

        Returns:
            Euler characteristic.
        """
        vertex_count = len([vid for vid in self.vertices if vid in self.adjacency])
        face_count = len([face for face in self.faces if face.visible])
        edge_count = len(self.edge_faces)
        return vertex_count - edge_count + face_count


def analyze_mesh_topology(mesh_level: MeshLevel) -> Dict[str, float]:
    """
    Analyze topological properties of a mesh.

    Args:
        mesh_level: MeshLevel to analyze.

    Returns:
        Dictionary with topology statistics.
    """
    topology = MeshTopology(mesh_level.vertices, mesh_level.faces)

    degrees = [topology.get_vertex_degree(vid) for vid in mesh_level.vertices]
    boundary_vertices = [
        vid for vid in mesh_level.vertices if topology.is_boundary_vertex(vid)
    ]

    stats = {
        "num_vertices": mesh_level.num_vertices(),
        "num_faces": mesh_level.num_faces(),
        "num_edges": len(topology.edge_faces),
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "min_degree": float(np.min(degrees)) if degrees else 0.0,
        "max_degree": float(np.max(degrees)) if degrees else 0.0,
        "num_boundary_vertices": len(boundary_vertices),
        "euler_characteristic": topology.compute_euler_characteristic(),
        "boundary_loops": len(topology.get_boundary_loops()),
    }

    return stats


def print_topology_stats(stats: Dict[str, float]) -> None:
    """Print topology statistics in a readable format."""
    print("\n=== Mesh Topology Analysis ===")
    print(f"Vertices: {stats['num_vertices']}")
    print(f"Faces: {stats['num_faces']}")
    print(f"Edges: {stats['num_edges']}")
    print(
        "Vertex degree: "
        f"min={stats['min_degree']}, max={stats['max_degree']}, "
        f"avg={stats['avg_degree']:.2f}"
    )
    print(f"Boundary vertices: {stats['num_boundary_vertices']}")
    print(f"Boundary loops: {stats['boundary_loops']}")
    print(f"Euler characteristic: {stats['euler_characteristic']}")

    chi = stats["euler_characteristic"]
    if stats["num_boundary_vertices"] == 0:
        genus = (2 - chi) / 2.0
        genus_display = int(genus) if math.isclose(genus, round(genus)) else genus
        print(f"Topology: Closed surface, genus ≈ {genus_display}")
    else:
        print("Topology: Open surface with boundary")


