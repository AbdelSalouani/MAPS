#!/usr/bin/env python3
"""
MAPS Implementation - Chunk 3: Geometry Utilities
Area and curvature computations for vertex prioritization.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)

from chunk1_data_structures import Vertex
from chunk2_mesh_topology import MeshTopology


class GeometryUtils:
    """Geometric computations for the MAPS algorithm."""

    @staticmethod
    def compute_triangle_area(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
        """
        Compute area of a triangle using the cross product.

        Args:
            v1, v2, v3: 3D vertex positions.

        Returns:
            Triangle area.
        """
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross = np.cross(edge1, edge2)
        return 0.5 * np.linalg.norm(cross)

    @staticmethod
    def compute_face_normal(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
        """
        Compute unit normal vector of a triangle face.

        Args:
            v1, v2, v3: 3D vertex positions.

        Returns:
            Unit normal vector (defaults to +Z if degenerate).
        """
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross = np.cross(edge1, edge2)
        norm = np.linalg.norm(cross)
        if norm < 1e-10:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return cross / norm

    @staticmethod
    def compute_area_1ring(center: Vertex, neighbors: List[Vertex], topology: MeshTopology) -> float:
        """
        Compute the total area of the 1-ring neighborhood.

        Args:
            center: Central vertex.
            neighbors: Neighboring vertices.
            topology: Mesh topology manager.

        Returns:
            Sum of triangle areas in the star of the center vertex.
        """
        star_faces = topology.get_star(center.id)
        total_area = 0.0

        for face in star_faces:
            v1 = topology.vertices[face.v1].position()
            v2 = topology.vertices[face.v2].position()
            v3 = topology.vertices[face.v3].position()
            total_area += GeometryUtils.compute_triangle_area(v1, v2, v3)

        return total_area

    @staticmethod
    def estimate_vertex_normal(center: Vertex, neighbors: List[Vertex], topology: MeshTopology) -> np.ndarray:
        """
        Estimate vertex normal using area-weighted face normals.

        Args:
            center: Central vertex.
            neighbors: Neighboring vertices.
            topology: Mesh topology.

        Returns:
            Unit normal vector at vertex (defaults to +Z if degenerate).
        """
        star_faces = topology.get_star(center.id)
        if not star_faces:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

        weighted_normal = np.zeros(3, dtype=np.float64)

        for face in star_faces:
            v1 = topology.vertices[face.v1].position()
            v2 = topology.vertices[face.v2].position()
            v3 = topology.vertices[face.v3].position()

            face_normal = GeometryUtils.compute_face_normal(v1, v2, v3)
            area = GeometryUtils.compute_triangle_area(v1, v2, v3)
            weighted_normal += face_normal * area

        norm = np.linalg.norm(weighted_normal)
        if norm < 1e-10:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return weighted_normal / norm

    @staticmethod
    def _build_tangent_basis(normal: np.ndarray) -> np.ndarray:
        """Construct an orthonormal basis (u, v, n) given a normal vector."""
        if abs(normal[2]) < 0.9:
            tangent_u = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        else:
            tangent_u = np.cross(normal, np.array([1.0, 0.0, 0.0], dtype=np.float64))

        tangent_u_norm = np.linalg.norm(tangent_u)
        if tangent_u_norm < 1e-10:
            tangent_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            tangent_u = tangent_u / tangent_u_norm

        tangent_v = np.cross(normal, tangent_u)
        tangent_v_norm = np.linalg.norm(tangent_v)
        if tangent_v_norm < 1e-10:
            tangent_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            tangent_v = tangent_v / tangent_v_norm

        return np.vstack((tangent_u, tangent_v, normal))

    @staticmethod
    def estimate_curvature(center: Vertex, neighbors: List[Vertex], topology: MeshTopology) -> float:
        """
        Conservative curvature estimate: κ = |κ₁| + |κ₂|.

        Uses tangent plane fitting and quadratic surface approximation.

        Args:
            center: Central vertex.
            neighbors: Neighboring vertices.
            topology: Mesh topology.

        Returns:
            Curvature estimate (sum of absolute principal curvatures).
        """
        if len(neighbors) < 3:
            return 0.0

        normal = GeometryUtils.estimate_vertex_normal(center, neighbors, topology)
        center_pos = center.position()

        basis = GeometryUtils._build_tangent_basis(normal)
        tangent_u = basis[0]
        tangent_v = basis[1]

        points_uv = []
        heights = []

        for neighbor in neighbors:
            diff = neighbor.position() - center_pos
            u = float(np.dot(diff, tangent_u))
            v = float(np.dot(diff, tangent_v))
            h = float(np.dot(diff, normal))
            points_uv.append((u, v))
            heights.append(h)

        design_matrix = []
        rhs = []

        for (u, v), h in zip(points_uv, heights):
            design_matrix.append([u * u, u * v, v * v])
            rhs.append(h)

        A = np.asarray(design_matrix, dtype=np.float64)
        b = np.asarray(rhs, dtype=np.float64)

        if A.shape[0] < 3:
            return 0.0

        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0

        a, b_coef, c = coeffs
        hessian = 2.0 * np.array([[a, b_coef / 2.0], [b_coef / 2.0, c]], dtype=np.float64)

        eigenvalues = np.linalg.eigvalsh(hessian)
        curvature = float(np.sum(np.abs(eigenvalues)))
        return curvature

    @staticmethod
    def compute_dihedral_angle(
        v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray
    ) -> float:
        """
        Compute dihedral angle between two triangles sharing an edge.

        Triangles: (v1, v2, v3) and (v1, v2, v4) share edge (v1, v2).

        Args:
            v1, v2: Edge vertices.
            v3, v4: Opposite vertices of each triangle.

        Returns:
            Dihedral angle in radians [0, π].
        """
        normal1 = GeometryUtils.compute_face_normal(v1, v2, v3)
        normal2 = GeometryUtils.compute_face_normal(v1, v2, v4)
        cos_angle = float(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
        return float(np.arccos(cos_angle))


