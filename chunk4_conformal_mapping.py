#!/usr/bin/env python3
"""
MAPS Implementation - Chunk 4: Conformal Mapping
Implements z^a conformal map for 1-ring flattening.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
from scipy.spatial import Delaunay

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)

from chunk1_data_structures import Vertex


class ConformalMapper:
    """Conformal mapping utilities for vertex removal."""

    @staticmethod
    def polygon_signed_area(points: List[Tuple[float, float]]) -> float:
        """
        Compute signed area of polygon defined by points.

        Positive value indicates counter-clockwise orientation.
        """
        area = 0.0
        count = len(points)
        if count < 3:
            return 0.0
        for idx in range(count):
            x1, y1 = points[idx]
            x2, y2 = points[(idx + 1) % count]
            area += x1 * y2 - x2 * y1
        return 0.5 * area

    @staticmethod
    def compute_angles_between_neighbors(center: Vertex, neighbors: List[Vertex]) -> List[float]:
        """
        Compute angles at the center vertex between consecutive neighbors.

        Args:
            center: Central vertex.
            neighbors: Ordered list of neighboring vertices.

        Returns:
            List of angles (radians) between consecutive neighbors.
        """
        center_pos = center.position()
        angles: List[float] = []
        count = len(neighbors)

        if count == 0:
            return angles

        for idx in range(count):
            prev_vertex = neighbors[idx - 1]
            current_vertex = neighbors[idx]
            v_prev = prev_vertex.position() - center_pos
            v_curr = current_vertex.position() - center_pos

            prev_norm = np.linalg.norm(v_prev)
            curr_norm = np.linalg.norm(v_curr)

            if prev_norm < 1e-12 or curr_norm < 1e-12:
                angles.append(0.0)
                continue

            v_prev /= prev_norm
            v_curr /= curr_norm
            cos_angle = float(np.clip(np.dot(v_prev, v_curr), -1.0, 1.0))
            angle = float(np.arccos(cos_angle))
            angles.append(angle)

        return angles

    @staticmethod
    def conformal_flatten_1ring(center: Vertex, neighbors: List[Vertex]) -> List[Tuple[float, float]]:
        """
        Map a 1-ring neighborhood to the plane using a conformal map.

        Args:
            center: Central vertex to be removed.
            neighbors: Ordered list of 1-ring neighbors.

        Returns:
            List of 2D coordinates for each neighbor.
        """
        count = len(neighbors)
        if count < 3:
            return [(float(idx), 0.0) for idx in range(count)]

        angles = ConformalMapper.compute_angles_between_neighbors(center, neighbors)
        theta_total = sum(angles)

        if theta_total < 1e-6:
            scale = 1.0
        else:
            scale = (2.0 * np.pi) / theta_total

        flattened: List[Tuple[float, float]] = []
        theta = 0.0

        for idx in range(count):
            neighbor = neighbors[idx]
            radius = center.distance_to(neighbor)
            x = radius * np.cos(scale * theta)
            y = radius * np.sin(scale * theta)
            flattened.append((float(x), float(y)))
            theta += angles[idx]

        return flattened

    @staticmethod
    def conformal_flatten_boundary(center: Vertex, neighbors: List[Vertex]) -> List[Tuple[float, float]]:
        """
        Map boundary vertex 1-ring to a half-disk.

        Args:
            center: Boundary vertex to be removed.
            neighbors: Ordered list of neighbors.

        Returns:
            List of 2D coordinates mapping to a half-disk.
        """
        count = len(neighbors)
        if count < 3:
            return [(float(idx), 0.0) for idx in range(count)]

        angles = ConformalMapper.compute_angles_between_neighbors(center, neighbors)
        theta_total = sum(angles)

        if theta_total < 1e-6:
            scale = 1.0
        else:
            scale = np.pi / theta_total

        flattened: List[Tuple[float, float]] = []
        theta = 0.0

        for idx in range(count):
            neighbor = neighbors[idx]
            radius = center.distance_to(neighbor)
            x = radius * np.cos(scale * theta)
            y = radius * np.sin(scale * theta)
            flattened.append((float(x), float(y)))
            theta += angles[idx]

        if flattened:
            flattened[0] = (flattened[0][0], 0.0)
            flattened[-1] = (flattened[-1][0], 0.0)
        return flattened

    @staticmethod
    def retriangulate_hole(flattened_points: List[Tuple[float, float]]) -> List[Tuple[int, int, int]]:
        """
        Retriangulate a hole using Delaunay triangulation.

        Args:
            flattened_points: 2D points forming boundary of the hole.

        Returns:
            List of triangle indices (each a tuple of three indices).
        """
        point_count = len(flattened_points)
        if point_count < 3:
            return []
        if point_count == 3:
            return [(0, 1, 2)]

        triangles = ConformalMapper._ear_clipping_triangulation(flattened_points)
        if triangles:
            return triangles

        points_array = np.array(flattened_points, dtype=np.float64)

        try:
            triangulation = Delaunay(points_array)
        except Exception:
            triangles = []
            for idx in range(1, point_count - 1):
                triangles.append((0, idx, idx + 1))
            return triangles

        valid_triangles: List[Tuple[int, int, int]] = []
        for simplex in triangulation.simplices:
            i, j, k = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
            if max(i, j, k) < point_count:
                valid_triangles.append((i, j, k))
        return valid_triangles

    @staticmethod
    def check_triangle_flipping(
        triangles: List[Tuple[int, int, int]], points: List[Tuple[float, float]]
    ) -> bool:
        """
        Check if any triangles have negative orientation (flipped).

        Args:
            triangles: List of triangle indices.
            points: 2D point coordinates.

        Returns:
            True if any triangle is flipped; False otherwise.
        """
        for i, j, k in triangles:
            pi = np.array(points[i], dtype=np.float64)
            pj = np.array(points[j], dtype=np.float64)
            pk = np.array(points[k], dtype=np.float64)
            v1 = pj - pi
            v2 = pk - pi
            signed_area = v1[0] * v2[1] - v1[1] * v2[0]
            if signed_area < 0:
                return True
        return False

    @staticmethod
    def _ear_clipping_triangulation(
        points: List[Tuple[float, float]]
    ) -> List[Tuple[int, int, int]]:
        """Triangulate polygon using ear clipping algorithm."""
        n = len(points)
        if n < 3:
            return []

        indices = list(range(n))
        triangles: List[Tuple[int, int, int]] = []
        orientation = 1.0 if ConformalMapper.polygon_signed_area(points) >= 0.0 else -1.0
        guard = 0

        while len(indices) > 3 and guard < n * n:
            ear_found = False
            for idx in range(len(indices)):
                prev_idx = indices[(idx - 1) % len(indices)]
                curr_idx = indices[idx]
                next_idx = indices[(idx + 1) % len(indices)]

                if not ConformalMapper._is_convex_vertex(
                    points[prev_idx], points[curr_idx], points[next_idx], orientation
                ):
                    continue

                if ConformalMapper._polygon_contains_point(
                    points, indices, prev_idx, curr_idx, next_idx
                ):
                    continue

                triangles.append((prev_idx, curr_idx, next_idx))
                del indices[idx]
                ear_found = True
                break

            if not ear_found:
                break
            guard += 1

        if len(indices) == 3:
            triangles.append(tuple(indices))

        return triangles

    @staticmethod
    def _is_convex_vertex(
        prev_pt: Tuple[float, float],
        curr_pt: Tuple[float, float],
        next_pt: Tuple[float, float],
        orientation: float,
    ) -> bool:
        ax = curr_pt[0] - prev_pt[0]
        ay = curr_pt[1] - prev_pt[1]
        bx = next_pt[0] - curr_pt[0]
        by = next_pt[1] - curr_pt[1]
        cross = ax * by - ay * bx
        return cross * orientation >= -1e-10

    @staticmethod
    def _polygon_contains_point(
        points: List[Tuple[float, float]],
        indices: List[int],
        prev_idx: int,
        curr_idx: int,
        next_idx: int,
    ) -> bool:
        a = np.array(points[prev_idx], dtype=np.float64)
        b = np.array(points[curr_idx], dtype=np.float64)
        c = np.array(points[next_idx], dtype=np.float64)

        for idx in indices:
            if idx in (prev_idx, curr_idx, next_idx):
                continue
            p = np.array(points[idx], dtype=np.float64)
            if ConformalMapper._point_in_triangle(p, a, b, c):
                return True
        return False

    @staticmethod
    def _point_in_triangle(
        p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> bool:
        v0 = c - a
        v1 = b - a
        v2 = p - a

        denom = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(denom) < 1e-12:
            return False

        inv_denom = 1.0 / denom
        u = (v2[0] * v1[1] - v1[0] * v2[1]) * inv_denom
        v = (v0[0] * v2[1] - v2[0] * v0[1]) * inv_denom
        w = 1.0 - u - v
        eps = -1e-9
        return u >= eps and v >= eps and w >= eps


