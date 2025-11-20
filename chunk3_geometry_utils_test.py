#!/usr/bin/env python3
"""
Test suite for MAPS Chunk 3: Geometry Utilities.
"""

import os
import sys
from typing import Callable, List

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from chunk1_data_structures import Face, MeshLevel, Vertex
from chunk2_mesh_topology import MeshTopology
from chunk3_geometry_utils import GeometryUtils


def build_pyramid_mesh() -> MeshLevel:
    """Construct a simple pyramid-like mesh centered at vertex 0."""
    level = MeshLevel(0)

    level.add_vertex(Vertex(0, 0.0, 0.0, 0.0))
    level.add_vertex(Vertex(1, 1.0, 0.0, 0.0))
    level.add_vertex(Vertex(2, 0.0, 1.0, 0.0))
    level.add_vertex(Vertex(3, -1.0, 0.0, 0.0))
    level.add_vertex(Vertex(4, 0.0, -1.0, 0.0))

    level.add_face(Face(0, 1, 2))
    level.add_face(Face(0, 2, 3))
    level.add_face(Face(0, 3, 4))
    level.add_face(Face(0, 4, 1))

    return level


def test_triangle_area() -> None:
    """Test triangle area calculation."""
    print("Testing triangle area...")

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([3.0, 0.0, 0.0])
    v3 = np.array([0.0, 4.0, 0.0])
    area = GeometryUtils.compute_triangle_area(v1, v2, v3)
    assert np.isclose(area, 6.0)
    print(f"  ✓ Right triangle area: {area}")

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3) / 2.0, 0.0])
    area = GeometryUtils.compute_triangle_area(v1, v2, v3)
    expected = np.sqrt(3) / 4.0
    assert np.isclose(area, expected, rtol=1e-4)
    print(f"  ✓ Equilateral triangle area: {area:.4f}")


def test_face_normal() -> None:
    """Test face normal calculation."""
    print("\nTesting face normal...")

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    normal = GeometryUtils.compute_face_normal(v1, v2, v3)
    expected = np.array([0.0, 0.0, 1.0])
    assert np.allclose(normal, expected)
    print(f"  ✓ Face normal: {normal}")


def test_1ring_area() -> None:
    """Test 1-ring area computation."""
    print("\nTesting 1-ring area...")

    level = build_pyramid_mesh()
    topo = MeshTopology(level.vertices, level.faces)
    center = level.vertices[0]
    neighbors = [level.vertices[i] for i in topo.get_neighbors(0)]
    area = GeometryUtils.compute_area_1ring(center, neighbors, topo)
    assert np.isclose(area, 2.0)
    print(f"  ✓ 1-ring area: {area}")


def test_vertex_normal() -> None:
    """Test vertex normal estimation."""
    print("\nTesting vertex normal...")

    level = build_pyramid_mesh()
    topo = MeshTopology(level.vertices, level.faces)
    center = level.vertices[0]
    neighbors = [level.vertices[i] for i in topo.get_neighbors(0)]
    normal = GeometryUtils.estimate_vertex_normal(center, neighbors, topo)
    expected = np.array([0.0, 0.0, 1.0])
    assert np.allclose(normal, expected, atol=1e-2)
    print(f"  ✓ Vertex normal: {normal}")


def test_curvature_flat_surface() -> None:
    """Curvature should be near zero for flat surfaces."""
    print("\nTesting curvature on flat surface...")

    level = build_pyramid_mesh()
    topo = MeshTopology(level.vertices, level.faces)
    center = level.vertices[0]
    neighbors = [level.vertices[i] for i in topo.get_neighbors(0)]
    curvature = GeometryUtils.estimate_curvature(center, neighbors, topo)
    assert curvature < 0.01
    print(f"  ✓ Flat surface curvature: {curvature:.6f}")


def test_curvature_sphere_cap() -> None:
    """Curvature should be positive for curved surfaces."""
    print("\nTesting curvature on curved surface...")

    radius = 1.0
    center = Vertex(0, 0.0, 0.0, radius)
    level = MeshLevel(0)
    level.add_vertex(center)

    num_neighbors = 8
    theta = np.pi / 4.0

    for i in range(num_neighbors):
        angle = 2 * np.pi * i / num_neighbors
        x = radius * np.sin(theta) * np.cos(angle)
        y = radius * np.sin(theta) * np.sin(angle)
        z = radius * np.cos(theta)
        level.add_vertex(Vertex(i + 1, x, y, z))

    for i in range(num_neighbors):
        next_i = (i + 1) % num_neighbors
        level.add_face(Face(0, i + 1, next_i + 1))

    topo = MeshTopology(level.vertices, level.faces)
    neighbors = [level.vertices[i] for i in topo.get_neighbors(0)]
    curvature = GeometryUtils.estimate_curvature(center, neighbors, topo)
    assert curvature > 0.1
    print(f"  ✓ Sphere cap curvature: {curvature:.4f}")


def test_dihedral_angle() -> None:
    """Test dihedral angle computation."""
    print("\nTesting dihedral angle...")

    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.5, 1.0, 0.0])
    v4 = np.array([0.5, 0.0, 1.0])
    angle = GeometryUtils.compute_dihedral_angle(v1, v2, v3, v4)
    expected = np.pi / 2.0
    assert np.isclose(angle, expected, rtol=1e-2)
    print(f"  ✓ Dihedral angle: {np.degrees(angle):.1f}°")


def run_all_tests() -> bool:
    """Run all geometry tests."""
    print("=" * 60)
    print("MAPS CHUNK 3: Geometry Utilities - Test Suite")
    print("=" * 60)

    tests: List[Callable[[], None]] = [
        test_triangle_area,
        test_face_normal,
        test_1ring_area,
        test_vertex_normal,
        test_curvature_flat_surface,
        test_curvature_sphere_cap,
        test_dihedral_angle,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as exc:  # pragma: no cover - manual test runner
            print(f"\n  ✗ FAILED: {test_func.__name__}")
            print(f"    Error: {exc}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


