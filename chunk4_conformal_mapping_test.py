#!/usr/bin/env python3
"""
Test suite for MAPS Chunk 4: Conformal Mapping.
"""

import os
import sys
from typing import Callable, List, Tuple

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from chunk1_data_structures import Vertex
from chunk4_conformal_mapping import ConformalMapper


def build_regular_polygon(center: Vertex, count: int, radius: float = 1.0) -> List[Vertex]:
    """Create neighbors forming a regular polygon around the center."""
    neighbors = []
    for idx in range(count):
        angle = 2.0 * np.pi * idx / count
        x = center.x + radius * np.cos(angle)
        y = center.y + radius * np.sin(angle)
        neighbors.append(Vertex(idx + 1, x, y, center.z))
    return neighbors


def test_angle_computation() -> None:
    """Verify angle computations for orthogonal neighbors."""
    print("Testing angle computation...")

    center = Vertex(0, 0.0, 0.0, 0.0)
    neighbors = [
        Vertex(1, 1.0, 0.0, 0.0),
        Vertex(2, 0.0, 1.0, 0.0),
        Vertex(3, -1.0, 0.0, 0.0),
        Vertex(4, 0.0, -1.0, 0.0),
    ]

    angles = ConformalMapper.compute_angles_between_neighbors(center, neighbors)
    total_angle = sum(angles)
    for idx, angle in enumerate(angles):
        assert np.isclose(angle, np.pi / 2.0, rtol=1e-2)
    assert np.isclose(total_angle, 2.0 * np.pi, rtol=1e-2)

    print(f"  ✓ Angles: {[np.degrees(a) for a in angles]}")
    print(f"  ✓ Total angle: {np.degrees(total_angle)}°")


def test_conformal_flatten_regular() -> None:
    """Flatten a regular polygon and verify distribution."""
    print("\nTesting conformal flattening (regular polygon)...")

    center = Vertex(0, 0.0, 0.0, 0.0)
    neighbors = build_regular_polygon(center, 6, radius=1.0)
    flattened = ConformalMapper.conformal_flatten_1ring(center, neighbors)

    for idx, (x, y) in enumerate(flattened):
        angle = np.arctan2(y, x)
        if angle < 0:
            angle += 2.0 * np.pi
        expected = 2.0 * np.pi * idx / len(flattened)
        assert np.isclose(angle, expected, atol=0.2)
        print(f"  ✓ Neighbor {idx}: ({x:.3f}, {y:.3f}), angle={np.degrees(angle):.1f}°")


def test_conformal_flatten_irregular() -> None:
    """Ensure flattening produces distinct points for irregular rings."""
    print("\nTesting conformal flattening (irregular)...")

    center = Vertex(0, 0.0, 0.0, 0.0)
    neighbors = [
        Vertex(1, 1.0, 0.0, 0.0),
        Vertex(2, 0.5, 0.5, 0.0),
        Vertex(3, -0.3, 0.8, 0.0),
        Vertex(4, -1.0, 0.2, 0.0),
        Vertex(5, -0.5, -0.7, 0.0),
        Vertex(6, 0.3, -0.9, 0.0),
    ]

    flattened = ConformalMapper.conformal_flatten_1ring(center, neighbors)
    for i in range(len(flattened)):
        for j in range(i + 1, len(flattened)):
            distance = np.linalg.norm(np.array(flattened[i]) - np.array(flattened[j]))
            assert distance > 1e-6
    print("  ✓ Flattened points are distinct")


def test_boundary_flattening() -> None:
    """Boundary flattening should map to half-disk with endpoints on x-axis."""
    print("\nTesting boundary flattening...")

    center = Vertex(0, 0.0, 0.0, 0.0)
    neighbors = [
        Vertex(1, 1.0, 0.0, 0.0),
        Vertex(2, 0.7, 0.5, 0.0),
        Vertex(3, 0.0, 0.8, 0.0),
        Vertex(4, -0.7, 0.5, 0.0),
        Vertex(5, -1.0, 0.0, 0.0),
    ]

    flattened = ConformalMapper.conformal_flatten_boundary(center, neighbors)
    assert flattened[0][1] == 0.0
    assert flattened[-1][1] == 0.0
    for idx, (_, y) in enumerate(flattened):
        assert y >= -1e-5, f"Point {idx} has negative y={y}"
    print("  ✓ Half-disk mapping valid")


def test_retriangulation() -> None:
    """Retriangulation should return valid triangles with correct orientation."""
    print("\nTesting retriangulation...")

    points = [
        (1.0, 0.0),
        (0.309, 0.951),
        (-0.809, 0.588),
        (-0.809, -0.588),
        (0.309, -0.951),
    ]

    triangles = ConformalMapper.retriangulate_hole(points)
    assert triangles, "Expected non-empty triangulation"

    flipped = ConformalMapper.check_triangle_flipping(triangles, points)
    assert flipped is False

    for tri in triangles:
        assert len(tri) == 3
        assert max(tri) < len(points)

    print(f"  ✓ Generated {len(triangles)} triangles with consistent orientation")


def test_degenerate_cases() -> None:
    """Verify handling of small neighborhood cases."""
    print("\nTesting degenerate cases...")

    center = Vertex(0, 0.0, 0.0, 0.0)
    two_neighbors = [
        Vertex(1, 1.0, 0.0, 0.0),
        Vertex(2, 0.0, 1.0, 0.0),
    ]
    flattened_two = ConformalMapper.conformal_flatten_1ring(center, two_neighbors)
    assert len(flattened_two) == 2

    three_neighbors = [
        Vertex(1, 1.0, 0.0, 0.0),
        Vertex(2, 0.0, 1.0, 0.0),
        Vertex(3, -1.0, 0.0, 0.0),
    ]
    flattened_three = ConformalMapper.conformal_flatten_1ring(center, three_neighbors)
    triangles = ConformalMapper.retriangulate_hole(flattened_three)
    assert len(triangles) == 1
    print("  ✓ Degenerate cases handled")


def run_all_tests() -> bool:
    """Execute all conformal mapping tests."""
    print("=" * 60)
    print("MAPS CHUNK 4: Conformal Mapping - Test Suite")
    print("=" * 60)

    tests: List[Callable[[], None]] = [
        test_angle_computation,
        test_conformal_flatten_regular,
        test_conformal_flatten_irregular,
        test_boundary_flattening,
        test_retriangulation,
        test_degenerate_cases,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as exc:  # pragma: no cover
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


