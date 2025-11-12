#!/usr/bin/env python3
"""
Test suite for MAPS Chunk 1: Core Data Structures.
Tests vertex, face, barycentric coordinates, and hierarchy classes.
"""

import os
import sys
from typing import Callable, List

import numpy as np

# Add project root to path to import implementation.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from chunk1_data_structures import (
    BarycentricCoord,
    Face,
    MeshHierarchy,
    MeshLevel,
    Vertex,
    create_mesh_level_from_obj,
)


def test_vertex_creation() -> None:
    """Test Vertex class instantiation and methods."""
    print("Testing Vertex class...")

    v = Vertex(id=0, x=1.0, y=2.0, z=3.0)
    assert v.id == 0
    assert v.x == 1.0
    assert v.y == 2.0
    assert v.z == 3.0

    pos = v.position()
    assert np.allclose(pos, [1.0, 2.0, 3.0])
    assert isinstance(pos, np.ndarray)

    v2 = Vertex(id=1, x=4.0, y=5.0, z=6.0)
    dist = v.distance_to(v2)
    expected = np.sqrt(9 + 9 + 9)  # sqrt(27)
    assert np.isclose(dist, expected)

    print(f"  ✓ Vertex creation: {v}")
    print(f"  ✓ Distance calculation: {dist:.3f}")


def test_face_creation() -> None:
    """Test Face class and methods."""
    print("\nTesting Face class...")

    f = Face(v1=0, v2=1, v3=2)
    assert f.v1 == 0
    assert f.v2 == 1
    assert f.v3 == 2
    assert f.visible is True

    assert f.vertices() == [0, 1, 2]
    assert f.contains_vertex(1) is True
    assert f.contains_vertex(5) is False

    f_clone = f.clone()
    assert f_clone.v1 == f.v1
    assert f_clone.v2 == f.v2
    assert f_clone.v3 == f.v3
    assert f_clone.visible is True

    f.visible = False
    assert f_clone.visible is True  # Clone is independent.

    print(f"  ✓ Face creation: {f}")
    print("  ✓ Face cloning works correctly")


def test_barycentric_coordinates() -> None:
    """Test BarycentricCoord validation."""
    print("\nTesting BarycentricCoord class...")

    # Valid coordinates.
    bc = BarycentricCoord(triangle_id=0, alpha=0.5, beta=0.3, gamma=0.2)
    assert bc.is_valid()
    print(f"  ✓ Valid barycentric: {bc}")

    # Test sum validation.
    try:
        _ = BarycentricCoord(triangle_id=0, alpha=0.5, beta=0.3, gamma=0.3)
        raise AssertionError("Should have raised assertion error")
    except AssertionError as exc:
        print(f"  ✓ Correctly rejects invalid sum: {str(exc)[:60]}...")

    # Edge case: vertex at corner.
    bc_corner = BarycentricCoord(triangle_id=1, alpha=1.0, beta=0.0, gamma=0.0)
    assert bc_corner.is_valid()
    print(f"  ✓ Corner vertex: {bc_corner}")

    # Edge case: vertex on edge.
    bc_edge = BarycentricCoord(triangle_id=2, alpha=0.5, beta=0.5, gamma=0.0)
    assert bc_edge.is_valid()
    print(f"  ✓ Edge vertex: {bc_edge}")


def test_mesh_level() -> None:
    """Test MeshLevel class."""
    print("\nTesting MeshLevel class...")

    level = MeshLevel(level=5)
    assert level.level == 5
    assert level.num_vertices() == 0
    assert level.num_faces() == 0

    # Add vertices.
    for i in range(4):
        v = Vertex(id=i, x=float(i), y=float(i), z=0.0)
        level.add_vertex(v)

    assert level.num_vertices() == 4
    v2 = level.get_vertex(2)
    assert v2 is not None
    assert v2.id == 2

    # Add faces.
    level.add_face(Face(0, 1, 2))
    level.add_face(Face(0, 2, 3))
    assert level.num_faces() == 2

    # Test active faces.
    level.faces[0].visible = False
    active = level.get_active_faces()
    assert len(active) == 1
    assert active[0].v1 == 0 and active[0].v2 == 2 and active[0].v3 == 3

    print(f"  ✓ MeshLevel: {level}")
    print(f"  ✓ Active faces: {len(active)}/2")


def test_mesh_hierarchy() -> None:
    """Test MeshHierarchy class."""
    print("\nTesting MeshHierarchy class...")

    hierarchy = MeshHierarchy()

    # Create 3 levels: 100 -> 50 -> 25 vertices.
    for level_idx, num_verts in [(2, 100), (1, 50), (0, 25)]:
        level = MeshLevel(level=level_idx)
        for i in range(num_verts):
            level.add_vertex(Vertex(id=i, x=0.0, y=0.0, z=0.0))
        hierarchy.add_level(level)

    assert hierarchy.num_levels() == 3
    assert hierarchy.finest_level == 2
    assert hierarchy.coarsest_level == 0

    finest = hierarchy.get_level(2)
    assert finest.num_vertices() == 100

    coarsest = hierarchy.get_level(0)
    assert coarsest.num_vertices() == 25

    ratio = hierarchy.compression_ratio()
    assert np.isclose(ratio, 4.0)

    print(f"  ✓ Hierarchy: {hierarchy}")
    print(f"  ✓ Compression ratio: {ratio:.1f}x")


def test_obj_loading() -> None:
    """Test loading OBJ file into MeshLevel."""
    print("\nTesting OBJ file loading...")

    obj_path = "/Users/hedi/LocalFiles/Maps/MAPS/obja/example/suzanne.obj"

    if os.path.exists(obj_path):
        level = create_mesh_level_from_obj(obj_path, level=0)

        print(f"  ✓ Loaded {level.num_vertices()} vertices")
        print(f"  ✓ Loaded {level.num_faces()} faces")

        v0 = level.get_vertex(0)
        assert v0 is not None
        print(f"  ✓ Sample vertex: {v0}")

        if level.num_faces() > 0:
            face = level.faces[0]
            assert level.get_vertex(face.v1) is not None
            assert level.get_vertex(face.v2) is not None
            assert level.get_vertex(face.v3) is not None
            print(f"  ✓ Sample face: {face}")
    else:
        print(f"  ⚠ Test file not found: {obj_path}")


def run_all_tests() -> bool:
    """Run all test functions."""
    print("=" * 60)
    print("MAPS CHUNK 1: Core Data Structures - Test Suite")
    print("=" * 60)

    tests: List[Callable[[], None]] = [
        test_vertex_creation,
        test_face_creation,
        test_barycentric_coordinates,
        test_mesh_level,
        test_mesh_hierarchy,
        test_obj_loading,
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


