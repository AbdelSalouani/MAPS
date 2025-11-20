#!/usr/bin/env python3
"""
Test suite for MAPS Chunk 2: Mesh Topology and Adjacency.
"""

import os
import sys
from typing import Callable, List

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from chunk1_data_structures import Face, MeshLevel, Vertex, create_mesh_level_from_obj
from chunk2_mesh_topology import (
    MeshTopology,
    analyze_mesh_topology,
    print_topology_stats,
)


def create_simple_quad_mesh() -> MeshLevel:
    """Create a simple quad mesh (2 triangles) for testing."""
    level = MeshLevel(0)

    level.add_vertex(Vertex(0, 0.0, 0.0, 0.0))
    level.add_vertex(Vertex(1, 1.0, 0.0, 0.0))
    level.add_vertex(Vertex(2, 1.0, 1.0, 0.0))
    level.add_vertex(Vertex(3, 0.0, 1.0, 0.0))

    level.add_face(Face(0, 1, 2))
    level.add_face(Face(0, 2, 3))

    return level


def test_adjacency_construction() -> None:
    """Test adjacency graph construction."""
    print("Testing adjacency construction...")

    level = create_simple_quad_mesh()
    topo = MeshTopology(level.vertices, level.faces)

    neighbors0 = topo.get_neighbors(0)
    assert set(neighbors0) == {1, 2, 3}
    print(f"  ✓ Vertex 0 neighbors: {neighbors0}")

    neighbors2 = topo.get_neighbors(2)
    assert set(neighbors2) == {0, 1, 3}
    print(f"  ✓ Vertex 2 neighbors: {neighbors2}")

    assert topo.get_vertex_degree(0) == 3
    assert topo.get_vertex_degree(1) == 2
    print("  ✓ Vertex degrees correct")


def test_vertex_star() -> None:
    """Test vertex star computation."""
    print("\nTesting vertex star...")

    level = create_simple_quad_mesh()
    topo = MeshTopology(level.vertices, level.faces)

    star0 = topo.get_star(0)
    assert len(star0) == 2
    for face in star0:
        assert face.contains_vertex(0)
    print(f"  ✓ Vertex 0 star: {len(star0)} faces")

    star1 = topo.get_star(1)
    assert len(star1) == 1
    print(f"  ✓ Vertex 1 star: {len(star1)} face")


def test_boundary_detection() -> None:
    """Test boundary vertex/edge detection."""
    print("\nTesting boundary detection...")

    level = create_simple_quad_mesh()
    topo = MeshTopology(level.vertices, level.faces)

    for vid in range(4):
        is_boundary = topo.is_boundary_vertex(vid)
        print(f"  Vertex {vid}: boundary={is_boundary}")
        assert is_boundary is True

    assert topo.is_boundary_edge(0, 1) is True
    assert topo.is_boundary_edge(0, 2) is False
    print("  ✓ Boundary edge detection correct")


def test_ordered_1ring() -> None:
    """Test ordered 1-ring traversal."""
    print("\nTesting ordered 1-ring...")

    level = create_simple_quad_mesh()
    topo = MeshTopology(level.vertices, level.faces)

    ordered = topo.get_1ring_ordered(0)
    assert len(ordered) == 3
    assert set(ordered) == {1, 2, 3}
    print(f"  ✓ Vertex 0 ordered 1-ring: {ordered}")


def test_independent_set() -> None:
    """Test independent set selection."""
    print("\nTesting independent set selection...")

    level = create_simple_quad_mesh()
    topo = MeshTopology(level.vertices, level.faces)

    indep_set = topo.find_independent_set(max_degree=12)

    for v1 in indep_set:
        for v2 in indep_set:
            if v1 != v2:
                assert v2 not in topo.get_neighbors(v1)

    print(f"  ✓ Independent set: {indep_set}")
    print(f"  ✓ Independence verified")


def test_euler_characteristic() -> None:
    """Test Euler characteristic computation."""
    print("\nTesting Euler characteristic...")

    level = create_simple_quad_mesh()
    topo = MeshTopology(level.vertices, level.faces)

    chi = topo.compute_euler_characteristic()
    V = len(level.vertices)
    E = len(topo.edge_faces)
    F = len(level.faces)

    assert chi == V - E + F
    print(f"  ✓ Euler characteristic: {chi} (V={V}, E={E}, F={F})")


def test_real_mesh() -> None:
    """Test topology analysis on real mesh."""
    print("\nTesting with real mesh...")

    obj_path = "/Users/hedi/LocalFiles/Maps/MAPS/obja/example/suzanne.obj"

    if not os.path.exists(obj_path):
        print(f"  ⚠ Test file not found: {obj_path}")
        return

    level = create_mesh_level_from_obj(obj_path)
    stats = analyze_mesh_topology(level)
    print_topology_stats(stats)

    assert stats["num_vertices"] > 0
    assert stats["num_faces"] > 0
    assert stats["avg_degree"] > 0

    print("  ✓ Real mesh analysis complete")


def run_all_tests() -> bool:
    """Run all topology tests."""
    print("=" * 60)
    print("MAPS CHUNK 2: Mesh Topology - Test Suite")
    print("=" * 60)

    tests: List[Callable[[], None]] = [
        test_adjacency_construction,
        test_vertex_star,
        test_boundary_detection,
        test_ordered_1ring,
        test_independent_set,
        test_euler_characteristic,
        test_real_mesh,
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


