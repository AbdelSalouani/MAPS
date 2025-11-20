#!/usr/bin/env python3
"""
Test suite for MAPS Chunk 5: Vertex Priority Queue.
"""

import os
import sys
from typing import Callable, List, Set

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from chunk1_data_structures import Face, MeshLevel, Vertex, create_mesh_level_from_obj
from chunk2_mesh_topology import MeshTopology
from chunk5_priority_queue import (
    VertexPriorityQueue,
    compute_removal_statistics,
    select_independent_set_with_priorities,
)


def create_test_mesh_with_varying_complexity() -> MeshLevel:
    """Create a mesh containing flat and curved regions."""
    level = MeshLevel(0)

    for i in range(5):
        for j in range(5):
            vid = i * 5 + j
            level.add_vertex(Vertex(vid, float(i), float(j), 0.0))

    offset = 25
    for i in range(3):
        for j in range(3):
            vid = offset + i * 3 + j
            x = float(i) + 5.0
            y = float(j)
            z = 2.0 * np.sqrt(max(0.0, 1.0 - (x - 6.0) ** 2 - (y - 1.0) ** 2))
            level.add_vertex(Vertex(vid, x, y, z))

    for i in range(4):
        for j in range(4):
            v1 = i * 5 + j
            v2 = v1 + 1
            v3 = v1 + 5
            v4 = v3 + 1
            level.add_face(Face(v1, v2, v3))
            level.add_face(Face(v2, v4, v3))

    for i in range(2):
        for j in range(2):
            v1 = offset + i * 3 + j
            v2 = v1 + 1
            v3 = v1 + 3
            v4 = v3 + 1
            level.add_face(Face(v1, v2, v3))
            level.add_face(Face(v2, v4, v3))

    return level


def test_priority_queue_construction() -> None:
    """Ensure priority queue builds and contains vertices."""
    print("Testing priority queue construction...")
    level = create_test_mesh_with_varying_complexity()
    topo = MeshTopology(level.vertices, level.faces)
    pq = VertexPriorityQueue(lambda_weight=0.5, max_degree=12)
    pq.build(level, topo, exclude_boundary=True)
    assert pq.size() > 0
    print(f"  ✓ Queue size: {pq.size()}")


def test_priority_ordering() -> None:
    """Vertices in flat regions should have higher removal priority."""
    print("\nTesting priority ordering...")
    level = create_test_mesh_with_varying_complexity()
    topo = MeshTopology(level.vertices, level.faces)
    pq = VertexPriorityQueue(lambda_weight=0.5, max_degree=12)
    pq.build(level, topo, exclude_boundary=True)

    popped: List[int] = []
    for _ in range(min(5, pq.size())):
        vid = pq.pop()
        if vid is not None:
            popped.append(vid)
    flat_count = sum(1 for vid in popped if vid < 25)
    assert flat_count >= 1
    print(f"  ✓ First vertices removed: {popped}")
    print(f"  ✓ Flat region vertices among first: {flat_count}")


def test_independent_set_selection() -> None:
    """Independent set should contain non-adjacent vertices."""
    print("\nTesting independent set selection...")
    level = create_test_mesh_with_varying_complexity()
    topo = MeshTopology(level.vertices, level.faces)
    independent = select_independent_set_with_priorities(level, topo, 0.5, 12)
    for vid in independent:
        neighbors = set(topo.get_neighbors(vid))
        assert independent.isdisjoint(neighbors)
    stats = compute_removal_statistics(independent, level)
    assert stats["removal_fraction"] > 0.05
    print(f"  ✓ Independent set size: {len(independent)}")
    print(f"  ✓ Removal fraction: {stats['removal_percentage']:.1f}%")


def test_lazy_deletion() -> None:
    """Verify lazy deletion prevents reusing removed vertices."""
    print("\nTesting lazy deletion...")
    level = create_test_mesh_with_varying_complexity()
    topo = MeshTopology(level.vertices, level.faces)
    pq = VertexPriorityQueue(lambda_weight=0.5, max_degree=12)
    pq.build(level, topo, exclude_boundary=True)

    initial_size = pq.size()
    first = pq.pop()
    assert first is not None
    pq.remove(first)
    remaining: Set[int] = set()
    while not pq.is_empty():
        vid = pq.pop()
        if vid is not None:
            remaining.add(vid)
    assert first not in remaining
    assert len(remaining) <= initial_size - 1
    print("  ✓ Lazy deletion confirmed")


def test_degree_constraint() -> None:
    """Priority queue should respect the degree constraint."""
    print("\nTesting degree constraint...")
    level = create_test_mesh_with_varying_complexity()
    topo = MeshTopology(level.vertices, level.faces)
    pq = VertexPriorityQueue(lambda_weight=0.5, max_degree=4)
    pq.build(level, topo, exclude_boundary=True)

    valid = True
    while not pq.is_empty():
        vid = pq.pop()
        if vid is not None and topo.get_vertex_degree(vid) > 4:
            valid = False
            break
    assert valid
    print("  ✓ Degree constraint enforced")


def test_boundary_exclusion() -> None:
    """Ensure boundary vertices are excluded when requested."""
    print("\nTesting boundary exclusion...")
    level = create_test_mesh_with_varying_complexity()
    topo = MeshTopology(level.vertices, level.faces)
    boundary_vertices = {vid for vid in level.vertices if topo.is_boundary_vertex(vid)}
    pq = VertexPriorityQueue(lambda_weight=0.5, max_degree=12)
    pq.build(level, topo, exclude_boundary=True)

    while not pq.is_empty():
        vid = pq.pop()
        if vid is not None:
            assert vid not in boundary_vertices
    print("  ✓ Boundary vertices excluded")


def test_with_real_mesh() -> None:
    """Run priority queue on real mesh data."""
    print("\nTesting with real mesh...")
    obj_path = "/Users/hedi/LocalFiles/Maps/MAPS/obja/example/suzanne.obj"
    if not os.path.exists(obj_path):
        print(f"  ⚠ Test file not found: {obj_path}")
        return

    level = create_mesh_level_from_obj(obj_path)
    topo = MeshTopology(level.vertices, level.faces)
    independent = select_independent_set_with_priorities(level, topo, 0.5, 12)
    stats = compute_removal_statistics(independent, level)
    assert stats["removal_fraction"] >= 0.04
    print(f"  ✓ Real mesh removal fraction: {stats['removal_percentage']:.1f}%")


def run_all_tests() -> bool:
    """Execute all priority queue tests."""
    print("=" * 60)
    print("MAPS CHUNK 5: Vertex Priority Queue - Test Suite")
    print("=" * 60)

    tests: List[Callable[[], None]] = [
        test_priority_queue_construction,
        test_priority_ordering,
        test_independent_set_selection,
        test_lazy_deletion,
        test_degree_constraint,
        test_boundary_exclusion,
        test_with_real_mesh,
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


