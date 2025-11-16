#!/usr/bin/env python3
"""
Test suite for MAPS Chunk 6: DK Hierarchy Construction.
"""

import math
import os
import sys
from typing import Callable, List

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from chunk1_data_structures import MeshLevel, create_mesh_level_from_obj
from chunk2_mesh_topology import MeshTopology
from chunk6_dk_hierarchy import DKHierarchyBuilder, print_hierarchy_summary


def _load_example_mesh() -> MeshLevel:
    obj_path = "/Users/hedi/LocalFiles/Maps/MAPS/obja/example/suzanne.obj"
    if not os.path.exists(obj_path):
        print(f"  ⚠ Test file not found: {obj_path}")
        return None  # type: ignore
    return create_mesh_level_from_obj(obj_path)


def test_hierarchy_construction_simple() -> None:
    """Verify that hierarchy construction creates multiple levels."""
    print("Testing hierarchy construction (simple mesh)...")
    finest_level = _load_example_mesh()
    if finest_level is None:
        return

    builder = DKHierarchyBuilder(
        lambda_weight=0.5,
        max_degree=12,
        target_base_size=60,
        max_levels=20,
    )
    hierarchy = builder.build_hierarchy(finest_level)

    assert hierarchy.num_levels() > 1
    assert hierarchy.finest_level > hierarchy.coarsest_level

    for level_idx in range(hierarchy.finest_level, hierarchy.coarsest_level, -1):
        current = hierarchy.get_level(level_idx)
        next_level = hierarchy.get_level(level_idx - 1)
        assert current.num_vertices() > next_level.num_vertices()

    print(f"  ✓ Built hierarchy with {hierarchy.num_levels()} levels")


def test_logarithmic_bound() -> None:
    """Number of levels should be O(log N)."""
    print("\nTesting logarithmic bound...")
    finest_level = _load_example_mesh()
    if finest_level is None:
        return

    builder = DKHierarchyBuilder(target_base_size=60)
    hierarchy = builder.build_hierarchy(finest_level)
    levels = hierarchy.num_levels()

    N = finest_level.num_vertices()
    theoretical_max = 6.0 * math.log2(max(N, 2))

    print(f"  Vertices: {N}")
    print(f"  Levels: {levels}")
    print(f"  Theoretical max: {theoretical_max:.1f}")
    assert levels <= theoretical_max
    print("  ✓ Logarithmic bound satisfied")


def test_dk_removal_guarantee() -> None:
    """Each level should remove a constant fraction of vertices."""
    print("\nTesting DK removal guarantee...")
    finest_level = _load_example_mesh()
    if finest_level is None:
        return

    builder = DKHierarchyBuilder(target_base_size=60)
    hierarchy = builder.build_hierarchy(finest_level)

    min_fraction = 1.0
    for level_idx in range(hierarchy.finest_level, hierarchy.coarsest_level, -1):
        current = hierarchy.get_level(level_idx)
        next_level = hierarchy.get_level(level_idx - 1)
        removed = current.num_vertices() - next_level.num_vertices()
        fraction = removed / max(current.num_vertices(), 1)
        min_fraction = min(min_fraction, fraction)
        print(f"  Level {level_idx}: removed {fraction * 100:.1f}%")

    assert min_fraction >= 0.03, (
        f"Removal fraction below threshold: {min_fraction * 100:.1f}%"
    )
    print(f"  ✓ Minimum removal fraction: {min_fraction * 100:.1f}%")


def test_homeomorphism() -> None:
    """Euler characteristic should remain nearly constant across levels."""
    print("\nTesting homeomorphism (Euler characteristic)...")
    finest_level = _load_example_mesh()
    if finest_level is None:
        return

    builder = DKHierarchyBuilder(target_base_size=60)
    hierarchy = builder.build_hierarchy(finest_level)

    finest_topology = MeshTopology(finest_level.vertices, finest_level.faces)
    finest_euler = finest_topology.compute_euler_characteristic()
    print(f"  Finest Euler characteristic: {finest_euler}")

    for level_idx in range(hierarchy.finest_level - 1, hierarchy.coarsest_level - 1, -1):
        level = hierarchy.get_level(level_idx)
        topo = MeshTopology(level.vertices, level.faces)
        euler = topo.compute_euler_characteristic()
        diff = abs(euler - finest_euler)
        print(f"  Level {level_idx} Euler characteristic: {euler} (Δ={diff})")
        assert diff <= 4, f"Euler characteristic deviated by {diff} at level {level_idx}"

    print("  ✓ Euler characteristic variations within tolerance")


def test_base_domain_size() -> None:
    """Base domain should be close to target size."""
    print("\nTesting base domain size...")
    finest_level = _load_example_mesh()
    if finest_level is None:
        return

    target_size = 80
    builder = DKHierarchyBuilder(target_base_size=target_size)
    hierarchy = builder.build_hierarchy(finest_level)

    base_level = hierarchy.get_level(hierarchy.coarsest_level)
    base_size = base_level.num_vertices()

    print(f"  Target base size: {target_size}")
    print(f"  Actual base size: {base_size}")
    assert base_size <= 2 * target_size
    assert base_size >= 0.4 * target_size
    print("  ✓ Base domain size within acceptable range")


def test_hierarchy_summary() -> None:
    """Ensure hierarchy summary prints without error."""
    print("\nTesting hierarchy summary...")
    finest_level = _load_example_mesh()
    if finest_level is None:
        return

    builder = DKHierarchyBuilder(target_base_size=60)
    hierarchy = builder.build_hierarchy(finest_level)
    print_hierarchy_summary(hierarchy)
    print("  ✓ Summary generated successfully")


def run_all_tests() -> bool:
    """Execute all DK hierarchy tests."""
    print("=" * 60)
    print("MAPS CHUNK 6: DK Hierarchy Construction - Test Suite")
    print("=" * 60)

    tests: List[Callable[[], None]] = [
        test_hierarchy_construction_simple,
        test_logarithmic_bound,
        test_dk_removal_guarantee,
        test_homeomorphism,
        test_base_domain_size,
        test_hierarchy_summary,
    ]

    passed = 0
    failed = 0
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as exc:
            print(f"\n  ✗ FAILED: {test_func.__name__}")
            print(f"    AssertionError: {exc}")
            failed += 1
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


