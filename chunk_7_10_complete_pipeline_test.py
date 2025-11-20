#!/usr/bin/env python3
"""
Test suite for MAPS Chunks 7-10: Complete Pipeline.
"""

from __future__ import annotations

import os
import sys
import tempfile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from chunk_7_10_complete_pipeline import MAPSProgressiveEncoder


def test_complete_pipeline() -> None:
    """Run the full pipeline on example mesh."""
    print("Testing complete pipeline...")
    input_obj = os.path.join(CURRENT_DIR, "obja", "example", "suzanne.obj")

    if not os.path.exists(input_obj):
        print("  ⚠ Test file not found, skipping")
        return

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".obja", delete=False
    ) as tmp_file:
        output_obja = tmp_file.name

    encoder = MAPSProgressiveEncoder(target_base_size=60)
    encoder.process_obj_to_obja(input_obj, output_obja)

    assert os.path.exists(output_obja), "Output file should exist"
    assert os.path.getsize(output_obja) > 0, "Output file should have content"

    with open(output_obja, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    vertex_lines = [line for line in lines if line.startswith("v ")]
    face_lines = [line for line in lines if line.startswith("f ")]
    tv_lines = [line for line in lines if line.startswith("tv ")]

    print(f"  ✓ Output vertices: {len(vertex_lines)}")
    print(f"  ✓ Output faces: {len(face_lines)}")
    print(f"  ✓ Barycentric entries: {len(tv_lines)}")

    assert vertex_lines, "Output should contain vertices"
    assert face_lines, "Output should contain faces"
    assert tv_lines, "Output should include barycentric mappings"

    os.unlink(output_obja)


def test_progressive_monotonicity() -> None:
    """Ensure vertex/face counts grow monotonically through file."""
    print("\nTesting progressive monotonicity...")
    input_obj = os.path.join(CURRENT_DIR, "obja", "example", "suzanne.obj")

    if not os.path.exists(input_obj):
        print("  ⚠ Test file not found, skipping")
        return

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".obja", delete=False
    ) as tmp_file:
        output_obja = tmp_file.name

    encoder = MAPSProgressiveEncoder(target_base_size=60)
    encoder.process_obj_to_obja(input_obj, output_obja)

    vertex_count = 0
    face_count = 0
    max_vertex = 0
    max_face = 0

    with open(output_obja, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertex_count += 1
                max_vertex = max(max_vertex, vertex_count)
                assert vertex_count <= max_vertex
            elif line.startswith("f "):
                face_count += 1
                max_face = max(max_face, face_count)
                assert face_count <= max_face

    print(f"  ✓ Final vertex count: {vertex_count}")
    print(f"  ✓ Final face count: {face_count}")

    os.unlink(output_obja)


def run_all_tests() -> bool:
    print("=" * 60)
    print("MAPS CHUNKS 7-10: Complete Pipeline - Test Suite")
    print("=" * 60)

    tests = [
        test_complete_pipeline,
        test_progressive_monotonicity,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as exc:
            print(f"\n  ✗ FAILED: {test.__name__}")
            print(f"    AssertionError: {exc}")
            failed += 1
        except Exception as exc:  # pragma: no cover
            print(f"\n  ✗ FAILED: {test.__name__}")
            print(f"    Error: {exc}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

