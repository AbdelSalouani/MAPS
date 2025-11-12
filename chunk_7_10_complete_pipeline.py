#!/usr/bin/env python3
"""
MAPS Implementation - Chunks 7-10: Complete End-to-End Pipeline
Full implementation from OBJ input to progressive OBJA output.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OBJA_DIR = os.path.join(PROJECT_ROOT, "obja")
if OBJA_DIR not in sys.path:
    sys.path.append(OBJA_DIR)

try:
    from obja import Output, parse_file, Face as ObjaFace
except ImportError:  # pragma: no cover
    raise RuntimeError(
        "Failed to import obja module. Ensure /obja directory is available."
    )

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from chunk1_data_structures import (
    BarycentricCoord,
    MeshHierarchy,
    MeshLevel,
    Vertex,
    create_mesh_level_from_obj,
)
from chunk2_mesh_topology import MeshTopology
from chunk6_dk_hierarchy import DKHierarchyBuilder, print_hierarchy_summary


class MAPSProgressiveEncoder:
    """
    Complete MAPS pipeline: OBJ → Hierarchy → Progressive OBJA.
    """

    def __init__(
        self,
        lambda_weight: float = 0.5,
        max_degree: int = 12,
        target_base_size: int = 80,
        max_levels: int = 30,
    ):
        self.builder = DKHierarchyBuilder(
            lambda_weight=lambda_weight,
            max_degree=max_degree,
            target_base_size=target_base_size,
            max_levels=max_levels,
        )

    def process_obj_to_obja(self, input_obj_path: str, output_obja_path: str) -> None:
        """
        Complete pipeline: OBJ file → Progressive OBJA file.
        """
        print("\n" + "=" * 70)
        print("MAPS PROGRESSIVE ENCODING PIPELINE")
        print("=" * 70)

        print("\n[Step 1/4] Loading input OBJ...")
        finest_level = create_mesh_level_from_obj(input_obj_path)
        print(
            f"  Loaded mesh: {finest_level.num_vertices()} vertices, "
            f"{finest_level.num_faces()} faces"
        )

        print("\n[Step 2/4] Building DK hierarchy...")
        hierarchy = self.builder.build_hierarchy(finest_level)
        print_hierarchy_summary(hierarchy)

        print("\n[Step 3/4] Building parameterization...")
        parameterization = self._build_parameterization(hierarchy)
        print(f"  Stored barycentric coords for {len(parameterization)} vertices")

        print("\n[Step 4/4] Generating progressive OBJA...")
        self._generate_obja(
            hierarchy=hierarchy,
            parameterization=parameterization,
            output_path=output_obja_path,
        )

        print("\n" + "=" * 70)
        print(f"SUCCESS: Progressive OBJA written to {output_obja_path}")
        print("=" * 70)

    def _build_parameterization(
        self, hierarchy: MeshHierarchy
    ) -> Dict[int, BarycentricCoord]:
        """
        Build parameterization mapping for vertices removed in the hierarchy.
        """
        parameterization: Dict[int, BarycentricCoord] = {}

        base_level = hierarchy.get_level(hierarchy.coarsest_level)
        base_faces = list(base_level.faces)

        face_vertices: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for face in base_faces:
            v1 = base_level.vertices[face.v1].position()
            v2 = base_level.vertices[face.v2].position()
            v3 = base_level.vertices[face.v3].position()
            face_vertices.append((v1, v2, v3))

        for level_idx in range(
            hierarchy.finest_level, hierarchy.coarsest_level, -1
        ):
            level = hierarchy.get_level(level_idx)
            for vid in level.removed_vertices:
                if vid in parameterization:
                    continue
                vertex = level.vertices.get(vid)
                if vertex is None:
                    continue
                bary = self._project_vertex_to_base(vertex.position(), face_vertices)
                if bary is not None:
                    face_idx, coords = bary
                    parameterization[vid] = BarycentricCoord(
                        triangle_id=face_idx,
                        alpha=float(coords[0]),
                        beta=float(coords[1]),
                        gamma=float(coords[2]),
                    )

        return parameterization

    def _project_vertex_to_base(
        self,
        point: np.ndarray,
        base_triangles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> Optional[Tuple[int, np.ndarray]]:
        best_idx: Optional[int] = None
        best_coords: Optional[np.ndarray] = None
        best_dist = float("inf")

        for idx, (v1, v2, v3) in enumerate(base_triangles):
            bary = self._compute_barycentric(point, v1, v2, v3)
            if bary is None:
                continue
            if np.any(bary < -0.05) or np.any(bary > 1.05):
                continue

            reconstructed = bary[0] * v1 + bary[1] * v2 + bary[2] * v3
            dist = np.linalg.norm(point - reconstructed)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
                best_coords = bary

        if best_idx is not None and best_coords is not None:
            bary = np.clip(best_coords, 0.0, 1.0)
            bary = bary / np.sum(bary)
            return best_idx, bary

        return None

    def _compute_barycentric(
        self, point: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray
    ) -> Optional[np.ndarray]:
        v0 = v2 - v1
        v1_vec = v3 - v1
        v2_vec = point - v1

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1_vec)
        d11 = np.dot(v1_vec, v1_vec)
        d20 = np.dot(v2_vec, v0)
        d21 = np.dot(v2_vec, v1_vec)

        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            return None
        inv_denom = 1.0 / denom
        beta = (d11 * d20 - d01 * d21) * inv_denom
        gamma = (d00 * d21 - d01 * d20) * inv_denom
        alpha = 1.0 - beta - gamma
        return np.array([alpha, beta, gamma], dtype=np.float64)

    def _generate_obja(
        self,
        hierarchy: MeshHierarchy,
        parameterization: Dict[int, BarycentricCoord],
        output_path: str,
    ) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as handle:
            output = Output(handle, random_color=True)

            base_level = hierarchy.get_level(hierarchy.coarsest_level)
            vertex_present: Dict[int, bool] = {}

            for vid in sorted(base_level.vertices.keys()):
                vertex = base_level.vertices[vid]
                output.add_vertex(vid, vertex.position())
                vertex_present[vid] = True

            face_index = 0
            for face in base_level.faces:
                if (
                    vertex_present.get(face.v1)
                    and vertex_present.get(face.v2)
                    and vertex_present.get(face.v3)
                ):
                    obja_face = ObjaFace(face.v1, face.v2, face.v3, face.visible)
                    output.add_face(face_index, obja_face)
                    face_index += 1

            handle.write(
                f"# Base domain: {base_level.num_vertices()} vertices, "
                f"{base_level.num_faces()} faces\n"
            )

            for level_idx in range(
                hierarchy.coarsest_level + 1, hierarchy.finest_level + 1
            ):
                prev_level = hierarchy.get_level(level_idx - 1)
                current_level = hierarchy.get_level(level_idx)
                removed_vertices = prev_level.removed_vertices

                if not removed_vertices:
                    continue

                handle.write(
                    f"# Level {level_idx}: {len(removed_vertices)} new vertices\n"
                )

                for vid in removed_vertices:
                    vertex = current_level.vertices.get(vid)
                    if vertex is None:
                        continue
                    output.add_vertex(vid, vertex.position())
                    vertex_present[vid] = True

                    bary = parameterization.get(vid)
                    if bary:
                        mapped_vid = output.vertex_mapping[vid] + 1
                        handle.write(
                            f"tv {mapped_vid} {bary.triangle_id} "
                            f"{bary.alpha:.6f} {bary.beta:.6f} {bary.gamma:.6f}\n"
                        )

                for face in current_level.faces:
                    if not (
                        vertex_present.get(face.v1)
                        and vertex_present.get(face.v2)
                        and vertex_present.get(face.v3)
                    ):
                        continue

                    if not self._face_exists_in_level(prev_level, face):
                        obja_face = ObjaFace(face.v1, face.v2, face.v3, face.visible)
                        output.add_face(face_index, obja_face)
                        face_index += 1

    def _face_exists_in_level(self, level: MeshLevel, face: Face) -> bool:
        for existing in level.faces:
            if not existing.visible:
                continue
            if {existing.v1, existing.v2, existing.v3} == {face.v1, face.v2, face.v3}:
                return True
        return False


def main_example() -> None:
    """Example usage of complete MAPS pipeline."""
    input_obj = os.path.join(PROJECT_ROOT, "obja", "example", "suzanne.obj")
    output_obja = os.path.join(PROJECT_ROOT, "obja", "example", "suzanne_maps.obja")

    if not os.path.exists(input_obj):
        print(f"Error: Input file not found: {input_obj}")
        return

    encoder = MAPSProgressiveEncoder()
    encoder.process_obj_to_obja(input_obj, output_obja)

    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    print(f"  cd {OBJA_DIR}")
    print("  ./server.py")
    print(
        "  Open browser: http://localhost:8000/?example/suzanne_maps.obja"
    )
    print("=" * 70)


if __name__ == "__main__":
    main_example()

