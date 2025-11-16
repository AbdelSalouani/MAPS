#!/usr/bin/env python3
"""
MAPS Implementation - Chunk 1: Core Data Structures
Provides mesh representation classes for hierarchical simplification.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Vertex:
    """
    Represents a 3D vertex in the mesh.

    Attributes:
        id: Unique vertex identifier (0-indexed).
        x, y, z: 3D coordinates.
    """

    id: int
    x: float
    y: float
    z: float

    def position(self) -> np.ndarray:
        """Return the position as a numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def __repr__(self) -> str:
        return f"Vertex(id={self.id}, pos=[{self.x:.3f}, {self.y:.3f}, {self.z:.3f}])"

    def distance_to(self, other: "Vertex") -> float:
        """Compute Euclidean distance to another vertex."""
        return np.linalg.norm(self.position() - other.position())


@dataclass
class Face:
    """
    Represents a triangular face with three vertex indices.

    Attributes:
        v1, v2, v3: Vertex indices (0-indexed).
        visible: Whether the face is currently active (not deleted).
    """

    v1: int
    v2: int
    v3: int
    visible: bool = True

    def vertices(self) -> List[int]:
        """Return a list of vertex indices."""
        return [self.v1, self.v2, self.v3]

    def contains_vertex(self, vid: int) -> bool:
        """Check if the face contains the given vertex."""
        return vid in (self.v1, self.v2, self.v3)

    def clone(self) -> "Face":
        """Create a copy of this face."""
        return Face(self.v1, self.v2, self.v3, self.visible)

    def __repr__(self) -> str:
        vis = "visible" if self.visible else "hidden"
        return f"Face({self.v1}, {self.v2}, {self.v3}) [{vis}]"


@dataclass
class BarycentricCoord:
    """
    Barycentric coordinates of a vertex within a triangle.

    Used during parameterization to map vertices from fine levels
    to their position in coarser level triangles.

    Attributes:
        triangle_id: Index of containing triangle in coarser level.
        alpha, beta, gamma: Barycentric coordinates (must sum to 1).
    """

    triangle_id: int
    alpha: float
    beta: float
    gamma: float

    def __post_init__(self) -> None:
        """Validate that barycentric coordinates sum to 1."""
        total = self.alpha + self.beta + self.gamma
        assert abs(total - 1.0) < 1e-5, (
            f"Barycentric coords must sum to 1, got {total} "
            f"(α={self.alpha}, β={self.beta}, γ={self.gamma})"
        )

    def is_valid(self) -> bool:
        """Check if coordinates are within valid range [0, 1]."""
        coords = (self.alpha, self.beta, self.gamma)
        return all(-1e-6 <= c <= 1.0 + 1e-6 for c in coords)

    def __repr__(self) -> str:
        return (
            f"Barycentric(tri={self.triangle_id}, "
            f"α={self.alpha:.3f}, β={self.beta:.3f}, γ={self.gamma:.3f})"
        )


class MeshLevel:
    """
    Represents one level in the DK mesh hierarchy.

    Levels are numbered from finest (L) to coarsest (0).
    Each level stores vertices, faces, and information about
    vertices removed when creating the next coarser level.
    """

    def __init__(self, level: int):
        self.level = level
        self.vertices: Dict[int, Vertex] = {}
        self.faces: List[Face] = []
        self.removed_vertices: List[int] = []
        self.barycentric_map: Dict[int, BarycentricCoord] = {}

    def add_vertex(self, vertex: Vertex) -> None:
        """Add a vertex to this level."""
        self.vertices[vertex.id] = vertex

    def add_face(self, face: Face) -> None:
        """Add a face to this level."""
        self.faces.append(face)

    def get_vertex(self, vid: int) -> Optional[Vertex]:
        """Get vertex by ID; returns None if not found."""
        return self.vertices.get(vid)

    def num_vertices(self) -> int:
        """Return the number of vertices at this level."""
        return len(self.vertices)

    def num_faces(self) -> int:
        """Return the number of faces at this level."""
        return len(self.faces)

    def get_active_faces(self) -> List[Face]:
        """Return only visible (non-deleted) faces."""
        return [face for face in self.faces if face.visible]

    def __repr__(self) -> str:
        return (
            f"MeshLevel(level={self.level}, "
            f"vertices={self.num_vertices()}, faces={self.num_faces()})"
        )


class MeshHierarchy:
    """
    Complete DK hierarchy from finest (level L) to coarsest (level 0).

    The hierarchy is constructed through iterative vertex removal,
    guaranteeing O(log N) levels where N is the number of vertices.
    """

    def __init__(self):
        self.levels: List[MeshLevel] = []
        self.finest_level: int = 0
        self.coarsest_level: int = 0

    def add_level(self, level: MeshLevel) -> None:
        """Add a level to the hierarchy."""
        self.levels.append(level)
        if len(self.levels) == 1:
            self.finest_level = level.level
        self.coarsest_level = level.level

    def get_level(self, l: int) -> MeshLevel:
        """Get level by index (0 = coarsest, L = finest)."""
        # Levels are stored from finest to coarsest.
        idx = self.finest_level - l
        if idx < 0 or idx >= len(self.levels):
            raise IndexError(f"Level {l} out of range [0, {self.finest_level}]")
        return self.levels[idx]

    def num_levels(self) -> int:
        """Return total number of levels."""
        return len(self.levels)

    def compression_ratio(self) -> float:
        """Compute compression ratio (finest vertices / coarsest vertices)."""
        if self.num_levels() < 2:
            return 1.0
        finest = self.get_level(self.finest_level).num_vertices()
        coarsest = self.get_level(self.coarsest_level).num_vertices()
        return finest / coarsest if coarsest > 0 else float("inf")

    def __repr__(self) -> str:
        return (
            f"MeshHierarchy(levels={self.num_levels()}, "
            f"finest={self.finest_level}, coarsest={self.coarsest_level})"
        )


def create_mesh_level_from_obj(obj_path: str, level: int = 0) -> MeshLevel:
    """
    Load an OBJ file and create a MeshLevel.

    Args:
        obj_path: Path to the OBJ file.
        level: Level number to assign.

    Returns:
        MeshLevel populated with vertices and faces.
    """

    import sys

    sys.path.append("/Users/hedi/LocalFiles/Maps/MAPS/obja")
    from obja import parse_file  # type: ignore

    model = parse_file(obj_path)
    mesh_level = MeshLevel(level)

    # Add vertices (convert from 1-indexed to 0-indexed).
    for i, vertex_pos in enumerate(model.vertices):
        vertex = Vertex(id=i, x=vertex_pos[0], y=vertex_pos[1], z=vertex_pos[2])
        mesh_level.add_vertex(vertex)

    # Add faces (convert from 1-indexed to 0-indexed).
    for face_obj in model.faces:
        if face_obj.visible:
            face = Face(
                v1=face_obj.a,
                v2=face_obj.b,
                v3=face_obj.c,
                visible=True,
            )
            mesh_level.add_face(face)

    return mesh_level


def print_hierarchy_stats(hierarchy: MeshHierarchy) -> None:
    """Print statistics about the mesh hierarchy."""
    print("\n=== Mesh Hierarchy Statistics ===")
    print(f"Total levels: {hierarchy.num_levels()}")
    print(f"Compression ratio: {hierarchy.compression_ratio():.2f}x")
    print("\nLevel breakdown:")
    for l in range(hierarchy.finest_level, hierarchy.coarsest_level - 1, -1):
        level = hierarchy.get_level(l)
        print(
            f"  Level {l:2d}: {level.num_vertices():6d} vertices, "
            f"{level.num_faces():6d} faces"
        )


