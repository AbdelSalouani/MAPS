"""
Minimal implementation of Data Structures
unecessary function were removed
TODO: Test the file
TODO: Add comment if necessary
"""

import numpy as np
import sys
from obja import parse_file


class Vertex:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
    
    def position(self):
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other):
        return np.linalg.norm(self.position() - other.position())
    
class Face:
    def __init__(self, v1, v2, v3, visible = True):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.visible = visible

    def vertices(self):
        return [self.v1, self.v2, self.v3]
    
    def contains_vertex(self, v):
        return v in self.vertices()
    
    def clone(self):
        return Face(self.v1, self.v2, self.v3)
    
class BarycentricCoordinates:
    triangle_id: int
    a: float
    b: float
    g: float

    def is_valid(self):
        coords = (self.a, self.b, self.g)
        return all(-1e-6 <= c <= 1.0 + 1e-6 for c in coords)
    
class MeshLevel:
    def __init__(self, l):
        self.level = l
        self.vertices = dict()
        self.faces = []
        self.removed_vertices = []
        self.barycentric_map = dict()

    def add_vertex(self, v):
        self.vertices[v.id] = v

    def add_face(self, f):
        self.faces.append(f)

    def get_vertex(self, vid):
        return self.vertices[vid]
    
    def num_vertices(self):
        return len(self.vertices)
    
    def num_faces(self):
        return len(self.faces)
    
    def get_active_faces(self):
        return [f for f in self.faces if f.visible]
    
class MeshHierarchy:
    def __init__(self):
        self.levels = []
        self.finest_level = 0
        self.coarsest_level = 0

    def add_level(self, ml):
        self.levels.append(ml)
        if len(self.levels) == 1:
            self.finest_level = ml.level
        self.coarsest_level = ml.level

    def get_level(self, lid):
        idx = self.finest_level - lid
        if idx < 0 or idx >= len(self.levels):
            raise IndexError(f"Level out of range")
        return self.levels[idx]
    
    def num_levels(self):
        return len(self.levels)
    
    def compression_ratio(self):
        if self.num_levels() < 2:
            return 1.
        
        finest = self.get_level(self.finest_level).num_vertices()
        coarsest = self.get_level(self.coarsest_level).num_vertices()
        return finest / coarsest if coarsest > 0 else float("inf")
    
## create_mesh_level_from_obj renamed to obj2mesh
def obj2mesh(obj_path, mlevel = 0):
    model = parse_file(obj_path)
    mesh_level = MeshLevel(mlevel)

    for i, vertex_pos in enumerate(model.vertices):
        vertex = Vertex(i, vertex_pos[0], vertex_pos[1], vertex_pos[2])
        mesh_level.add_vertex(vertex)

    for f in model.faces:
        if f.visible: 
            face = Face(f.a, f.b, f.c)
            mesh_level.add_face(face)
    
    return mesh_level

def print_hierarchy_stats(hierarchy):
    print("RUNNING: print_hierarchy_stats(hierarchy)")