
import numpy as np
import heapq

from data_structures import MeshLevel
from mesh_topology import MeshTopology
from geometry_utils import GeometryUtils

class VertexPriorityQueue:
    def __init__(self, lambda_weight=0.5, max_degree=12):
        self.lambda_weight = float(np.clip(lambda_weight, 0.0, 1.0))
        self.max_degree = max_degree
        self.heap = list()
        self.valid_vertices = set()
        self.removed_vertices = set()

    def build(self, mesh_level, topology, exclude_boundary=True):
        self.heap.clear()
        self.valid_vertices.clear()
        self.removed_vertices.clear()

        geom_utils = GeometryUtils()

        candidate_data = list()
        max_area = 0.0
        max_curvature = 0.0

        for vid, vertex in mesh_level.vertices.items():
            degree = topology.get_vertex_degree(vid)
            if degree > self.max_degree:
                continue
            if exclude_boundary and topology.is_boundary_vertex(vid):
                continue

            neighbor_ids = topology.get_neighbors(vid)
            if len(neighbor_ids) < 3:
                continue

            neighbors = [mesh_level.vertices[nid] for nid in neighbor_ids]
            area = geom_utils.compute_area_1ring(vertex, neighbors, topology)
            curvature = geom_utils.estimate_curvature(vertex, neighbors, topology)

            max_area = max(max_area, area)
            max_curvature = max(max_curvature, curvature)
            candidate_data.append((vid, area, curvature))

        if max_area < 1e-12:
            max_area = 1.0
        if max_curvature < 1e-12:
            max_curvature = 1.0

        for vid, area, curvature in candidate_data:
            priority = self._compute_priority(area, curvature, max_area, max_curvature)
            heapq.heappush(self.heap, (priority, vid))
            self.valid_vertices.add(vid)
        
    def _compute_priority(self, area, curvature, max_area, max_curvature):
        norm_area = area / max_area
        norm_curvature = curvature / max_curvature
        priority = (self.lambda_weight*norm_area + (1.0-self.lambda_weight) * norm_curvature)
        
        return float(priority)
    
    def pop(self):
        while self.heap:
            priority, vid = heapq.heappop(self.heap)
            if vid in self.valid_vertices and vid not in self.removed_vertices:
                self.removed_vertices.add(vid)
                return vid
            
        return None
    
    def remove(self, vertex_id):
        self.removed_vertices.add(vertex_id)
        self.valid_vertices.discard(vertex_id)

    def is_valid(self, vertex_id):
        return vertex_id in self.valid_vertices and vertex_id not in self.removed_vertices
    
    def size(self):
        return len(self.valid_vertices-self.removed_vertices)
    
    def is_empty(self):
        return self.size() == 0
    
def select_independent_set_with_priorities(mesh_level, topology, lambda_weight=0.5, max_degree=12, exclude_boundary=True):
    pq = VertexPriorityQueue(lambda_weight, max_degree)
    pq.build(mesh_level, topology, exclude_boundary)
    independent_set = set()
    blocked = set()
    
    while not pq.is_empty():
        vid = pq.pop()
        if vid is None:
            break
        
        if vid in blocked:
            continue
        independent_set.add(vid)
        blocked.add(vid)
        
        for neighbor in topology.get_neighbors(vid):
            blocked.add(neighbor)
            pq.remove(neighbor)
    return independent_set

def compute_removel_statistics(independent_set, mesh_level):
    pass