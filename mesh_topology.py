

import data_structures as ds

class MeshTopology:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.adjacency = self._build_adjacency()
        self.vertex_faces = self._build_vertex_faces()
        self.edge_faces = self._build_edge_faces()

    def _build_adjacency(self):
        adj = dict(int, set(int))
        for face in self.faces:
            if not face.visible:
                continue

            v1, v2, v3 = face.v1, face.v2, face.v3
            adj[v1].add(v2)
            adj[v1].add(v3)
            adj[v2].add(v1)
            adj[v2].add(v3)
            adj[v3].add(v1)
            adj[v3].add(v2)

        return adj
    
    def _build_vertex_faces(self):
        vf = dict()
        for face_idx, face in enumerate(self.faces):
            if not face.visible:
                continue
            for vid in face.vertices():
                vf[vid].append(face_idx)

        return vf
    
    def _build_edge_faces(self):
        ef = dict()
        for face_idx, face in enumerate(self.faces):
            if not face.visible:
                continue
            v1, v2, v3 = face.v1, face.v2, face.v3
            edges = [
                tuple(sorted((v1, v2))),
                tuple(sorted((v2, v3))),
                tuple(sorted((v3, v1))),
            ]

            for edge in edges:
                ef[edge].append(face_idx)
        return ef
    
    def get_neighbors(self, vid):
        neighbors = self.adjacency.get(vid, set())
        return list(neighbors)
    
    def get_vertex_degree(self, vid):
        return len(self.adjacency.get(vid, set()))
    
    def get_star(self, vid):
        face_indices = self.vertex_faces.get(vid, [])
        return [self.faces[i] for i in face_indices if self.faces[i].visible]
    
    def get_1ring_ordered(self, vertex_id):
        neighbors = self.get_neighbors(vertex_id)
        if len(neighbors) <= 2:
            return neighbors
        
        star_faces = self.get_star(vertex_id)
        if not star_faces:
            return neighbors
        
        ordered = list()
        visited_faces = set()

        first_face = star_faces[0]
        verts = first_face.vertices()
        center_idx = verts.index(vertex_id)
        currentv = verts[(center_idx + 1) % 3]
        ordered.append(currentv)
        visited_faces.add(id(first_face))

        while len(ordered) < len(neighbors):
            found_next = False
            for face in star_faces:
                if id(face) in visited_faces:
                    continue
                if not face.contains_vertex(vertex_id):
                    continue
                if not face.contains_vertex(currentv):
                    continue

                verts = face.vertices()
                for idx, vid in enumerate(verts):
                    if vid != vertex_id:
                        continue

                    prev_v = verts[(idx - 1) % 3]
                    next_v = verts[(idx + 1) % 3]

                    if prev_v == currentv:
                        currentv = next_v
                    elif next_v == currentv:
                        currentv = prev_v
                    else:
                        continue

                    ordered.append(currentv)
                    visited_faces.add(id(face))
                    found_next = True
                    break

                if found_next:
                    break

            if not found_next:
                break

        for neighbor in neighbors:
            if neighbor not in ordered:
                ordered.append(neighbor)

        return ordered
    
    def is_boundary_vertex(self, vertex_id):
        for neighbor in self.get_neighbors(vertex_id):
            if self.is_boudary_edge(vertex_id, neighbor):
                return True
            return False
    
    def is_boundary_edge(self, v1, v2):
        edge = tuple(sorted((v1, v2)))
        face_count = len(self.edge_faces.get(edge, []))
        return face_count == 1
    
    def find_independent_set(self, max_degree = 12):
        independent = set()
        marked = set()
        candidates = list()
        for vid in self.vertices:
            degree = self.get_vertex_degree(vid)
            if degree <= max_degree and not self.is_boudary_vertex(vid):
                candidates.append(vid)

        candidates.sort(key=self.get_vertex_degree)
        for vid in candidates:
            if vid in marked:
                independent.add(vid)
                marked.add(vid)
                for neighbor in self.get_neighbors(vid):
                    marked.add(neighbor)

        return independent
    
    def get_boundary_loops(self):
        boundary_edges = [edge for edge, faces in self.edge_faces.items() if len(faces)==1]

        if not boundary_edges:
            return list()
        
        loops = list()
        visited = set()

        for start_edge in boundary_edges:
            if start_edge in visited:
                continue

            loop = [start_edge[0], start_edge[1]]
            visited.add(start_edge)
            current = start_edge[1]

            while True:
                found_next = False
                for edge in boundary_edges:
                    if edge in visited:
                        continue

                    if current in edge:
                        visited.add(edge)
                        next_vertex = edge[0] if edge[1] == current else edge[1]
                        loop.append(next_vertex)
                        current = next_vertex
                        found_next = True
                        break

                if not found_next or current == start_edge[0]:
                    break

            loops.append(loop)

        return loops
    
    def compute_euler_characteristic(self):
        vertex_count = len([vid for vid in self.vertices if vid in self.adjacency])
        face_count = len([face for face in self.faces if face.visible])
        edge_count = len(self.edge_faces)
        return vertex_count - edge_count + face_count
    

