

import numpy as np

# from data_structures import Vertex
# from mesh_topology import MeshTopology

class GeometryUtils:
    def compute_triangle_area(v1, v2, v3):
        e1 = v2 - v1
        e2 = v3 - v1
        cross = np.cross(e1, e2)
        return 0.5 * np.linalg.norm(cross)
    
    def compute_face_normal(v1, v2, v3):
        e1 = v2 - v1
        e2 = v3 - v1
        cross = np.cross(e1, e2)
        norm = np.linalg.norm(cross)
        if norm < 1e-10:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return cross / norm
    
    def compute_area_1ring(self, center, neighbors, topology):
        star_faces = topology.get_star(center.id)
        total_area = 0.0
        for face in star_faces:
            v1 = topology.vertices[face.v1].position()
            v2 = topology.vertices[face.v2].position()
            v3 = topology.vertices[face.v3].position()
            total_area += GeometryUtils.compute_triangle_area(v1, v2, v3)

        return total_area
    
    def estimate_vertex_normal(center, neighbors, topology):
        star_faces = topology.get_star(center.id)
        if not star_faces:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        
        weighted_normal = np.zeros(3, dtype=np.float64)
        for face in star_faces:
            v1 = topology.vertices[face.v1].position()
            v2 = topology.vertices[face.v2].position()
            v3 = topology.vertices[face.v3].position()

            face_normal = GeometryUtils.compute_face_normal(v1, v2, v3)
            area = GeometryUtils.compute_triangle_area(v1, v2, v3)
            weighted_normal += face_normal * area

        norm = np.linalg.norm(weighted_normal)
        if norm < 1e-10:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        
        return weighted_normal / norm
    
    def _build_tangent_basis(normal):
        if abs(normal[2]) < 0.9:
            tan_u = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        else:
            tan_u = np.cross(normal, np.array([1.0, 0.0, 0.0], dtype=np.float64))

        tan_u_norm = np.linalg.norm(tan_u)
        if tan_u_norm < 1e-10:
            tan_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            tan_u = tan_u / tan_u_norm

        tan_v = np.cross(normal, tan_u)
        tan_v_norm = np.linalg.norm(tan_v)
        if tan_v_norm < 1e-10:
            tan_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            tan_v = tan_v / tan_v_norm

        return np.vstack((tan_u, tan_v, normal))
    
    def estimate_curvature(self, center, neighbors, topology):
        if len(neighbors) < 3:
            return 0.0
        
        normal = GeometryUtils.estimate_vertex_normal(center, neighbors, topology)
        center_pos = center.position()

        basis = GeometryUtils._build_tangent_basis(normal)
        tan_u, tan_v = basis[0], basis[1]
        points_uv, heights = list(), list()

        for neighbor in neighbors:
            diff = neighbor.position() - center_pos
            u = float(np.dot(diff, tan_u))
            v = float(np.dot(diff, tan_v))
            h = float(np.dot(diff, normal))
            points_uv.append((u, v))
            heights.append(h)

        design_matrix, rhs = list(), list()
        for (u, v), h in zip(points_uv, heights):
            design_matrix.append([u*u, u*v, v*v])
            rhs.append(h)

        A = np.asarray(design_matrix, dtype=np.float64)
        b = np.asarray(rhs, dtype=np.float64)

        if A.shape[0] < 3:
            return 0.0
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0
        
        a, b_coeff, c = coeffs
        hessian = 2.0 * np.array([[a, b_coeff/2.0], [b_coeff/2.0, c]], dtype=np.float64)
        eigvals = np.linalg.eigvalsh(hessian)
        curvs = float(np.sum(np.abs(eigvals)))
        return curvs
    
    def compute_dihedral_angle(v1, v2, v3, v4):
        n1 = GeometryUtils.compute_face_normal(v1, v2, v3)
        n2 = GeometryUtils.compute_face_normal(v1, v2, v4)
        cos_angle = float(np.clip(np.dot(n1, n2), -1., 1.))
        return float(np.arccos(cos_angle))