
import numpy as np
from scipy.spatial import Delaunay
from data_structures import Vertex

def polygon_signed_area(points):
    area = 0.0
    count = len(points)
    if count < 3:
        return 0.0
    for idx in range(count):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % count]
        area += x1*y2 - x2*y1
    return 0.5 * area

def compute_angles_between_neighbors(center, neighbors):
    center_pos = center.position()
    angles = list()
    count = len(neighbors)

    if count == 0:
        return angles
    
    for idx in range(count):
        prev_vertex = neighbors[idx - 1]
        current_vertex = neighbors[idx]
        v_prev = prev_vertex.position() -  center_pos
        v_curr = current_vertex.position() - center_pos

        prev_norm = np.linalg.norm(v_prev)
        curr_norm = np.linalg.norm(v_curr)

        if prev_norm < 1e-12 or curr_norm < 1e-12:
            angles.append(0.0)
            continue

        v_prev /= prev_norm
        v_curr /= curr_norm
        cos_angle = float(np.clip(np.dot(v_prev, v_curr), -1.0, 1.0))
        angle = float(np.arccos(cos_angle))
        angles.append(angle)

    return angles

def conformal_flatten_1ring(center, neighbors):
    count = len(neighbors)
    if count < 3:
        return [(float(idx), 0.0) for idx in range(count)]
    
    angles = compute_angles_between_neighbors(center, neighbors)
    theta_total = sum(angles)

    if theta_total < 1e-6:
        scale = 1.0
    else:
        scale = (2.0 * np.pi) / theta_total

    flattened = list()
    theta = 0.0

    for idx in range(count):
        neighbor = neighbors[idx]
        radius = center.distance_to(neighbor)
        x = radius * np.cos(scale*theta)
        y = radius * np.sin(scale*theta)      
        flattened.append((float(x), float(y)))
        theta += angles[idx]

    return flattened

## maybe we should use the previous function
def conformal_flatten_boundary(center, neighbors):
    count = len(neighbors)
    if count < 3:
        return [(float(idx), 0.0) for idx in range(count)]
      
    angles = compute_angles_between_neighbors(center, neighbors)
    theta_total = sum(angles)

    if theta_total < 1e-6:
        scale = 1.0
    else:
        scale = np.pi / theta_total

    flattened = list()
    theta = 0.0

    for idx in range(count):
        neighbor = neighbors(idx)
        radius = center.distance_to(neighbor)
        x = radius * np.cos(scale*theta)
        y = radius * np.sin(scale*theta)      
        flattened.append((float(x), float(y)))
        theta += angles[idx]

    if flattened:
        flattened[0] = (flattened[0][0], 0.0)
        flattened[-1] = (flattened[-1][0], 0.0)

    return flattened

def retriangulate_hole(flattened_points):
    point_count = len(flattened_points)
    if point_count < 3:
        return list()
    if point_count == 3:
        return [(0, 1, 2)]
    
    triangles = _ear_clipping_triangulation(flattened_points)
    if triangles:
        return triangles
    
    points_array = np.array(flattened_points, dtype=np.float64)
    
    try:
        triangulation = Delaunay(points_array)
    except Exception:
        triangles = list()
        for idx in range(1, point_count -1):
            triangles.append((0, idx, idx+1))
        return triangles
    
    valid_triangles = list()
    for simplex in triangulation.simplices:
        i, j, k = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
        if max(i, j, k) < point_count:
            valid_triangles.append((i, j, k))

    return valid_triangles

def check_triangle_flipping(triangles, points):
    for i, j, k in triangles:
        pi = np.array(points[i], dtype=np.float64)
        pj = np.array(points[j], dtype=np.float64)
        pk = np.array(points[k], dtype=np.float64)
        v1 = pj - pi
        v2 = pk - pi
        signed_area = v1[0] * v2[1] - v1[1] * v2[0]
        if signed_area < 0:
            return True
        
    return False

## better to be implemented right after or before retriangulate_hole
def _ear_clipping_triangulation(points):
    n = len(points)
    if n < 3:
        return list()
    
    indices = list(range(n))
    triangles = list()
    orientation = 1.0 if polygon_signed_area(points) >= 0.0 else -1.0
    guard = 0

    while len(indices) > 3 and guard < n*n:
        ear_found = False
        for idx in range(len(indices)):
            prev_idx = indices[(idx-1) % len(indices)]
            curr_idx = indices[idx]
            next_idx = indices[(idx+1) % len(indices)]

            if not _is_convex_vertex(points[prev_idx], points[curr_idx], points[next_idx], orientation):
                continue

            if _polygon_contains_point(points, indices, prev_idx, curr_idx, next_idx):
                continue

            triangles.append((prev_idx, curr_idx, next_idx))
            del indices[idx]
            ear_found = True
            break

        if not ear_found:
            break
        guard += 1

    if len(indices) == 3:
        triangles.append(tuple(indices))

    return triangles

def _is_convex_vertex(prev_pt, curr_pt, next_pt, orientation):
    ax = curr_pt[0] - prev_pt[0]
    ay = curr_pt[1] - prev_pt[1]
    bx = next_pt[0] - curr_pt[0]
    by = next_pt[1] - curr_pt[1]
    cross = ax*by -ay*bx
    return cross*orientation >= -1e-10

def _polygon_contains_point(points, indices, prev_idx, curr_idx, next_idx):
    a = np.array(points[prev_idx], dtype=np.float64)
    b = np.array(points[curr_idx], dtype=np.float64)
    c = np.array(points[next_idx], dtype=np.float64)

    for idx in indices:
        if idx in(prev_idx, curr_idx, next_idx):
            continue
        p = np.array(points[idx], dtype=np.float64)
        if _point_in_triangle(p, a, b, c):
            return True
        return False

def _point_in_triangle(p, a, b, c):
    v0 = c - a
    v1 = b - a
    v2 = p - a

    denom = v0[0]*v1[1] - v1[0]*v0[1]
    if abs(denom) < 1e-12:
        return False
    
    inv_denom = 1.0 / denom
    u = (v2[0]*v1[1] - v1[0]*v2[1]) * inv_denom
    v = (v0[0]*v2[1] - v2[0]*v0[1]) * inv_denom
    w = 1.0 - u - v
    eps = -1e-9
    return u >= eps and v >= eps and w >= eps