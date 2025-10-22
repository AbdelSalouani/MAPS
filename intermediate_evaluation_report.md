# Progressive 3D Model Transmission - Intermediate Evaluation Report
## MAPS: Multiresolution Adaptive Parameterization of Surfaces

**Group Members:** [Member 1], [Member 2], [Member 3], [Member 4]  
**Date:** October 2025

---

## 1. Context Understanding

### 1.1 MAPS Compression Method

The MAPS algorithm performs mesh compression through three main phases:

**Phase 1: Hierarchical Simplification (Dobkin-Kirkpatrick)**
- Builds a logarithmic hierarchy from fine mesh (level L) to coarse base domain (level 0)
- Removes maximally independent vertices based on geometric criteria (area, curvature)
- Guarantees O(log N) levels by removing ≥1/24 of vertices per level
- Each removed vertex creates a hole that is retriangulated using conformal mapping

**Phase 2: Parameterization Construction**
- Concurrent with simplification, builds mapping Π from original mesh to base domain
- Uses conformal maps (z^a) to flatten 1-rings during vertex removal
- Each vertex gets barycentric coordinates relative to base domain triangle
- Results in smooth parameterization over base domain

**Phase 3: Remeshing**
- Inverse map Π^(-1) samples base domain to generate subdivision connectivity mesh
- Loop subdivision smoothing improves parameterization quality
- Supports adaptive remeshing with error bounds

### 1.2 Inversion Strategy for Progressive Transmission

To create progressive OBJA format, we **invert the MAPS compression**:

```
MAPS Compression:     Dense Mesh → Simplification → Base Domain
Our Transmission:     Base Domain → Progressive Refinement → Dense Mesh
```

**Key Inversion Steps:**

1. **Start with Base Domain (Level 0)**
   - Transmit coarsest mesh vertices and faces
   - This is the initial low-resolution preview

2. **Progressive Vertex Insertion (Levels 1 to L)**
   - Reverse the vertex removal operation
   - For each level, reinsert previously removed vertices
   - Use stored barycentric coordinates to position vertices
   - Retriangulate by reversing the simplification topology

3. **Detail Refinement**
   - Add vertices in order of geometric importance (inverse priority queue)
   - High curvature and large area features added first
   - Maintain subdivision connectivity for smooth refinement

### 1.3 OBJA Format Understanding

The OBJA format extends OBJ with progressive operations:

**Basic Structure:**
```
v x y z          # Add vertex at position (x,y,z)
f v1 v2 v3       # Add face using vertex indices
```

**Progressive Strategy:**
- **Packet 0:** Base domain mesh (minimal vertices + faces)
- **Packet 1-N:** Incremental refinements (new vertices + face updates)
- **Geometry-driven:** High-priority features transmitted early
- **Error-bounded:** Each packet reduces approximation error

**Our OBJA Generation:**
```
Level 0: v (base vertices) → f (base faces)
Level 1: v (new vertices) → f (new faces using old+new vertices)
Level 2: v (refinement) → f (refined triangulation)
...
Level L: v (final details) → f (complete mesh)
```

---

## 2. Project Architecture

### 2.1 System Overview

```
┌─────────────────┐
│   Input OBJ     │
│   (Dense Mesh)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  HIERARCHY BUILDER                       │
│  - DK simplification (L levels)          │
│  - Priority queue (area, curvature)      │
│  - Conformal flattening                  │
│  - Topology tracking                     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  PARAMETERIZATION MODULE                 │
│  - Build mapping Π (fine → coarse)       │
│  - Track barycentric coordinates         │
│  - Store vertex-to-triangle assignment   │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  PROGRESSIVE ENCODER                     │
│  - Invert hierarchy (coarse → fine)      │
│  - Generate OBJA packets                 │
│  - Prioritize by geometry importance     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Output OBJA    │
│  (Progressive)  │
└─────────────────┘
```

### 2.2 Module Dependencies

```
obja_parser.py  ──→  mesh_simplifier.py  ──→  progressive_encoder.py
      ↑                      ↑                         ↑
      │                      │                         │
      └──────────────────────┴─────────────────────────┘
                     geometry_utils.py
                    (conformal maps, 
                     barycentric coords)
```

**Module Descriptions:**

1. **obja_parser.py** (provided base)
   - Parse input OBJ files
   - Generate OBJA output format
   - Manage vertex/face indices

2. **geometry_utils.py**
   - Conformal mapping (z^a flattening)
   - Barycentric coordinate computation
   - Curvature and area estimation
   - Distance metrics for error bounds

3. **mesh_simplifier.py**
   - DK hierarchy construction
   - Independent set selection (priority queue)
   - Vertex removal and retriangulation
   - Topology management

4. **progressive_encoder.py**
   - Invert hierarchy levels
   - Generate OBJA packets
   - Order vertices by importance
   - Compute progressive error metrics

### 2.3 Data Structures

**Mesh Hierarchy Storage:**
```python
class MeshLevel:
    vertices: List[Vector3]      # Vertex positions
    faces: List[Tuple[int,int,int]]  # Face indices
    removed_vertices: List[int]  # Vertices removed at this level
    barycentric_map: Dict[int, Tuple[int,float,float,float]]
        # vertex_id -> (triangle_id, alpha, beta, gamma)
```

**Progressive Packets:**
```python
class OBJAPacket:
    level: int
    new_vertices: List[Vector3]
    new_faces: List[Tuple[int,int,int]]
    cumulative_error: float
```

### 2.4 Testing Strategy

**Unit Tests:**

1. **test_conformal_mapping()**
   - Verify 1-ring flattening preserves angles
   - Check bijection property (no overlaps)
   - Test boundary vertex handling

2. **test_hierarchy_construction()**
   - Verify logarithmic depth O(log N)
   - Check independent set property
   - Validate removal fraction ≥1/24 per level

3. **test_barycentric_coords()**
   - Verify α + β + γ = 1
   - Check point reconstruction accuracy
   - Test triangle containment

4. **test_obja_generation()**
   - Parse generated OBJA back to mesh
   - Verify progressive property (each packet valid)
   - Check index consistency

**Integration Tests:**

1. **test_full_pipeline()**
   - Input: simple mesh (cube, sphere)
   - Output: OBJA file
   - Verify: reconstruction matches original

2. **test_error_bounds()**
   - Measure Hausdorff distance at each level
   - Verify monotonic error decrease
   - Compare to theoretical bounds

3. **test_visualization()**
   - Stream OBJA to provided server.py
   - Visual inspection of progressive quality
   - Performance profiling (bitrate vs distortion)

---

## 3. Project Management

### 3.1 Work Distribution

**Member 1: Hierarchy Construction & Geometry**
- Implement DK simplification algorithm
- Priority queue with area/curvature metrics
- Conformal mapping for 1-ring flattening
- Unit tests for geometric operations

**Member 2: Parameterization & Mapping**
- Build Π mapping (fine → coarse)
- Track barycentric coordinates
- Vertex-to-triangle assignment
- Retriangulation after vertex removal

**Member 3: Progressive Encoding**
- Invert hierarchy (coarse → fine)
- Generate OBJA packets
- Implement prioritization strategy
- Error bound computation

**Member 4: Integration & Testing**
- Integrate all modules
- Comprehensive testing suite
- Visualization with server.py
- Online evaluation (csi.alcouffe.eu)
- Documentation and code cleanup

### 3.2 Timeline and Milestones

**Week 1 (Oct 25 - Oct 31): Foundation**
- Members 1-2: Implement hierarchy construction
- Members 3-4: Set up testing framework and OBJA parser integration
- **Milestone:** Basic DK simplification working

**Week 2 (Nov 1 - Nov 7): Core Implementation**
- Member 1: Complete conformal mapping
- Member 2: Finish parameterization tracking
- Member 3: Start progressive encoder
- Member 4: Unit tests for all modules
- **Milestone:** Full hierarchy with parameterization

**Week 3 (Nov 8 - Nov 14): Integration & Refinement**
- Member 3: Complete OBJA generation
- Member 4: Integration testing
- All: Debug and optimize
- **Milestone:** Working end-to-end pipeline

**Week 4 (Nov 15 - Nov 21): Final Polish**
- All: Performance optimization
- Member 4: Prepare presentation materials
- All: Test on provided datasets
- **Milestone:** Final submission ready

### 3.3 Workload Estimation

| Task | Member | Hours | Status |
|------|--------|-------|--------|
| Conformal mapping implementation | 1 | 8h | In progress |
| DK hierarchy construction | 1 | 10h | Planned |
| Barycentric coord tracking | 2 | 6h | Planned |
| Retriangulation logic | 2 | 8h | Planned |
| Progressive encoder | 3 | 12h | Planned |
| OBJA packet generation | 3 | 6h | Planned |
| Testing framework | 4 | 8h | In progress |
| Integration & debugging | 4 | 10h | Planned |
| **Total** | | **68h** | |

### 3.4 Risk Management

**Technical Risks:**
- **Complexity of conformal mapping:** Mitigation - use simplified piecewise linear approximation
- **Triangle flipping in parameterization:** Mitigation - implement unflipping mechanism from paper
- **Performance on large meshes:** Mitigation - optimize data structures, use spatial indexing

**Project Risks:**
- **Underestimated complexity:** Mitigation - weekly progress checks, adjust scope if needed
- **Integration issues:** Mitigation - early integration testing, clear module interfaces
- **Time constraints:** Mitigation - prioritize core features, have fallback simple implementation

---

## 4. Current Progress & Next Steps

### 4.1 Completed Work
- Paper analysis and algorithm understanding
- Architecture design and module specification
- Testing strategy defined
- Work distribution among team members

### 4.2 Immediate Next Steps (This Week)
1. Set up collaborative Colab notebook
2. Implement basic mesh data structures
3. Begin DK simplification with priority queue
4. Start unit tests for geometric utilities

### 4.3 Expected Challenges
- Handling arbitrary topology meshes (boundaries, genus)
- Efficient point location in irregular triangulation
- Balancing compression ratio vs. quality
- Achieving good bitrate/distortion tradeoff

---

## Appendix: Key Algorithms

### A.1 Vertex Removal Priority

```python
def compute_priority(vertex, mesh, lambda_weight=0.5):
    area = compute_1ring_area(vertex, mesh)
    curvature = estimate_curvature(vertex, mesh)
    
    max_area = max(compute_1ring_area(v, mesh) for v in mesh.vertices)
    max_curv = max(estimate_curvature(v, mesh) for v in mesh.vertices)
    
    priority = lambda_weight * (area / max_area) + \
               (1 - lambda_weight) * (curvature / max_curv)
    
    return priority  # Lower priority = remove first
```

### A.2 Conformal Flattening

```python
def flatten_1ring(center, neighbors):
    """Map 1-ring to plane using conformal map z^a"""
    K = len(neighbors)
    angles = compute_angles(center, neighbors)
    theta_total = sum(angles)
    a = 2 * pi / theta_total
    
    flattened = []
    theta = 0
    for k, neighbor in enumerate(neighbors):
        r = distance(center, neighbor)
        flattened.append((r * cos(a * theta), r * sin(a * theta)))
        theta += angles[k]
    
    return flattened
```

