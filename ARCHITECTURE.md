# Wiggle Rendering Architecture

## Vision: Figma-style Infinite Canvas

The most performant long-term WebGPU approach is a **retained-mode, compute-driven, tile-based vector renderer** (Vello-style), with aggressive caching of scene encoding + GPU work and minimal per-frame CPU submission.

---

## Architecture Layers

### 1. Retained Scene Graph → Display List Encoding (CPU/WASM)

Don't rebuild triangles every frame. Instead, encode the document into a compact stream of draw ops (paths, strokes, images, text, clips, transforms):

- Edits only touch small parts of the stream
- Multi-thread encoding (workers) possible
- Upload efficiently to GPU

**Vello's approach:** Scene fragments, variable-length path encoding, multistream encoding so transforms/clips/groups are GPU-friendly and parallelizable.

### 2. GPU Compute Pipeline (Decode → Stroke → Bin → Raster)

High-performance pipeline stages:

1. **Decode & flatten curves** (adaptive subdivision)
2. **Stroke expansion** (joins/caps/dashes)
3. **Binning into tiles** (commonly 16×16 px)
4. **Per-tile instruction list** generation
5. **Raster/coverage + shading** (gradients, images, blend)

The "tile VM" model is excellent for:

- Huge documents
- Constant pan/zoom
- Partial invalidation (dirty tiles)
- Lots of clipping and overlapping layers

### 3. Caching Strategy

Cache **work**, not pixels:

| Cache Level   | What to Cache                                                   |
| ------------- | --------------------------------------------------------------- |
| Scene         | Encoded scene fragments per node/subtree                        |
| GPU Resources | Image atlases, gradient textures, glyph atlases                 |
| Submission    | Compute shader dispatch for geometry generation                 |
| Rendering     | Dirty-rect / dirty-tile (only re-run where edits affect output) |

---

## Current Implementation Status

### ✅ Already in Place

- [x] Retained scene graph (shapes array in Zig WASM)
- [x] Dirty flag system for UI sync (`shapes_dirty` for layers panel updates)
- [x] Basic shape persistence (geometry regenerated per frame)
- [x] 4x MSAA antialiasing
- [x] Viewport culling (skip off-screen shapes)
- [x] Undo/redo history stack
- [x] Optimized vertex buffer upload (only upload used bytes, not full buffer)
- [x] **Separated scene vs UI overlay geometry** (scene_vertex_count, ui_vertex_count tracking)
- [x] **Compute shaders for GPU geometry generation** (curve flattening, stroke expansion)
- [x] **Per-shape styles** (fill color, stroke color, stroke width, corner radius)
- [x] **Pen tool for arbitrary paths** (line segments, closed/open paths)
- [x] **Path rendering via GPU compute** (separate compute pipeline for paths)
- [x] **Bezier curve support in pen tool** (click+drag to create smooth curves)
- [x] **Adaptive bezier subdivision** (4-128 segments based on curvature + zoom)
- [x] **Proper line segment joins** (miter joins at path vertices)
- [x] **Debug wireframe mode** (W key toggles wireframe rendering)
- [x] **Auto-select after creation** (shapes/paths selected + switch to select tool)
- [x] **Path bounding box** (selected paths show bounds including control points)
- [x] **Fragment shader antialiasing** (fwidth/smoothstep edge softening)
- [x] **Stroke styling** (caps: butt/round/square, joins: miter/round/bevel)
- [x] **Dashed strokes** (dash length + gap parameters)
- [x] **Layer ordering** (shapes and paths render in correct z-order via shape_index)
- [x] **Nested groups** (arbitrary hierarchy depth, recursive bounds/movement)
- [x] **Group selection** (clicking child selects parent group)
- [x] **Object-level transforms** (Mat3x2 transform matrices with hierarchy propagation)
- [x] **Rotation handle UI** (circular handle above shape for interactive rotation)
- [x] **Rotated selection overlay** (bounding box and handles rotate with shape)
- [x] **Proper shape/path deletion** (cleans up path data, updates indices, shifts multi_selection)
- [x] **Pointer capture for dragging** (drag operations work across entire screen, not just canvas)

### ❌ What NOT to do long-term

- Pure CPU tessellation to triangles every frame (becomes CPU-bound)
- Stencil-then-cover everywhere (pass/overdraw heavy at scale)

---

## Improvement Roadmap

### Architectural Note: View-Dependent Geometry

~~The current implementation bakes **zoom and pan into vertex positions** at tessellation time.~~ **FIXED!**

**Current approach (after Phase 1 optimization):**

1. **Vertex positions**: Stored in **world space**
2. **Zoom/pan transform**: Applied in **vertex shader** via uniforms
3. **Stroke widths**: Stored per-shape in world space, scale with zoom (thicker when zoomed in)
4. **Circle segments**: Still zoom-dependent (more segments at higher zoom for quality)

**Benefits of shader-based transform:**

- **Pan changes require NO vertex regeneration** - only uniform update
- **Zoom changes still regenerate vertices** due to stroke width and segment count dependencies
- Foundation laid for future geometry caching

### Phase 1: Low-Hanging Fruit ✅ COMPLETE

- [x] **Separate scene vs UI overlay geometry** ✅

  - Scene geometry: shapes (rectangles, ellipses)
  - UI geometry: selection boxes, resize handles, drag previews
  - Tracked via `scene_vertex_count` and `ui_vertex_count`
  - Scene geometry now generated by GPU compute shaders

- [x] **Move pan/zoom to shader uniforms** ✅

  - Vertices stored in world space (`worldToVertex`)
  - Shader applies: `screen_x = (world_x * zoom + pan_x) / aspect_ratio`
  - Uniforms struct extended with zoom, pan_x, pan_y
  - Pan-only changes are now very cheap (uniform update only)

- [x] **Scene vertex caching** ✅

  - Cached scene vertices in `cached_scene_vertices` buffer
  - Cache valid when: no shape changes AND no zoom changes
  - Pan-only changes: copy from cache instead of regenerating
  - Exports: `isSceneCacheValid()`, `getCachedSceneVertexCount()`
  - Zoom changes invalidate cache (stroke widths depend on zoom)

- [x] **Batch similar shapes** ✅
  - Vertices organized in batches: rect fills → ellipse fills → rect strokes → ellipse strokes
  - Each batch tracked via `BatchInfo` struct (start_vertex, vertex_count)
  - Exports: `getBatchRectFillStart/Count`, `getBatchEllipseFillStart/Count`, etc.
  - Improves GPU cache utilization by grouping similar geometry
  - Foundation for future multi-draw-call optimization

### Phase 2: Compute-Based Geometry Generation ✅ COMPLETE

**Replaces RenderBundles** - GPU compute shaders now generate all scene geometry:

- [x] **Compute shader pipeline** ✅

  - Shape descriptors encoded in Zig (`ShapeDescriptor` struct, 64 bytes each)
  - Path descriptors encoded in Zig (`PathDescriptor` struct, 64 bytes each)
  - Uploaded to GPU storage buffer each frame
  - Compute shader dispatches one workgroup per shape

- [x] **Curve flattening on GPU** ✅

  - `flattenEllipseFill()` generates triangle fan from ellipse parameters
  - `flattenRectFill()` generates 2 triangles for rectangles
  - Adaptive LOD: `calcSegments()` scales with screen size (24-128 segments)

- [x] **Stroke expansion on GPU** ✅

  - `expandEllipseStroke()` generates ring geometry with proper normals
  - `expandRectStroke()` generates 4 edge rectangles (24 vertices)
  - Rounded rectangle support via path-based rendering (4 arcs + 4 edges)
  - Stroke width scales with shape (world-space coordinates)

- [x] **Path rendering on GPU** ✅

  - Separate compute pipeline for arbitrary paths (`pathComputeShaderCode`)
  - `PathDescriptor` struct: point_start, point_count, closed, stroke_width, color
  - Flattened points buffer for variable-length path data
  - Adaptive bezier subdivision (4-128 segments based on curvature and zoom)
  - Proper miter joins at path vertices (pre-computed offsets)
  - Supports open and closed paths

- [x] **Bezier curve rendering** ✅

  - Cubic bezier curves with adaptive subdivision
  - Curvature-based LOD: more subdivisions for sharper bends
  - Zoom-aware: higher zoom = more subdivisions for quality
  - Screen-space quality: ~5 world units per segment
  - Control point tangents used for proper endpoint joins

- [x] **Per-shape draw calls** ✅
  - Sparse vertex buffer layout (each shape gets fixed allocation)
  - JavaScript calculates draw info per shape
  - Individual `renderPass.draw()` calls for each shape

**Benefits over RenderBundles:**

- Scene geometry generated entirely on GPU (no CPU tessellation)
- Automatic LOD adjustment based on zoom
- Foundation for future tile binning and advanced GPU rendering

### Phase 3: Tile-Based Rendering

- [x] **Partition canvas into tiles** ✅

  - 16×16 tile grid (MAX_TILES_X × MAX_TILES_Y)
  - TILE_SIZE = 0.25 NDC units (~256px at 1024px canvas)
  - Tile grid covers NDC space (-1 to 1)

- [x] **Track tile dirty state** ✅

  - `tile_dirty[MAX_TILES]` array tracks per-tile dirty state
  - `markShapeTilesDirty()` marks tiles overlapping a shape's bounds
  - Move/resize now uses per-shape tile marking (old + new positions)
  - Tiles cleared after each frame for accurate dirty tracking
  - Exports: `isTileDirty()`, `getDirtyTileCount()`, `clearAllTiles()`
  - Tile stats shown in FPS counter (dirty/total tiles)

- [x] **Tile atlas with MSAA** ✅

  - `tileAtlasMsaaTexture` for MSAA rendering
  - `tileAtlasTexture` as resolve target for sampling
  - Both sized to tile grid dimensions

- [x] **Only re-render affected tiles** ✅

  - Scissor rect rendering for each dirty tile
  - Skip clean tiles entirely
  - Falls back to standard rendering when all tiles dirty

- [x] **Composite shader for atlas blitting** ✅
  - `compositeShaderCode` samples tile atlas texture
  - `compositePipeline` and `compositeBindGroup` for blitting
  - UI overlay drawn on top of composited scene

### Phase 4: Advanced Compute (Vello-style)

- [x] **Basic compute shaders** ✅ (implemented in Phase 2)
- [ ] **Scene encoding as compact binary stream** (future)
- [ ] **Tile binning in compute** (future)
- [ ] **Per-tile rasterization kernel** (future)

---

## Configuration

- **MAX_SHAPES**: 10,000
- **COMPUTE_MAX_SHAPES**: 10,000 (GPU compute limit)
- **COMPUTE_MAX_SEGMENTS**: 128 (per ellipse)
- **PATH_MAX_PATHS**: 128 (maximum paths)
- **PATH_MAX_POINTS**: 256 (points per path)
- **BEZIER_SUBDIVISIONS**: 128 (max subdivisions per bezier curve)
- **MAX_HISTORY**: 100 undo steps
- **TILE_SIZE_PX**: 256 pixels per tile
- **SAMPLE_COUNT**: 4x MSAA

---

## Tool IDs

| ID  | Tool   | Description                    |
| --- | ------ | ------------------------------ |
| 0   | Circle | Create ellipse shapes          |
| 1   | Rect   | Create rectangle shapes        |
| 2   | Select | Select and manipulate objects  |
| 3   | Pen    | Draw paths with bezier support |

---

## Keyboard Shortcuts

| Key         | Action                      |
| ----------- | --------------------------- |
| W           | Toggle wireframe debug mode |
| Delete/Bksp | Delete selected shape       |
| Escape      | Cancel pen tool / deselect  |
| Enter       | Finish pen path             |
| Cmd/Ctrl+Z  | Undo                        |
| Cmd/Ctrl+Y  | Redo                        |
| Cmd/Ctrl+G  | Group selected              |
| Cmd/Ctrl+U  | Ungroup selected            |

---

## Implementation Notes

### Per-Shape Vertex Caching (Phase 1 - Legacy)

Note: This approach was superseded by GPU compute shaders in Phase 2.

```
Shape struct additions:
  - vertices_dirty: bool (set when shape changes)
  - vertex_offset: usize (position in global vertex buffer)
  - vertex_count: usize (number of vertices for this shape)

On shape change:
  1. Mark shape.vertices_dirty = true
  2. In render loop, only regenerate vertices for dirty shapes
  3. Update their region in the vertex buffer
  4. Clear dirty flag
```

### GPU Compute Geometry (Phase 2 - Current)

```
Shape rendering:
  1. Build shape descriptors in Zig (buildShapeDescriptors)
  2. Upload descriptors to GPU storage buffer
  3. Dispatch compute shader (one workgroup per shape)
  4. Each workgroup generates vertices for its shape
  5. Issue per-shape draw calls from sparse vertex buffer

Path rendering:
  1. Build path descriptors in Zig (buildPathDescriptors)
  2. Upload path descriptors and flattened points (with bezier data) to GPU
  3. Dispatch path compute shader (one workgroup per path)
  4. For each segment:
     - Line segments: generate single quad (6 vertices)
     - Bezier segments: subdivide into 16 line segments (96 vertices)
  5. Issue per-path draw calls from path vertex buffer

Pen tool bezier interaction:
  - Click: add point with straight line segment
  - Click+drag: add point with bezier handles (drag sets symmetric control handles)
  - The outgoing handle = anchor + drag_delta
  - The incoming handle = anchor - drag_delta (mirrored)
  - Enter: finish path, auto-select it, switch to select tool

Auto-selection after creation:
  - Shapes (rect/circle): selected immediately, tool switches to select
  - Paths: selected on Enter, tool switches to select
  - getCurrentTool() export syncs tool state to React UI

UI overlay:
  - CPU-generated fresh each frame
  - Selection boxes, resize handles, pen preview
  - Path bounding box includes control points (getPathBounds)

Debug wireframe mode:
  - Toggle with W key
  - Shows triangle edges overlaid on shapes
  - Helps visualize tessellation quality and subdivision density
```

### Shape Data Model

```
Shape struct:
  - shape_type: 0=ellipse, 1=rect, 2=group, 3=path
  - x, y: center position (world space)
  - width, height: half-dimensions
  - fill_enabled, fill_color: fill style
  - stroke_enabled, stroke_color, stroke_width: stroke style
  - stroke_cap: 0=butt, 1=round, 2=square
  - stroke_join: 0=miter, 1=round, 2=bevel
  - dash_length, dash_gap: dashed stroke parameters
  - corner_radius: for rounded rectangles
  - path_index: reference to Path array (for type=3)
  - parent_id: group hierarchy (-1 = no parent)
  - visible: layer visibility

Path struct:
  - points[256]: array of (x,y) anchor point coordinates
  - control_points[256][2]: bezier control handles per point
    - [i][0]: in-handle (from previous segment)
    - [i][1]: out-handle (to next segment)
  - segment_types[256]: type of each segment (line or cubic_bezier)
  - point_count: number of points
  - closed: whether path forms a closed loop
  - stroke_width: line thickness (world space)
  - color: RGBA stroke color

SegmentType enum:
  - 0 = line: straight line to next point
  - 1 = cubic_bezier: cubic bezier curve using control handles

Key exports for paths:
  - finishPenPath(): completes path, auto-selects, switches to select tool
  - getPathBounds(shape_id): returns bounding box including control points
  - getCurrentTool(): returns current tool ID for UI sync
```

### Path GPU Buffer Layout

```
Path points are exported as 7 floats per point:
  [x, y, ctrl_in_x, ctrl_in_y, ctrl_out_x, ctrl_out_y, segment_type]

GPU shader processes segments:
  - For line segments: single quad (6 vertices) with miter joins
  - For bezier segments: 4-128 subdivision steps (adaptive)

Bezier curve evaluation:
  p0 = anchor point i
  c0 = control_out of point i
  c1 = control_in of point i+1
  p1 = anchor point i+1

  B(t) = (1-t)³p0 + 3(1-t)²tc0 + 3(1-t)t²c1 + t³p1

Adaptive subdivision algorithm:
  1. Calculate curvature metric:
     - Ratio of control polygon length to chord length
     - Perpendicular distance of control points from chord
  2. Factor in zoom level (more subdivisions when zoomed in)
  3. Factor in screen-space curve length
  4. Clamp subdivisions to 4-128 range

Miter join algorithm:
  1. Pre-compute perpendicular offsets at each path point
  2. For each point, average incoming and outgoing tangent directions
  3. Calculate miter scale to maintain consistent stroke width
  4. Clamp miter to avoid spikes at sharp angles (max 2x)
  5. All segments sharing a joint use the same offset values
```

### Line Segment Joins

```
Problem: Adjacent line segments with different directions create gaps at joints.

Solution: Pre-compute offsets at each path vertex considering both directions.

getSegmentTangent(point_idx, at_start):
  - For bezier: tangent toward first/from last control point
  - For line: direction of the line segment

getMiterOffset(dir_before, dir_after, thickness):
  - Average the two directions
  - Calculate perpendicular to averaged direction
  - Apply miter scale compensation for angle
  - Clamp to avoid extreme spikes

Rendering flow:
  1. Pre-compute point_offsets[i] for all path points
  2. For each segment, use point_offsets[i] and point_offsets[i+1]
  3. Bezier curves use endpoint offsets and compute internal offsets
```

### Tile-Based Rendering (Phase 3)

```
Tile structure:
  - bounds: (min_x, min_y, max_x, max_y)
  - dirty: bool
  - shapes: list of shape indices that intersect

On shape change:
  1. Find all tiles the shape intersects
  2. Mark those tiles dirty

On render:
  1. For each dirty tile:
     a. Render shapes in that tile to tile texture
     b. Mark tile clean
  2. Composite all tiles to screen
```

### Group Hierarchy

```
Nested groups support:
  - Arbitrary nesting depth (groups can contain groups)
  - parent_id field links child to parent (-1 = root level)
  - Selection: clicking any descendant selects top-level ancestor
  - Movement: moving group recursively moves all descendants
  - Bounds: getGroupBounds() recursively includes nested group bounds
  - Ungrouping: children inherit parent's parent_id (move up one level)

Helper functions:
  - isDescendantOf(shape_index, ancestor_id): check ancestry
  - getTopLevelParent(shape_index): find root ancestor
  - moveGroupDescendants(group_index, dx, dy): recursive move
  - markGroupDescendantTilesDirty(group_index): recursive dirty marking
```

### Object-Level Transforms

```
Transform system (similar to 3D scene graph):
  - Each shape stores: x, y (position), rotation (radians), scale_x, scale_y
  - Mat3x2 affine matrix type for 2D transforms
  - getLocalTransform(): builds matrix from position/rotation/scale
  - getWorldTransform(): recursively multiplies with parent transforms

Mat3x2 structure (column-major):
  [a, c, tx]   stored as [a, b, c, d, tx, ty]
  [b, d, ty]   where columns are (a,b), (c,d), (tx,ty)

Transform composition order: T * R * S
  - Scale applied first (in local space)
  - Then rotation (around local origin)
  - Then translation (to world position)

Hierarchy propagation:
  world_transform = parent.getWorldTransform() * child.getLocalTransform()

GPU pipeline:
  1. Zig computes world transforms per shape via getWorldTransform()
  2. Transform matrix stored in ShapeDescriptor/PathDescriptor (6 floats)
  3. GPU compute shaders apply transforms when writing vertices:
     - Shape shader: transformPoint() + writeVertexTransformed()
     - Path shader: g_path_transform global + automatic transform in writePathVertexFull()

Descriptor sizes:
  - ShapeDescriptor: 80 bytes (includes transform matrix + vec4 alignment)
  - PathDescriptor: 96 bytes (includes transform matrix + extra padding)

Rotation handle UI:
  - Circular handle positioned above shape's top edge (in rotated space)
  - Connected to top-center by a line
  - isOnRotationHandle(): hit tests against rotated handle position
  - Dragging calculates angle delta from shape center to cursor
  - rotation = initial_rotation + atan2(cursor - center) - atan2(start - center)

Rotated selection overlay:
  - All 8 resize handles positioned at rotated corner/edge midpoints
  - Bounding box drawn as 4 line segments (not axis-aligned rectangle)
  - Handle positions calculated via rotatePoint() helper
  - Handles themselves remain axis-aligned squares (only positions rotate)
```

### Shape/Path Deletion

```
When deleting a shape:
  1. If shape_type == 3 (path): remove path data from paths[] array
  2. Update path_index in remaining path shapes (decrement if > deleted)
  3. Shift shapes[] array down
  4. Shift multi_selection[] array down (critical for selection state)
  5. Update parent_id references
  6. Update selected_shape index

This ensures:
  - No ghost objects appearing after delete/create cycles
  - Selection state stays aligned with shape indices
  - Path data doesn't leak or get orphaned
```

---

## References

- [Vello: GPU Vector Rendering](https://github.com/linebender/vello)
- [Raph Levien's Blog on GPU Text/Vector](https://raphlinus.github.io/)
- [Figma's WebGPU Journey](https://www.figma.com/blog/figma-faster/)
- [WebGPU Compute Shaders](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html)
