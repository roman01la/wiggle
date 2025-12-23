// Web Worker for WebGPU 2D rendering with Zig WASM

let canvas = null;
let ctx = null;
let device = null;
let pipeline = null;
let bgPipelineMSAA = null;
let vertexBuffer = null;
let uniformBuffer = null;
let uniformBindGroup = null;
let wasm = null;
let msaaTexture = null;
let msaaTextureView = null;
const SAMPLE_COUNT = 4; // 4x MSAA

// Tile-based rendering
let tileAtlasTexture = null;
let tileAtlasView = null;
let tileAtlasMsaaTexture = null;
let tileAtlasMsaaView = null;
let compositePipeline = null;
let compositeBindGroup = null;
let atlasSampler = null;
const TILE_SIZE_PX = 256; // Tile size in pixels
let tilesX = 0;
let tilesY = 0;
let useTileRendering = true; // Enable tile-based rendering for complex scenes
let debugTiles = false; // Show tile grid and highlight dirty tiles
let wireframeMode = false; // Debug: show path wireframe
let wireframePipeline = null; // Wireframe render pipeline
let pathRenderPipeline = null; // Path-specific pipeline with AA vertex format

// Canvas dimensions
let width = 800;
let height = 600;
let canvasFormat = null;

// Input state
let mouseX = 0;
let mouseY = 0;
let mousePressed = false;

// Post status message to main thread
function postStatus(status) {
  self.postMessage({ type: "status", data: status });
}

function postError(error) {
  self.postMessage({ type: "error", data: error });
}

// WASM import object
const wasmImports = {
  env: {
    jsLog: (ptr, len) => {
      const bytes = new Uint8Array(
        wasm.instance.exports.memory.buffer,
        ptr,
        len
      );
      const message = new TextDecoder().decode(bytes);
      console.log("[Zig]", message);
    },
  },
};

// 2D Shape shader code with analytic anti-aliasing
const shaderCode = `
struct Uniforms {
    time: f32,
    mouse_x: f32,
    mouse_y: f32,
    aspect_ratio: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    // Apply zoom and pan transform, then correct for aspect ratio
    let screen_x = (input.position.x * uniforms.zoom + uniforms.pan_x) / uniforms.aspect_ratio;
    let screen_y = input.position.y * uniforms.zoom + uniforms.pan_y;
    output.position = vec4<f32>(screen_x, screen_y, 0.0, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // For solid geometry (filled shapes), just return color
    // The analytic AA is handled in the path shader for strokes
    return input.color;
}
`;

// Path shader with analytic anti-aliasing for stroked paths
// This shader passes through centerline distance for smooth edge AA
// Also supports dashed strokes via arc length
const pathShaderCode = `
struct Uniforms {
    time: f32,
    mouse_x: f32,
    mouse_y: f32,
    aspect_ratio: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) edge_info: vec2<f32>,  // x: signed distance to centerline (in world units), y: half stroke width
    @location(3) arc_length: f32,        // Accumulated arc length along path
    @location(4) dash_pattern: vec2<f32>, // x: dash_length, y: gap_length (0,0 = solid)
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) edge_info: vec2<f32>,  // Pass through for fragment AA
    @location(2) arc_length: f32,
    @location(3) dash_pattern: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let screen_x = (input.position.x * uniforms.zoom + uniforms.pan_x) / uniforms.aspect_ratio;
    let screen_y = input.position.y * uniforms.zoom + uniforms.pan_y;
    output.position = vec4<f32>(screen_x, screen_y, 0.0, 1.0);
    output.color = input.color;
    // Scale edge info to screen space for proper AA width
    output.edge_info = vec2<f32>(input.edge_info.x * uniforms.zoom, input.edge_info.y * uniforms.zoom);
    output.arc_length = input.arc_length * uniforms.zoom;  // Scale arc length to screen space
    output.dash_pattern = input.dash_pattern * uniforms.zoom;  // Scale dash pattern to screen space
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Analytic anti-aliasing using signed distance to stroke edge
    // edge_info.x = signed distance from centerline (negative inside, positive outside)
    // edge_info.y = half stroke width
    let dist_to_center = abs(input.edge_info.x);
    let half_width = input.edge_info.y;
    
    // Distance to stroke edge (positive = outside stroke)
    let dist_to_edge = dist_to_center - half_width;
    
    // AA width based on screen-space derivatives (~1 pixel)
    // fwidth gives us the rate of change per pixel
    // Must compute ALL fwidth calls before any non-uniform control flow
    let aa_width = fwidth(dist_to_edge) * 1.0;
    let aa_dash = fwidth(input.arc_length) * 1.5;
    
    // Smooth coverage: 1 inside, 0 outside, smooth transition at edge
    let edge_coverage = 1.0 - smoothstep(-aa_width, aa_width, dist_to_edge);
    
    // Dash pattern handling
    let dash_length = input.dash_pattern.x;
    let gap_length = input.dash_pattern.y;
    let pattern_length = dash_length + gap_length;
    
    // Calculate position within dash pattern
    let pattern_pos = input.arc_length % max(pattern_length, 0.001);
    
    // Compute dash coverage: 1 in dash, 0 in gap, smooth at edges
    // For solid lines (pattern_length near 0), dash_coverage will be 1.0
    var dash_coverage = 1.0;
    if (pattern_length > 0.001) {
        if (pattern_pos > dash_length) {
            // In gap region
            dash_coverage = 0.0;
        } else {
            // Smooth transition at dash end
            dash_coverage = 1.0 - smoothstep(dash_length - aa_dash, dash_length + aa_dash, pattern_pos);
        }
        // Smooth transition at dash start (pattern wrap)
        let dist_from_start = pattern_pos;
        dash_coverage = max(dash_coverage, smoothstep(-aa_dash, aa_dash, dist_from_start));
    }
    
    // Combine dash and edge coverage
    let final_coverage = edge_coverage * dash_coverage;
    
    return vec4<f32>(input.color.rgb, input.color.a * final_coverage);
}
`;

// Dashed path shader - extends path shader with dash pattern support
// Uses arc_length passed from vertex shader to compute dash visibility
const dashedPathShaderCode = `
struct Uniforms {
    time: f32,
    mouse_x: f32,
    mouse_y: f32,
    aspect_ratio: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) edge_info: vec2<f32>,  // x: signed distance to centerline, y: half stroke width
    @location(3) arc_length: f32,        // Accumulated arc length along path
    @location(4) dash_pattern: vec2<f32>, // x: dash_length, y: gap_length (0,0 = solid)
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) edge_info: vec2<f32>,
    @location(2) arc_length: f32,
    @location(3) dash_pattern: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let screen_x = (input.position.x * uniforms.zoom + uniforms.pan_x) / uniforms.aspect_ratio;
    let screen_y = input.position.y * uniforms.zoom + uniforms.pan_y;
    output.position = vec4<f32>(screen_x, screen_y, 0.0, 1.0);
    output.color = input.color;
    // Scale to screen space
    output.edge_info = vec2<f32>(input.edge_info.x * uniforms.zoom, input.edge_info.y * uniforms.zoom);
    output.arc_length = input.arc_length * uniforms.zoom;  // Scale arc length to screen space
    output.dash_pattern = input.dash_pattern * uniforms.zoom;  // Scale dash pattern to screen space
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dash_length = input.dash_pattern.x;
    let gap_length = input.dash_pattern.y;
    let pattern_length = dash_length + gap_length;
    
    // Edge anti-aliasing (same as solid path)
    let dist_to_center = abs(input.edge_info.x);
    let half_width = input.edge_info.y;
    let dist_to_edge = dist_to_center - half_width;
    let aa_edge = fwidth(dist_to_edge) * 1.0;
    let edge_coverage = 1.0 - smoothstep(-aa_edge, aa_edge, dist_to_edge);
    
    // Solid line if no dash pattern
    if (pattern_length < 0.001) {
        return vec4<f32>(input.color.rgb, input.color.a * edge_coverage);
    }
    
    // Calculate position within dash pattern
    let pattern_pos = input.arc_length % pattern_length;
    
    // Smooth dash transitions using derivative-based AA
    let aa_dash = fwidth(input.arc_length) * 1.5;
    
    // Compute dash coverage: 1 in dash, 0 in gap, smooth at edges
    var dash_coverage = 1.0;
    if (pattern_pos > dash_length) {
        // In gap region
        dash_coverage = 0.0;
    } else {
        // Smooth transition at dash end
        dash_coverage = 1.0 - smoothstep(dash_length - aa_dash, dash_length + aa_dash, pattern_pos);
    }
    // Smooth transition at dash start (pattern wrap)
    let dist_from_start = pattern_pos;
    dash_coverage = max(dash_coverage, smoothstep(-aa_dash, aa_dash, dist_from_start));
    
    // Combine dash and edge coverage
    let final_coverage = edge_coverage * dash_coverage;
    
    return vec4<f32>(input.color.rgb, input.color.a * final_coverage);
}
`;

// Background shader - white background
const bgShaderCode = `
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0)
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
`;

// Composite shader - blit tile atlas to screen
const compositeShaderCode = `
struct CompositeUniforms {
    viewport_size: vec2<f32>,
    atlas_size: vec2<f32>,
}

@group(0) @binding(0) var atlas_texture: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0)
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0)
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(atlas_texture, atlas_sampler, input.uv);
}
`;

// Debug tile shader - draws tile grid lines and highlights
const debugTileShaderCode = `
struct DebugUniforms {
    viewport_width: f32,
    viewport_height: f32,
    tile_size: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> debug_uniforms: DebugUniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    // Convert pixel coords to NDC
    let ndc_x = (input.position.x / debug_uniforms.viewport_width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (input.position.y / debug_uniforms.viewport_height) * 2.0;
    output.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
`;

// ============================================================================
// Compute Shaders for GPU Geometry Generation
// ============================================================================

// Shape descriptor struct - compact representation for GPU processing
// Each shape is encoded as: type, position, size, color, stroke info
const computeShaderCode = `
// Shape types (matches Zig Shape struct: 0=ellipse, 1=rect, 2=group)
const SHAPE_ELLIPSE: u32 = 0u;
const SHAPE_RECT: u32 = 1u;
const SHAPE_GROUP: u32 = 2u;

// Render modes
const MODE_FILL: u32 = 0u;
const MODE_STROKE: u32 = 1u;

// Shape descriptor - input from CPU (80 bytes)
// Uses a 3x2 transform matrix for position/rotation/scale
struct ShapeDescriptor {
    shape_type: u32,        // 0=ellipse, 1=rect, 2=group
    render_mode: u32,       // 0=fill, 1=stroke
    width: f32,             // half-width (local space)
    height: f32,            // half-height (local space)
    stroke_width: f32,      // stroke thickness (for stroke mode)
    corner_radius: f32,     // corner radius for rounded rectangles
    shape_index: u32,       // original shape index for layer ordering
    _padding0: u32,
    // Transform matrix (3x2 affine): [a, b, c, d, tx, ty] at offset 32
    // Column-major: m[0]=a, m[1]=b, m[2]=c, m[3]=d, m[4]=tx, m[5]=ty
    // Transforms local coords to world: world.x = a*x + c*y + tx, world.y = b*x + d*y + ty
    transform: mat3x2<f32>,  // ends at offset 56
    // Padding for vec4 alignment (offset 64)
    _padding1: u32,
    _padding2: u32,
    color: vec4<f32>,       // RGBA color at offset 64
}

// Uniforms for compute
struct ComputeUniforms {
    zoom: f32,
    aspect_ratio: f32,
    shape_count: u32,
    max_segments: u32,      // max segments per ellipse
}

// Vertex layout: 6 floats per vertex (position.xy + color.rgba)
// Each vertex occupies indices [i*6 .. i*6+5]
const FLOATS_PER_VERTEX: u32 = 6u;

// Apply transform matrix to a point: world = mat * local
fn transformPoint(p: vec2<f32>, mat: mat3x2<f32>) -> vec2<f32> {
    return mat * vec3<f32>(p, 1.0);
}

// Get position from transform matrix
fn getPosition(mat: mat3x2<f32>) -> vec2<f32> {
    return vec2<f32>(mat[2].x, mat[2].y);
}

// Get rotation angle from transform matrix (assumes no shear)
fn getRotation(mat: mat3x2<f32>) -> f32 {
    return atan2(mat[0].y, mat[0].x);
}

// Helper to write a vertex to the output buffer
fn writeVertex(vertex_idx: u32, pos: vec2<f32>, color: vec4<f32>) {
    let base = vertex_idx * FLOATS_PER_VERTEX;
    vertices[base] = pos.x;
    vertices[base + 1u] = pos.y;
    vertices[base + 2u] = color.x;
    vertices[base + 3u] = color.y;
    vertices[base + 4u] = color.z;
    vertices[base + 5u] = color.w;
}

@group(0) @binding(0) var<uniform> uniforms: ComputeUniforms;
@group(0) @binding(1) var<storage, read> shapes: array<ShapeDescriptor>;
@group(0) @binding(2) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(3) var<storage, read_write> vertex_counts: array<atomic<u32>>;

const PI: f32 = 3.14159265359;
const MIN_SEGMENTS: u32 = 24u;
const MAX_SEGMENTS: u32 = 128u;

// Calculate number of segments based on screen size
// Uses higher base multiplier for smoother curves at all scales
fn calcSegments(radius_x: f32, radius_y: f32, zoom: f32) -> u32 {
    let max_radius = max(radius_x, radius_y);
    let screen_size = max_radius * zoom * 2.0;
    // Higher multiplier (500) ensures smooth curves even when scaled up
    let base_segments = u32(screen_size * 500.0);
    return clamp(base_segments, MIN_SEGMENTS, MAX_SEGMENTS);
}

// Flatten ellipse to line segments (curve flattening)
fn flattenEllipseFill(shape_idx: u32, shape: ShapeDescriptor, base_vertex: u32) {
    let segments = calcSegments(shape.width, shape.height, uniforms.zoom);
    let angle_step = (2.0 * PI) / f32(segments);
    
    let rx = shape.width;
    let ry = shape.height;
    // Center in local space is origin, transform it to get world center
    let local_center = vec2<f32>(0.0, 0.0);
    let center = transformPoint(local_center, shape.transform);
    
    // Generate triangle fan
    for (var i: u32 = 0u; i < segments; i = i + 1u) {
        let angle1 = f32(i) * angle_step;
        let angle2 = f32(i + 1u) * angle_step;
        
        // Generate points in local space (centered at origin)
        let local_p1 = vec2<f32>(cos(angle1) * rx, sin(angle1) * ry);
        let local_p2 = vec2<f32>(cos(angle2) * rx, sin(angle2) * ry);
        
        // Transform to world space
        let p1 = transformPoint(local_p1, shape.transform);
        let p2 = transformPoint(local_p2, shape.transform);
        
        let vertex_offset = base_vertex + i * 3u;
        
        // Center vertex
        writeVertex(vertex_offset, center, shape.color);
        // Edge vertices
        writeVertex(vertex_offset + 1u, p1, shape.color);
        writeVertex(vertex_offset + 2u, p2, shape.color);
    }
    
    // Store actual vertex count for this shape
    atomicAdd(&vertex_counts[shape_idx], segments * 3u);
}

// Stroke expansion - convert centerline to filled polygon
fn expandEllipseStroke(shape_idx: u32, shape: ShapeDescriptor, base_vertex: u32) {
    let segments = calcSegments(shape.width, shape.height, uniforms.zoom);
    let angle_step = (2.0 * PI) / f32(segments);
    
    let rx = shape.width;
    let ry = shape.height;
    let thickness = shape.stroke_width;
    
    // Generate stroke ring (two triangles per segment)
    for (var i: u32 = 0u; i < segments; i = i + 1u) {
        let angle1 = f32(i) * angle_step;
        let angle2 = f32(i + 1u) * angle_step;
        
        let cos1 = cos(angle1);
        let sin1 = sin(angle1);
        let cos2 = cos(angle2);
        let sin2 = sin(angle2);
        
        // Outer vertices (local space)
        let local_outer1 = vec2<f32>(cos1 * rx, sin1 * ry);
        let local_outer2 = vec2<f32>(cos2 * rx, sin2 * ry);
        
        // Normal calculation for ellipse (in local space)
        let nx1_unnorm = cos1 / rx;
        let ny1_unnorm = sin1 / ry;
        let len1 = sqrt(nx1_unnorm * nx1_unnorm + ny1_unnorm * ny1_unnorm);
        let nx1 = select(0.0, nx1_unnorm / len1, len1 > 0.0001);
        let ny1 = select(0.0, ny1_unnorm / len1, len1 > 0.0001);
        
        let nx2_unnorm = cos2 / rx;
        let ny2_unnorm = sin2 / ry;
        let len2 = sqrt(nx2_unnorm * nx2_unnorm + ny2_unnorm * ny2_unnorm);
        let nx2 = select(0.0, nx2_unnorm / len2, len2 > 0.0001);
        let ny2 = select(0.0, ny2_unnorm / len2, len2 > 0.0001);
        
        // Inner vertices (offset inward by thickness, in local space)
        let local_inner1 = vec2<f32>(local_outer1.x - nx1 * thickness, local_outer1.y - ny1 * thickness);
        let local_inner2 = vec2<f32>(local_outer2.x - nx2 * thickness, local_outer2.y - ny2 * thickness);
        
        // Transform all vertices to world space
        let outer1 = transformPoint(local_outer1, shape.transform);
        let outer2 = transformPoint(local_outer2, shape.transform);
        let inner1 = transformPoint(local_inner1, shape.transform);
        let inner2 = transformPoint(local_inner2, shape.transform);
        
        let vertex_offset = base_vertex + i * 6u;
        
        // Triangle 1: outer1, outer2, inner1
        writeVertex(vertex_offset, outer1, shape.color);
        writeVertex(vertex_offset + 1u, outer2, shape.color);
        writeVertex(vertex_offset + 2u, inner1, shape.color);
        
        // Triangle 2: inner1, outer2, inner2
        writeVertex(vertex_offset + 3u, inner1, shape.color);
        writeVertex(vertex_offset + 4u, outer2, shape.color);
        writeVertex(vertex_offset + 5u, inner2, shape.color);
    }
    
    atomicAdd(&vertex_counts[shape_idx], segments * 6u);
}

// ============================================================================
// Path-based rounded rectangle rendering (SVG-like approach)
// ============================================================================
// The rounded rectangle is defined as a closed path:
//   - 4 straight edges (between corners)
//   - 4 quarter-circle arcs (at corners)
// 
// Path order (counter-clockwise starting from bottom-right):
//   1. Bottom edge:       (x2-r, y1) → (x1+r, y1)
//   2. Bottom-left arc:   center (x1+r, y1+r), 180° → 270°
//   3. Left edge:         (x1, y1+r) → (x1, y2-r)
//   4. Top-left arc:      center (x1+r, y2-r), 90° → 180°
//   5. Top edge:          (x1+r, y2) → (x2-r, y2)
//   6. Top-right arc:     center (x2-r, y2-r), 0° → 90°
//   7. Right edge:        (x2, y2-r) → (x2, y1+r)
//   8. Bottom-right arc:  center (x2-r, y1+r), 270° → 360°
//
// For FILL: Triangle fan from center to perimeter points
// For STROKE: Quad strip between outer and inner perimeter
// ============================================================================

const ARC_SEGMENTS: u32 = 8u;

// Generate a point on a corner arc (in local space)
fn arcPoint(center: vec2<f32>, radius: f32, angle: f32) -> vec2<f32> {
    return vec2<f32>(center.x + radius * cos(angle), center.y + radius * sin(angle));
}

// Helper to write a transformed vertex
fn writeVertexTransformed(vertex_idx: u32, local_pos: vec2<f32>, color: vec4<f32>, mat: mat3x2<f32>) {
    let world_pos = transformPoint(local_pos, mat);
    writeVertex(vertex_idx, world_pos, color);
}

// Flatten rectangle fill using path-based triangle fan
fn flattenRectFill(shape_idx: u32, shape: ShapeDescriptor, base_vertex: u32) {
    // Rectangle dimensions in local space (centered at origin)
    let x1 = -shape.width;
    let y1 = -shape.height;
    let x2 = shape.width;
    let y2 = shape.height;
    let local_center = vec2<f32>(0.0, 0.0);
    let mat = shape.transform;
    
    // Clamp corner radius
    let max_r = min(shape.width, shape.height);
    let r = min(shape.corner_radius, max_r);
    
    // Simple rectangle (no rounded corners)
    if (r < 0.0001) {
        writeVertexTransformed(base_vertex, vec2<f32>(x1, y1), shape.color, mat);
        writeVertexTransformed(base_vertex + 1u, vec2<f32>(x2, y1), shape.color, mat);
        writeVertexTransformed(base_vertex + 2u, vec2<f32>(x2, y2), shape.color, mat);
        writeVertexTransformed(base_vertex + 3u, vec2<f32>(x1, y1), shape.color, mat);
        writeVertexTransformed(base_vertex + 4u, vec2<f32>(x2, y2), shape.color, mat);
        writeVertexTransformed(base_vertex + 5u, vec2<f32>(x1, y2), shape.color, mat);
        atomicAdd(&vertex_counts[shape_idx], 6u);
        return;
    }
    
    // Corner centers (local space)
    let bl = vec2<f32>(x1 + r, y1 + r);  // bottom-left
    let tl = vec2<f32>(x1 + r, y2 - r);  // top-left
    let tr = vec2<f32>(x2 - r, y2 - r);  // top-right
    let br = vec2<f32>(x2 - r, y1 + r);  // bottom-right
    
    var vi = base_vertex;
    
    // Triangle fan: center → perimeter points (counter-clockwise)
    // Start at bottom-right corner, go around CCW
    var prev = vec2<f32>(x2 - r, y1);  // Start at bottom edge, right side
    
    // Bottom edge: (x2-r, y1) → (x1+r, y1)
    var curr = vec2<f32>(x1 + r, y1);
    writeVertexTransformed(vi, local_center, shape.color, mat);
    writeVertexTransformed(vi + 1u, prev, shape.color, mat);
    writeVertexTransformed(vi + 2u, curr, shape.color, mat);
    vi = vi + 3u;
    prev = curr;
    
    // Bottom-left arc: from 270° to 180° (start from i=1 to avoid degenerate triangle)
    for (var i = 1u; i <= ARC_SEGMENTS; i = i + 1u) {
        let angle = PI * 1.5 - f32(i) * PI * 0.5 / f32(ARC_SEGMENTS);
        curr = arcPoint(bl, r, angle);
        writeVertexTransformed(vi, local_center, shape.color, mat);
        writeVertexTransformed(vi + 1u, prev, shape.color, mat);
        writeVertexTransformed(vi + 2u, curr, shape.color, mat);
        vi = vi + 3u;
        prev = curr;
    }
    
    // Left edge: (x1, y1+r) → (x1, y2-r)
    curr = vec2<f32>(x1, y2 - r);
    writeVertexTransformed(vi, local_center, shape.color, mat);
    writeVertexTransformed(vi + 1u, prev, shape.color, mat);
    writeVertexTransformed(vi + 2u, curr, shape.color, mat);
    vi = vi + 3u;
    prev = curr;
    
    // Top-left arc: from 180° to 90° (start from i=1)
    for (var j = 1u; j <= ARC_SEGMENTS; j = j + 1u) {
        let angle = PI - f32(j) * PI * 0.5 / f32(ARC_SEGMENTS);
        curr = arcPoint(tl, r, angle);
        writeVertexTransformed(vi, local_center, shape.color, mat);
        writeVertexTransformed(vi + 1u, prev, shape.color, mat);
        writeVertexTransformed(vi + 2u, curr, shape.color, mat);
        vi = vi + 3u;
        prev = curr;
    }
    
    // Top edge: (x1+r, y2) → (x2-r, y2)
    curr = vec2<f32>(x2 - r, y2);
    writeVertexTransformed(vi, local_center, shape.color, mat);
    writeVertexTransformed(vi + 1u, prev, shape.color, mat);
    writeVertexTransformed(vi + 2u, curr, shape.color, mat);
    vi = vi + 3u;
    prev = curr;
    
    // Top-right arc: from 90° to 0° (start from i=1)
    for (var k = 1u; k <= ARC_SEGMENTS; k = k + 1u) {
        let angle = PI * 0.5 - f32(k) * PI * 0.5 / f32(ARC_SEGMENTS);
        curr = arcPoint(tr, r, angle);
        writeVertexTransformed(vi, local_center, shape.color, mat);
        writeVertexTransformed(vi + 1u, prev, shape.color, mat);
        writeVertexTransformed(vi + 2u, curr, shape.color, mat);
        vi = vi + 3u;
        prev = curr;
    }
    
    // Right edge: (x2, y2-r) → (x2, y1+r)
    curr = vec2<f32>(x2, y1 + r);
    writeVertexTransformed(vi, local_center, shape.color, mat);
    writeVertexTransformed(vi + 1u, prev, shape.color, mat);
    writeVertexTransformed(vi + 2u, curr, shape.color, mat);
    vi = vi + 3u;
    prev = curr;
    
    // Bottom-right arc: from 0° (360°) to 270° (start from i=1)
    for (var m = 1u; m <= ARC_SEGMENTS; m = m + 1u) {
        let angle = -f32(m) * PI * 0.5 / f32(ARC_SEGMENTS);
        curr = arcPoint(br, r, angle);
        writeVertexTransformed(vi, local_center, shape.color, mat);
        writeVertexTransformed(vi + 1u, prev, shape.color, mat);
        writeVertexTransformed(vi + 2u, curr, shape.color, mat);
        vi = vi + 3u;
        prev = curr;
    }
    
    // Close: the last arc point should be at (x2-r, y1), connect back to start
    // Actually, the bottom-right arc ends at 270° = (x2-r, y1+r) + (0, -r) = (x2-r, y1)
    // which is exactly where we started, so no closing triangle needed
    
    atomicAdd(&vertex_counts[shape_idx], vi - base_vertex);
}

// Expand rectangle stroke using path-based quad strip
fn expandRectStroke(shape_idx: u32, shape: ShapeDescriptor, base_vertex: u32) {
    // Rectangle in local space (centered at origin)
    let x1 = -shape.width;
    let y1 = -shape.height;
    let x2 = shape.width;
    let y2 = shape.height;
    let t = shape.stroke_width;
    let mat = shape.transform;
    
    // Clamp corner radius
    let max_r = min(shape.width, shape.height);
    let r = min(shape.corner_radius, max_r);
    
    // Simple rectangle stroke (no rounded corners)
    if (r < 0.0001) {
        // Top edge
        writeVertexTransformed(base_vertex, vec2<f32>(x1, y2 - t), shape.color, mat);
        writeVertexTransformed(base_vertex + 1u, vec2<f32>(x2, y2 - t), shape.color, mat);
        writeVertexTransformed(base_vertex + 2u, vec2<f32>(x2, y2), shape.color, mat);
        writeVertexTransformed(base_vertex + 3u, vec2<f32>(x1, y2 - t), shape.color, mat);
        writeVertexTransformed(base_vertex + 4u, vec2<f32>(x2, y2), shape.color, mat);
        writeVertexTransformed(base_vertex + 5u, vec2<f32>(x1, y2), shape.color, mat);
        // Bottom edge
        writeVertexTransformed(base_vertex + 6u, vec2<f32>(x1, y1), shape.color, mat);
        writeVertexTransformed(base_vertex + 7u, vec2<f32>(x2, y1), shape.color, mat);
        writeVertexTransformed(base_vertex + 8u, vec2<f32>(x2, y1 + t), shape.color, mat);
        writeVertexTransformed(base_vertex + 9u, vec2<f32>(x1, y1), shape.color, mat);
        writeVertexTransformed(base_vertex + 10u, vec2<f32>(x2, y1 + t), shape.color, mat);
        writeVertexTransformed(base_vertex + 11u, vec2<f32>(x1, y1 + t), shape.color, mat);
        // Left edge
        writeVertexTransformed(base_vertex + 12u, vec2<f32>(x1, y1 + t), shape.color, mat);
        writeVertexTransformed(base_vertex + 13u, vec2<f32>(x1 + t, y1 + t), shape.color, mat);
        writeVertexTransformed(base_vertex + 14u, vec2<f32>(x1 + t, y2 - t), shape.color, mat);
        writeVertexTransformed(base_vertex + 15u, vec2<f32>(x1, y1 + t), shape.color, mat);
        writeVertexTransformed(base_vertex + 16u, vec2<f32>(x1 + t, y2 - t), shape.color, mat);
        writeVertexTransformed(base_vertex + 17u, vec2<f32>(x1, y2 - t), shape.color, mat);
        // Right edge
        writeVertexTransformed(base_vertex + 18u, vec2<f32>(x2 - t, y1 + t), shape.color, mat);
        writeVertexTransformed(base_vertex + 19u, vec2<f32>(x2, y1 + t), shape.color, mat);
        writeVertexTransformed(base_vertex + 20u, vec2<f32>(x2, y2 - t), shape.color, mat);
        writeVertexTransformed(base_vertex + 21u, vec2<f32>(x2 - t, y1 + t), shape.color, mat);
        writeVertexTransformed(base_vertex + 22u, vec2<f32>(x2, y2 - t), shape.color, mat);
        writeVertexTransformed(base_vertex + 23u, vec2<f32>(x2 - t, y2 - t), shape.color, mat);
        atomicAdd(&vertex_counts[shape_idx], 24u);
        return;
    }
    
    // Corner centers
    let bl = vec2<f32>(x1 + r, y1 + r);
    let tl = vec2<f32>(x1 + r, y2 - r);
    let tr = vec2<f32>(x2 - r, y2 - r);
    let br = vec2<f32>(x2 - r, y1 + r);
    
    let inner_r = max(r - t, 0.0);
    var vi = base_vertex;
    
    // Quad strip: for each segment, we have outer and inner points
    // Each quad: outer1, inner1, inner2, outer1, inner2, outer2
    
    var prev_out = vec2<f32>(x2 - r, y1);
    var prev_in = vec2<f32>(x2 - r, y1 + t);
    
    // Bottom edge
    var curr_out = vec2<f32>(x1 + r, y1);
    var curr_in = vec2<f32>(x1 + r, y1 + t);
    writeVertexTransformed(vi, prev_out, shape.color, mat);
    writeVertexTransformed(vi + 1u, prev_in, shape.color, mat);
    writeVertexTransformed(vi + 2u, curr_in, shape.color, mat);
    writeVertexTransformed(vi + 3u, prev_out, shape.color, mat);
    writeVertexTransformed(vi + 4u, curr_in, shape.color, mat);
    writeVertexTransformed(vi + 5u, curr_out, shape.color, mat);
    vi = vi + 6u;
    prev_out = curr_out;
    prev_in = curr_in;
    
    // Bottom-left arc (start from i=1 to avoid degenerate quad)
    for (var i = 1u; i <= ARC_SEGMENTS; i = i + 1u) {
        let angle = PI * 1.5 - f32(i) * PI * 0.5 / f32(ARC_SEGMENTS);
        curr_out = arcPoint(bl, r, angle);
        curr_in = arcPoint(bl, inner_r, angle);
        writeVertexTransformed(vi, prev_out, shape.color, mat);
        writeVertexTransformed(vi + 1u, prev_in, shape.color, mat);
        writeVertexTransformed(vi + 2u, curr_in, shape.color, mat);
        writeVertexTransformed(vi + 3u, prev_out, shape.color, mat);
        writeVertexTransformed(vi + 4u, curr_in, shape.color, mat);
        writeVertexTransformed(vi + 5u, curr_out, shape.color, mat);
        vi = vi + 6u;
        prev_out = curr_out;
        prev_in = curr_in;
    }
    
    // Left edge
    curr_out = vec2<f32>(x1, y2 - r);
    curr_in = vec2<f32>(x1 + t, y2 - r);
    writeVertexTransformed(vi, prev_out, shape.color, mat);
    writeVertexTransformed(vi + 1u, prev_in, shape.color, mat);
    writeVertexTransformed(vi + 2u, curr_in, shape.color, mat);
    writeVertexTransformed(vi + 3u, prev_out, shape.color, mat);
    writeVertexTransformed(vi + 4u, curr_in, shape.color, mat);
    writeVertexTransformed(vi + 5u, curr_out, shape.color, mat);
    vi = vi + 6u;
    prev_out = curr_out;
    prev_in = curr_in;
    
    // Top-left arc (start from j=1 to avoid degenerate quad)
    for (var j = 1u; j <= ARC_SEGMENTS; j = j + 1u) {
        let angle = PI - f32(j) * PI * 0.5 / f32(ARC_SEGMENTS);
        curr_out = arcPoint(tl, r, angle);
        curr_in = arcPoint(tl, inner_r, angle);
        writeVertexTransformed(vi, prev_out, shape.color, mat);
        writeVertexTransformed(vi + 1u, prev_in, shape.color, mat);
        writeVertexTransformed(vi + 2u, curr_in, shape.color, mat);
        writeVertexTransformed(vi + 3u, prev_out, shape.color, mat);
        writeVertexTransformed(vi + 4u, curr_in, shape.color, mat);
        writeVertexTransformed(vi + 5u, curr_out, shape.color, mat);
        vi = vi + 6u;
        prev_out = curr_out;
        prev_in = curr_in;
    }
    
    // Top edge
    curr_out = vec2<f32>(x2 - r, y2);
    curr_in = vec2<f32>(x2 - r, y2 - t);
    writeVertexTransformed(vi, prev_out, shape.color, mat);
    writeVertexTransformed(vi + 1u, prev_in, shape.color, mat);
    writeVertexTransformed(vi + 2u, curr_in, shape.color, mat);
    writeVertexTransformed(vi + 3u, prev_out, shape.color, mat);
    writeVertexTransformed(vi + 4u, curr_in, shape.color, mat);
    writeVertexTransformed(vi + 5u, curr_out, shape.color, mat);
    vi = vi + 6u;
    prev_out = curr_out;
    prev_in = curr_in;
    
    // Top-right arc (start from k=1 to avoid degenerate quad)
    for (var k = 1u; k <= ARC_SEGMENTS; k = k + 1u) {
        let angle = PI * 0.5 - f32(k) * PI * 0.5 / f32(ARC_SEGMENTS);
        curr_out = arcPoint(tr, r, angle);
        curr_in = arcPoint(tr, inner_r, angle);
        writeVertexTransformed(vi, prev_out, shape.color, mat);
        writeVertexTransformed(vi + 1u, prev_in, shape.color, mat);
        writeVertexTransformed(vi + 2u, curr_in, shape.color, mat);
        writeVertexTransformed(vi + 3u, prev_out, shape.color, mat);
        writeVertexTransformed(vi + 4u, curr_in, shape.color, mat);
        writeVertexTransformed(vi + 5u, curr_out, shape.color, mat);
        vi = vi + 6u;
        prev_out = curr_out;
        prev_in = curr_in;
    }
    
    // Right edge
    curr_out = vec2<f32>(x2, y1 + r);
    curr_in = vec2<f32>(x2 - t, y1 + r);
    writeVertexTransformed(vi, prev_out, shape.color, mat);
    writeVertexTransformed(vi + 1u, prev_in, shape.color, mat);
    writeVertexTransformed(vi + 2u, curr_in, shape.color, mat);
    writeVertexTransformed(vi + 3u, prev_out, shape.color, mat);
    writeVertexTransformed(vi + 4u, curr_in, shape.color, mat);
    writeVertexTransformed(vi + 5u, curr_out, shape.color, mat);
    vi = vi + 6u;
    prev_out = curr_out;
    prev_in = curr_in;
    
    // Bottom-right arc (start from m=1 to avoid degenerate quad)
    for (var m = 1u; m <= ARC_SEGMENTS; m = m + 1u) {
        let angle = -f32(m) * PI * 0.5 / f32(ARC_SEGMENTS);
        curr_out = arcPoint(br, r, angle);
        curr_in = arcPoint(br, inner_r, angle);
        writeVertexTransformed(vi, prev_out, shape.color, mat);
        writeVertexTransformed(vi + 1u, prev_in, shape.color, mat);
        writeVertexTransformed(vi + 2u, curr_in, shape.color, mat);
        writeVertexTransformed(vi + 3u, prev_out, shape.color, mat);
        writeVertexTransformed(vi + 4u, curr_in, shape.color, mat);
        writeVertexTransformed(vi + 5u, curr_out, shape.color, mat);
        vi = vi + 6u;
        prev_out = curr_out;
        prev_in = curr_in;
    }
    
    // Bottom-right arc ends at (x2-r, y1) which is where we started, so no closing quad needed
    
    atomicAdd(&vertex_counts[shape_idx], vi - base_vertex);
}

// Main compute entry point - one workgroup per shape
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let shape_idx = global_id.x;
    if (shape_idx >= uniforms.shape_count) {
        return;
    }
    
    let shape = shapes[shape_idx];
    
    // Skip groups (they don't generate geometry)
    if (shape.shape_type == SHAPE_GROUP) {
        return;
    }
    
    // Calculate base vertex offset for this shape
    // Each shape gets a fixed allocation: MAX_SEGMENTS * 6 vertices (stroke needs most)
    let base_vertex = shape_idx * uniforms.max_segments * 6u;
    
    // Dispatch to appropriate flattener/expander
    if (shape.shape_type == SHAPE_RECT) {
        if (shape.render_mode == MODE_FILL) {
            flattenRectFill(shape_idx, shape, base_vertex);
        } else {
            expandRectStroke(shape_idx, shape, base_vertex);
        }
    } else if (shape.shape_type == SHAPE_ELLIPSE) {
        if (shape.render_mode == MODE_FILL) {
            flattenEllipseFill(shape_idx, shape, base_vertex);
        } else {
            expandEllipseStroke(shape_idx, shape, base_vertex);
        }
    }
}
`;

// ============================================================================
// Path Compute Shader for GPU Path Geometry Generation
// ============================================================================

const pathComputeShaderCode = `
// Path descriptor - input from CPU (80 bytes = 20 floats/u32s)
struct PathDescriptor {
    point_start: u32,       // Start index in path_points buffer
    point_count: u32,       // Number of points
    closed: u32,            // 1 if closed, 0 if open
    stroke_width: f32,      // Stroke thickness
    color: vec4<f32>,       // RGBA color
    stroke_cap: u32,        // 0=butt, 1=round, 2=square
    stroke_join: u32,       // 0=miter, 1=round, 2=bevel
    dash_length: f32,       // Dash length (0 = solid)
    dash_gap: f32,          // Gap length
    shape_index: u32,       // Original shape index for layer ordering
    _padding0: u32,
    // Transform matrix (3x2 affine): transforms path points from local to world space
    transform: mat3x2<f32>,
    // Padding to reach 96 bytes total (aligned to 16 for vec4)
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
    _padding4: u32,
}

// Uniforms for path compute
struct PathComputeUniforms {
    zoom: f32,
    aspect_ratio: f32,
    path_count: u32,
    max_points_per_path: u32,
}

// Segment types
const SEGMENT_LINE: u32 = 0u;
const SEGMENT_CUBIC_BEZIER: u32 = 1u;

// Point data stride: x, y, ctrl_in_x, ctrl_in_y, ctrl_out_x, ctrl_out_y, segment_type
const POINT_STRIDE: u32 = 7u;

// Bezier subdivision steps for curve flattening (max for adaptive)
const BEZIER_SUBDIVISIONS: u32 = 128u;

// Vertex layout: 11 floats per vertex (position.xy + color.rgba + edge_info.xy + arc_length + dash_pattern.xy)
// edge_info.x = signed distance from centerline (for fragment AA)
// edge_info.y = half stroke width (for fragment AA)
// arc_length = cumulative arc length along path (for dash pattern)
// dash_pattern.x = dash length, dash_pattern.y = gap length
const FLOATS_PER_VERTEX: u32 = 11u;

// Miter limit - above this, fall back to bevel join
const MITER_LIMIT: f32 = 4.0;  // Typical SVG default is 4.0

// Cap/join style constants
const CAP_BUTT: u32 = 0u;
const CAP_ROUND: u32 = 1u;
const CAP_SQUARE: u32 = 2u;
const JOIN_MITER: u32 = 0u;
const JOIN_ROUND: u32 = 1u;
const JOIN_BEVEL: u32 = 2u;

// Apply transform matrix to a point: world = mat * local
fn transformPathPoint(p: vec2<f32>, mat: mat3x2<f32>) -> vec2<f32> {
    return mat * vec3<f32>(p, 1.0);
}

// Storage for current path's transform matrix - set at start of main(), used by all vertex writes
// Initialize to identity matrix
var<private> g_path_transform: mat3x2<f32> = mat3x2<f32>(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);

// Write vertex with edge distance info, arc length, and dash pattern for analytic AA
// Automatically applies g_path_transform to convert local positions to world space
fn writePathVertexFull(vertex_idx: u32, local_pos: vec2<f32>, color: vec4<f32>, dist_to_center: f32, half_width: f32, arc_len: f32, dash_len: f32, dash_gap_len: f32) {
    let pos = transformPathPoint(local_pos, g_path_transform);
    let base = vertex_idx * FLOATS_PER_VERTEX;
    path_vertices[base] = pos.x;
    path_vertices[base + 1u] = pos.y;
    path_vertices[base + 2u] = color.x;
    path_vertices[base + 3u] = color.y;
    path_vertices[base + 4u] = color.z;
    path_vertices[base + 5u] = color.w;
    path_vertices[base + 6u] = dist_to_center;  // Signed distance from centerline
    path_vertices[base + 7u] = half_width;       // Half stroke width for AA
    path_vertices[base + 8u] = arc_len;          // Arc length along path
    path_vertices[base + 9u] = dash_len;         // Dash length (0 = solid)
    path_vertices[base + 10u] = dash_gap_len;    // Gap length
}

// Write vertex with transform applied (explicit transform, for cases where g_path_transform isn't set)
fn writePathVertexTransformed(vertex_idx: u32, local_pos: vec2<f32>, color: vec4<f32>, dist_to_center: f32, half_width: f32, arc_len: f32, dash_len: f32, dash_gap_len: f32, mat: mat3x2<f32>) {
    let world_pos = transformPathPoint(local_pos, mat);
    let base = vertex_idx * FLOATS_PER_VERTEX;
    path_vertices[base] = world_pos.x;
    path_vertices[base + 1u] = world_pos.y;
    path_vertices[base + 2u] = color.x;
    path_vertices[base + 3u] = color.y;
    path_vertices[base + 4u] = color.z;
    path_vertices[base + 5u] = color.w;
    path_vertices[base + 6u] = dist_to_center;
    path_vertices[base + 7u] = half_width;
    path_vertices[base + 8u] = arc_len;
    path_vertices[base + 9u] = dash_len;
    path_vertices[base + 10u] = dash_gap_len;
}

// Helper to write a vertex (legacy compatibility - solid lines)
fn writePathVertex(vertex_idx: u32, pos: vec2<f32>, color: vec4<f32>) {
    writePathVertexFull(vertex_idx, pos, color, 0.0, 0.0, 0.0, 0.0, 0.0);
}

// Write vertex with edge distance info for analytic AA (legacy - solid lines)
fn writePathVertexWithEdge(vertex_idx: u32, pos: vec2<f32>, color: vec4<f32>, dist_to_center: f32, half_width: f32) {
    writePathVertexFull(vertex_idx, pos, color, dist_to_center, half_width, 0.0, 0.0, 0.0);
}

@group(0) @binding(0) var<uniform> uniforms: PathComputeUniforms;
@group(0) @binding(1) var<storage, read> paths: array<PathDescriptor>;
@group(0) @binding(2) var<storage, read> path_points: array<f32>;  // Point data with bezier info
@group(0) @binding(3) var<storage, read_write> path_vertices: array<f32>;
@group(0) @binding(4) var<storage, read_write> path_vertex_counts: array<atomic<u32>>;

// Get a point position from the flattened points buffer
fn getPoint(point_start: u32, index: u32) -> vec2<f32> {
    let base = (point_start + index) * POINT_STRIDE;
    return vec2<f32>(path_points[base], path_points[base + 1u]);
}

// Get the incoming control point for a point (used for bezier from previous point)
fn getControlIn(point_start: u32, index: u32) -> vec2<f32> {
    let base = (point_start + index) * POINT_STRIDE;
    return vec2<f32>(path_points[base + 2u], path_points[base + 3u]);
}

// Get the outgoing control point for a point (used for bezier to next point)
fn getControlOut(point_start: u32, index: u32) -> vec2<f32> {
    let base = (point_start + index) * POINT_STRIDE;
    return vec2<f32>(path_points[base + 4u], path_points[base + 5u]);
}

// Get segment type for a point (determines how to connect to next point)
fn getSegmentType(point_start: u32, index: u32) -> u32 {
    let base = (point_start + index) * POINT_STRIDE;
    return u32(path_points[base + 6u]);
}

// Evaluate cubic bezier at parameter t
fn cubicBezier(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;
    
    return p0 * mt3 + p1 * 3.0 * mt2 * t + p2 * 3.0 * mt * t2 + p3 * t3;
}

// Get perpendicular offset for a point given direction
fn getPerpOffset(dir: vec2<f32>, thickness: f32) -> vec2<f32> {
    let len = length(dir);
    if (len < 0.0001) {
        return vec2<f32>(0.0, 0.0);
    }
    let perp = vec2<f32>(-dir.y, dir.x) / len;
    return perp * thickness * 0.5;
}

// Generate a line segment as a quad (2 triangles) - basic version for single segments
fn addLineSegment(base_vertex: ptr<function, u32>, p0: vec2<f32>, p1: vec2<f32>, thickness: f32, color: vec4<f32>) {
    let dir = p1 - p0;
    let len = length(dir);
    if (len < 0.0001) {
        return;
    }
    
    let offset = getPerpOffset(dir, thickness);
    
    let v0 = p0 - offset;
    let v1 = p0 + offset;
    let v2 = p1 + offset;
    let v3 = p1 - offset;
    
    let vi = *base_vertex;
    
    writePathVertex(vi, v0, color);
    writePathVertex(vi + 1u, v1, color);
    writePathVertex(vi + 2u, v2, color);
    writePathVertex(vi + 3u, v0, color);
    writePathVertex(vi + 4u, v2, color);
    writePathVertex(vi + 5u, v3, color);
    
    *base_vertex = vi + 6u;
}

// Calculate how "bent" a bezier curve is - higher values = more curved
fn bezierCurvature(p0: vec2<f32>, c0: vec2<f32>, c1: vec2<f32>, p1: vec2<f32>) -> f32 {
    // Measure total control polygon length vs chord length
    let chord_len = length(p1 - p0);
    let poly_len = length(c0 - p0) + length(c1 - c0) + length(p1 - c1);
    
    if (chord_len < 0.0001) {
        return 1.0; // Degenerate - use max subdivisions
    }
    
    // Ratio of polygon length to chord - 1.0 means straight, higher means more curved
    // A semicircle has ratio of about 1.57 (pi/2)
    // A very sharp curve can have ratio of 3-4+
    let ratio = poly_len / chord_len;
    
    // Also check perpendicular distance of control points from chord
    let line_dir = (p1 - p0) / chord_len;
    let perp = vec2<f32>(-line_dir.y, line_dir.x);
    let d0 = abs(dot(c0 - p0, perp));
    let d1 = abs(dot(c1 - p0, perp));
    let max_perp = max(d0, d1);
    
    // Combine both metrics - ratio captures "S" curves, perp captures bulge
    return max(ratio - 1.0, max_perp / chord_len) * 2.0;
}

// Add a cubic bezier curve with adaptive subdivision and proper joins
fn addBezierCurve(base_vertex: ptr<function, u32>, p0: vec2<f32>, c0: vec2<f32>, c1: vec2<f32>, p1: vec2<f32>, thickness: f32, color: vec4<f32>, zoom: f32) {
    // Use version with default endpoint offsets and solid line (no dash)
    let start_offset = getPerpOffset(c0 - p0, thickness);
    let end_offset = getPerpOffset(p1 - c1, thickness);
    addBezierCurveWithEndOffsets(base_vertex, p0, c0, c1, p1, thickness, color, zoom, start_offset, end_offset, 0.0, 0.0, 0.0);
}

// Add a cubic bezier curve with adaptive subdivision and pre-computed endpoint offsets for proper joins
// Returns the total arc length of the curve
fn addBezierCurveWithEndOffsets(base_vertex: ptr<function, u32>, p0: vec2<f32>, c0: vec2<f32>, c1: vec2<f32>, p1: vec2<f32>, thickness: f32, color: vec4<f32>, zoom: f32, start_offset: vec2<f32>, end_offset: vec2<f32>, arc_start: f32, dash_len: f32, dash_gap: f32) -> f32 {
    // Calculate curvature to determine subdivision count
    let curvature = bezierCurvature(p0, c0, c1, p1);
    
    // Factor in zoom level - more subdivisions when zoomed in
    let zoom_factor = max(1.0, zoom);
    
    // Also factor in curve length for screen-space quality
    let chord_len = length(p1 - p0);
    let screen_len = chord_len * zoom_factor;
    
    // Adaptive subdivisions based on screen-space curve size
    // Base: reasonable minimum, then scale up based on zoom and curvature
    let base_subdivs = 16u;
    let curve_subdivs = u32(min(curvature * 16.0, 24.0)); // More for sharp curves
    let screen_subdivs = u32(min(screen_len / 5.0, 24.0)); // ~5 world units per segment at current zoom
    let subdivisions = base_subdivs + curve_subdivs + screen_subdivs;
    let actual_subdivs = clamp(subdivisions, 4u, BEZIER_SUBDIVISIONS);
    
    // Pre-calculate all points
    var points: array<vec2<f32>, 129>; // max 128 + 1
    
    for (var i: u32 = 0u; i <= actual_subdivs; i = i + 1u) {
        let t = f32(i) / f32(actual_subdivs);
        points[i] = cubicBezier(p0, c0, c1, p1, t);
    }
    
    // Pre-calculate arc lengths at each point
    var arc_lengths: array<f32, 129>;
    arc_lengths[0u] = arc_start;
    for (var i: u32 = 1u; i <= actual_subdivs; i = i + 1u) {
        arc_lengths[i] = arc_lengths[i - 1u] + length(points[i] - points[i - 1u]);
    }
    let total_arc_length = arc_lengths[actual_subdivs] - arc_start;
    
    // Pre-calculate perpendicular offsets at each point using miter join math
    // This ensures both segments sharing a joint use the SAME offset
    var offsets: array<vec2<f32>, 129>;
    
    for (var i: u32 = 0u; i <= actual_subdivs; i = i + 1u) {
        // Use pre-computed offsets at endpoints for proper joins with adjacent segments
        if (i == 0u) {
            offsets[i] = start_offset;
            continue;
        }
        if (i == actual_subdivs) {
            offsets[i] = end_offset;
            continue;
        }
        
        // Get directions of segments before and after this point
        var dir_before = vec2<f32>(1.0, 0.0);
        var dir_after = vec2<f32>(1.0, 0.0);
        
        let d_before = points[i] - points[i - 1u];
        let len_before = length(d_before);
        if (len_before > 0.0001) {
            dir_before = d_before / len_before;
        }
        
        let d_after = points[i + 1u] - points[i];
        let len_after = length(d_after);
        if (len_after > 0.0001) {
            dir_after = d_after / len_after;
        }
        
        // Average the tangent directions for smooth join
        let sum_dir = dir_before + dir_after;
        let sum_len = length(sum_dir);
        
        var perp: vec2<f32>;
        if (sum_len < 0.0001) {
            // Directions are opposite (180 degree turn) - use perpendicular to either
            perp = vec2<f32>(-dir_before.y, dir_before.x);
        } else {
            let avg_dir = sum_dir / sum_len;
            perp = vec2<f32>(-avg_dir.y, avg_dir.x);
        }
        
        // No miter scaling - for smooth curves with many subdivisions, 
        // the angle change is small enough that scaling isn't needed
        // and it causes visible thickening
        offsets[i] = perp * thickness * 0.5; // Half thickness for each side
    }
    
    // Draw segments using pre-computed offsets (same offset at each joint)
    // Include edge distance info for fragment shader anti-aliasing
    let half_width = thickness * 0.5;
    
    for (var i: u32 = 0u; i < actual_subdivs; i = i + 1u) {
        let curr_point = points[i];
        let next_point = points[i + 1u];
        let curr_offset = offsets[i];
        let next_offset = offsets[i + 1u];
        let curr_arc = arc_lengths[i];
        let next_arc = arc_lengths[i + 1u];
        
        // Draw the line segment quad with edge distance for AA
        let vi = *base_vertex;
        let v0 = curr_point - curr_offset;  // left side
        let v1 = curr_point + curr_offset;  // right side
        let v2 = next_point + next_offset;  // right side
        let v3 = next_point - next_offset;  // left side
        
        // Edge distance: negative on left, positive on right
        writePathVertexFull(vi, v0, color, -half_width, half_width, curr_arc, dash_len, dash_gap);
        writePathVertexFull(vi + 1u, v1, color, half_width, half_width, curr_arc, dash_len, dash_gap);
        writePathVertexFull(vi + 2u, v2, color, half_width, half_width, next_arc, dash_len, dash_gap);
        writePathVertexFull(vi + 3u, v0, color, -half_width, half_width, curr_arc, dash_len, dash_gap);
        writePathVertexFull(vi + 4u, v2, color, half_width, half_width, next_arc, dash_len, dash_gap);
        writePathVertexFull(vi + 5u, v3, color, -half_width, half_width, next_arc, dash_len, dash_gap);
        *base_vertex = vi + 6u;
        
        // Write degenerate triangles to maintain vertex count
        for (var j: u32 = 0u; j < 6u; j = j + 1u) {
            writePathVertexFull(*base_vertex, next_point, color, 0.0, half_width, next_arc, dash_len, dash_gap);
            *base_vertex = *base_vertex + 1u;
        }
    }
    
    return total_arc_length;
}

// Get the tangent direction at a point for a given segment
fn getSegmentTangent(point_start: u32, point_idx: u32, at_start: bool, point_count: u32) -> vec2<f32> {
    let seg_type = getSegmentType(point_start, point_idx);
    let p0 = getPoint(point_start, point_idx);
    let next_idx = (point_idx + 1u) % point_count;
    let p1 = getPoint(point_start, next_idx);
    
    if (seg_type == SEGMENT_CUBIC_BEZIER) {
        if (at_start) {
            // Tangent at start of bezier = direction toward first control point
            let c0 = getControlOut(point_start, point_idx);
            let d = c0 - p0;
            let len = length(d);
            if (len > 0.0001) {
                return d / len;
            }
        } else {
            // Tangent at end of bezier = direction from last control point
            let c1 = getControlIn(point_start, next_idx);
            let d = p1 - c1;
            let len = length(d);
            if (len > 0.0001) {
                return d / len;
            }
        }
    }
    
    // For straight lines, tangent is just the line direction
    let d = p1 - p0;
    let len = length(d);
    if (len > 0.0001) {
        return d / len;
    }
    return vec2<f32>(1.0, 0.0);
}

// Calculate miter offset at a joint between two segments
// Returns: (offset_vector, is_bevel) where is_bevel=1.0 means miter limit exceeded
fn getMiterOffsetWithBevel(dir_before: vec2<f32>, dir_after: vec2<f32>, thickness: f32) -> vec3<f32> {
    let sum_dir = dir_before + dir_after;
    let sum_len = length(sum_dir);
    
    var perp: vec2<f32>;
    var miter_scale = 1.0;
    var is_bevel = 0.0;
    
    if (sum_len < 0.0001) {
        // Directions are opposite (180 degree turn) - use perpendicular to either
        perp = vec2<f32>(-dir_before.y, dir_before.x);
    } else {
        let avg_dir = sum_dir / sum_len;
        perp = vec2<f32>(-avg_dir.y, avg_dir.x);
        
        // Calculate miter scale to maintain consistent stroke width at joints
        let perp_before = vec2<f32>(-dir_before.y, dir_before.x);
        let dot_val = dot(perp, perp_before);
        if (abs(dot_val) > 0.1) {
            miter_scale = 1.0 / dot_val;
            // Check miter limit - if exceeded, signal bevel fallback
            if (miter_scale > MITER_LIMIT) {
                is_bevel = 1.0;
                miter_scale = 1.0;  // Use simple perpendicular for bevel
            }
        }
    }
    
    let offset = perp * thickness * 0.5 * miter_scale;
    return vec3<f32>(offset.x, offset.y, is_bevel);
}

// Legacy function for compatibility
fn getMiterOffset(dir_before: vec2<f32>, dir_after: vec2<f32>, thickness: f32) -> vec2<f32> {
    let result = getMiterOffsetWithBevel(dir_before, dir_after, thickness);
    return vec2<f32>(result.x, result.y);
}

// Add a line segment with pre-computed offsets at both ends, including edge info for AA
// Returns the arc length of this segment
fn addLineSegmentWithOffsets(base_vertex: ptr<function, u32>, p0: vec2<f32>, p1: vec2<f32>, offset0: vec2<f32>, offset1: vec2<f32>, color: vec4<f32>, arc_start: f32, dash_len: f32, dash_gap: f32) -> f32 {
    let vi = *base_vertex;
    let half_width0 = length(offset0);
    let half_width1 = length(offset1);
    let segment_length = length(p1 - p0);
    
    let v0 = p0 - offset0;  // left side at p0
    let v1 = p0 + offset0;  // right side at p0
    let v2 = p1 + offset1;  // right side at p1
    let v3 = p1 - offset1;  // left side at p1
    
    let arc_end = arc_start + segment_length;
    
    // Write vertices with edge distance: negative = left side, positive = right side
    writePathVertexFull(vi, v0, color, -half_width0, half_width0, arc_start, dash_len, dash_gap);
    writePathVertexFull(vi + 1u, v1, color, half_width0, half_width0, arc_start, dash_len, dash_gap);
    writePathVertexFull(vi + 2u, v2, color, half_width1, half_width1, arc_end, dash_len, dash_gap);
    writePathVertexFull(vi + 3u, v0, color, -half_width0, half_width0, arc_start, dash_len, dash_gap);
    writePathVertexFull(vi + 4u, v2, color, half_width1, half_width1, arc_end, dash_len, dash_gap);
    writePathVertexFull(vi + 5u, v3, color, -half_width1, half_width1, arc_end, dash_len, dash_gap);
    
    *base_vertex = vi + 6u;
    return segment_length;
}

// Add a round cap at an endpoint
fn addRoundCap(base_vertex: ptr<function, u32>, center: vec2<f32>, dir: vec2<f32>, thickness: f32, color: vec4<f32>, is_start: bool) {
    let half_width = thickness * 0.5;
    let perp = vec2<f32>(-dir.y, dir.x);
    
    // Cap direction: pointing outward from the path
    var cap_dir = dir;
    if (is_start) {
        cap_dir = -dir;  // Start cap points backward
    }
    
    // Generate semicircle using triangle fan
    let segments = 8u;
    let angle_step = 3.14159265359 / f32(segments);
    
    for (var i = 0u; i < segments; i = i + 1u) {
        let angle1 = f32(i) * angle_step - 1.5707963;  // Start at -90 degrees
        let angle2 = f32(i + 1u) * angle_step - 1.5707963;
        
        // Rotate angles based on direction
        let cos_base = cap_dir.x;
        let sin_base = cap_dir.y;
        
        let p1_local = vec2<f32>(cos(angle1), sin(angle1)) * half_width;
        let p2_local = vec2<f32>(cos(angle2), sin(angle2)) * half_width;
        
        // Rotate to align with path direction
        let p1 = center + vec2<f32>(
            p1_local.x * cos_base - p1_local.y * sin_base,
            p1_local.x * sin_base + p1_local.y * cos_base
        );
        let p2 = center + vec2<f32>(
            p2_local.x * cos_base - p2_local.y * sin_base,
            p2_local.x * sin_base + p2_local.y * cos_base
        );
        
        let vi = *base_vertex;
        // Distance from center for AA (0 at center, half_width at edge)
        let d1 = length(p1 - center);
        let d2 = length(p2 - center);
        
        writePathVertexWithEdge(vi, center, color, 0.0, half_width);
        writePathVertexWithEdge(vi + 1u, p1, color, d1, half_width);
        writePathVertexWithEdge(vi + 2u, p2, color, d2, half_width);
        *base_vertex = vi + 3u;
    }
}

// Add a round join between two segments
fn addRoundJoin(base_vertex: ptr<function, u32>, center: vec2<f32>, dir_before: vec2<f32>, dir_after: vec2<f32>, thickness: f32, color: vec4<f32>) {
    let half_width = thickness * 0.5;
    
    // Calculate the angle between the two directions
    let perp_before = vec2<f32>(-dir_before.y, dir_before.x);
    let perp_after = vec2<f32>(-dir_after.y, dir_after.x);
    
    // Determine which side needs the join (outer side of the bend)
    let cross = dir_before.x * dir_after.y - dir_before.y * dir_after.x;
    if (abs(cross) < 0.0001) {
        return;  // Nearly straight, no join needed
    }
    
    // Generate arc on the outer side
    var start_perp = perp_before;
    var end_perp = perp_after;
    if (cross < 0.0) {
        start_perp = -perp_before;
        end_perp = -perp_after;
    }
    
    // Calculate angle span
    let start_angle = atan2(start_perp.y, start_perp.x);
    var end_angle = atan2(end_perp.y, end_perp.x);
    
    // Ensure we go the short way around
    var angle_diff = end_angle - start_angle;
    if (angle_diff > 3.14159265359) {
        angle_diff = angle_diff - 6.28318530718;
    } else if (angle_diff < -3.14159265359) {
        angle_diff = angle_diff + 6.28318530718;
    }
    
    let segments = max(4u, u32(abs(angle_diff) * 4.0));  // More segments for larger angles
    let angle_step = angle_diff / f32(segments);
    
    for (var i = 0u; i < segments; i = i + 1u) {
        let a1 = start_angle + f32(i) * angle_step;
        let a2 = start_angle + f32(i + 1u) * angle_step;
        
        let p1 = center + vec2<f32>(cos(a1), sin(a1)) * half_width;
        let p2 = center + vec2<f32>(cos(a2), sin(a2)) * half_width;
        
        let vi = *base_vertex;
        writePathVertexWithEdge(vi, center, color, 0.0, half_width);
        writePathVertexWithEdge(vi + 1u, p1, color, half_width, half_width);
        writePathVertexWithEdge(vi + 2u, p2, color, half_width, half_width);
        *base_vertex = vi + 3u;
    }
}

// Add a bevel join (simple triangle) between two segments
fn addBevelJoin(base_vertex: ptr<function, u32>, center: vec2<f32>, offset_before: vec2<f32>, offset_after: vec2<f32>, thickness: f32, color: vec4<f32>) {
    let half_width = thickness * 0.5;
    
    // Determine which side needs the bevel
    let cross = offset_before.x * offset_after.y - offset_before.y * offset_after.x;
    if (abs(cross) < 0.0001) {
        return;  // Nearly straight, no join needed
    }
    
    let vi = *base_vertex;
    
    // Create bevel triangle on outer side
    if (cross > 0.0) {
        // Left side is outer
        writePathVertexWithEdge(vi, center, color, 0.0, half_width);
        writePathVertexWithEdge(vi + 1u, center - offset_before, color, half_width, half_width);
        writePathVertexWithEdge(vi + 2u, center - offset_after, color, half_width, half_width);
    } else {
        // Right side is outer
        writePathVertexWithEdge(vi, center, color, 0.0, half_width);
        writePathVertexWithEdge(vi + 1u, center + offset_before, color, half_width, half_width);
        writePathVertexWithEdge(vi + 2u, center + offset_after, color, half_width, half_width);
    }
    
    *base_vertex = vi + 3u;
}

// Main compute entry point - one workgroup per path
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let path_idx = global_id.x;
    if (path_idx >= uniforms.path_count) {
        return;
    }
    
    let path = paths[path_idx];
    if (path.point_count < 2u) {
        return;
    }
    
    // Initialize global transform for this path
    g_path_transform = path.transform;
    
    // Calculate base vertex offset for this path
    // Each path gets allocation for max_points_per_path * BEZIER_SUBDIVISIONS * 12 vertices
    // (6 for quad + 6 for bevel join per sub-segment)
    let base_vertex = path_idx * uniforms.max_points_per_path * BEZIER_SUBDIVISIONS * 12u;
    var current_vertex = base_vertex;
    
    let segment_count = path.point_count - 1u;
    let is_closed = path.closed == 1u;
    
    // Pre-compute perpendicular offsets at each point for proper joins
    // For straight line segments, we need miter joins at vertices
    var point_offsets: array<vec2<f32>, 129>; // max points + 1
    var point_dirs_before: array<vec2<f32>, 129>; // incoming direction at each point
    var point_dirs_after: array<vec2<f32>, 129>; // outgoing direction at each point
    
    for (var i: u32 = 0u; i < path.point_count; i = i + 1u) {
        var dir_before = vec2<f32>(1.0, 0.0);
        var dir_after = vec2<f32>(1.0, 0.0);
        
        // Get direction of incoming segment (segment ending at this point)
        if (i > 0u) {
            // Tangent at end of previous segment
            dir_before = getSegmentTangent(path.point_start, i - 1u, false, path.point_count);
        } else if (is_closed) {
            // For closed paths, incoming segment is the last one
            dir_before = getSegmentTangent(path.point_start, segment_count, false, path.point_count);
        }
        
        // Get direction of outgoing segment (segment starting at this point)
        if (i < segment_count) {
            // Tangent at start of this segment
            dir_after = getSegmentTangent(path.point_start, i, true, path.point_count);
        } else if (is_closed) {
            // For closed paths, outgoing from last point goes to first point
            dir_after = getSegmentTangent(path.point_start, i, true, path.point_count);
        }
        
        // At open path endpoints, use the single segment direction
        if (!is_closed) {
            if (i == 0u) {
                dir_before = dir_after;
            }
            if (i == segment_count) {
                dir_after = dir_before;
            }
        }
        
        point_dirs_before[i] = dir_before;
        point_dirs_after[i] = dir_after;
        point_offsets[i] = getMiterOffset(dir_before, dir_after, path.stroke_width);
    }
    
    // Add start cap for open paths
    if (!is_closed && path.stroke_cap == CAP_ROUND) {
        let start_point = getPoint(path.point_start, 0u);
        let start_dir = point_dirs_after[0u];
        addRoundCap(&current_vertex, start_point, start_dir, path.stroke_width, path.color, true);
    }
    
    // Track arc length for dash pattern
    var arc_length: f32 = 0.0;
    let dash_len = path.dash_length;
    let dash_gap = path.dash_gap;
    
    // Generate segments between consecutive points
    for (var i: u32 = 0u; i < segment_count; i = i + 1u) {
        let p0 = getPoint(path.point_start, i);
        let p1 = getPoint(path.point_start, i + 1u);
        let seg_type = getSegmentType(path.point_start, i);
        
        if (seg_type == SEGMENT_CUBIC_BEZIER) {
            // Get control points: outgoing from p0, incoming to p1
            let c0 = getControlOut(path.point_start, i);
            let c1 = getControlIn(path.point_start, i + 1u);
            arc_length = arc_length + addBezierCurveWithEndOffsets(&current_vertex, p0, c0, c1, p1, path.stroke_width, path.color, uniforms.zoom, point_offsets[i], point_offsets[i + 1u], arc_length, dash_len, dash_gap);
        } else {
            // Line segment with pre-computed offsets for proper joins
            arc_length = arc_length + addLineSegmentWithOffsets(&current_vertex, p0, p1, point_offsets[i], point_offsets[i + 1u], path.color, arc_length, dash_len, dash_gap);
        }
        
        // Add join at the end of this segment (if not the last segment or if closed)
        let next_i = i + 1u;
        let needs_join = (next_i < segment_count) || is_closed;
        if (needs_join) {
            let joint_point = p1;
            let dir_in = point_dirs_before[next_i];
            let dir_out = point_dirs_after[next_i];
            
            // Check if join is needed (angle is significant)
            let cross = dir_in.x * dir_out.y - dir_in.y * dir_out.x;
            if (abs(cross) > 0.01) {
                if (path.stroke_join == JOIN_ROUND) {
                    addRoundJoin(&current_vertex, joint_point, dir_in, dir_out, path.stroke_width, path.color);
                } else if (path.stroke_join == JOIN_BEVEL) {
                    let offset_before = getPerpOffset(dir_in, path.stroke_width);
                    let offset_after = getPerpOffset(dir_out, path.stroke_width);
                    addBevelJoin(&current_vertex, joint_point, offset_before, offset_after, path.stroke_width, path.color);
                }
                // JOIN_MITER is handled by the miter offset calculation already
            }
        }
    }
    
    // If closed, add segment from last point to first point
    if (is_closed) {
        let last_idx = path.point_count - 1u;
        let p0 = getPoint(path.point_start, last_idx);
        let p1 = getPoint(path.point_start, 0u);
        let seg_type = getSegmentType(path.point_start, last_idx);
        
        if (seg_type == SEGMENT_CUBIC_BEZIER) {
            let c0 = getControlOut(path.point_start, last_idx);
            let c1 = getControlIn(path.point_start, 0u);
            arc_length = arc_length + addBezierCurveWithEndOffsets(&current_vertex, p0, c0, c1, p1, path.stroke_width, path.color, uniforms.zoom, point_offsets[last_idx], point_offsets[0u], arc_length, dash_len, dash_gap);
        } else {
            arc_length = arc_length + addLineSegmentWithOffsets(&current_vertex, p0, p1, point_offsets[last_idx], point_offsets[0u], path.color, arc_length, dash_len, dash_gap);
        }
        
        // Add join at first point to close the loop
        let dir_in = point_dirs_before[0u];
        let dir_out = point_dirs_after[0u];
        let cross = dir_in.x * dir_out.y - dir_in.y * dir_out.x;
        if (abs(cross) > 0.01) {
            if (path.stroke_join == JOIN_ROUND) {
                addRoundJoin(&current_vertex, p1, dir_in, dir_out, path.stroke_width, path.color);
            } else if (path.stroke_join == JOIN_BEVEL) {
                let offset_before = getPerpOffset(dir_in, path.stroke_width);
                let offset_after = getPerpOffset(dir_out, path.stroke_width);
                addBevelJoin(&current_vertex, p1, offset_before, offset_after, path.stroke_width, path.color);
            }
        }
    } else {
        // Add end cap for open paths
        if (path.stroke_cap == CAP_ROUND) {
            let end_idx = path.point_count - 1u;
            let end_point = getPoint(path.point_start, end_idx);
            let end_dir = point_dirs_before[end_idx];
            addRoundCap(&current_vertex, end_point, end_dir, path.stroke_width, path.color, false);
        }
    }
    
    // Store actual vertex count for this path
    atomicAdd(&path_vertex_counts[path_idx], current_vertex - base_vertex);
}
`;

// Compute pipeline resources
let computePipeline = null;
let computeBindGroup = null;
let shapeDescriptorBuffer = null; // Input: shape descriptors from CPU
let computeVertexBuffer = null; // Output: generated vertices
let computeUniformBuffer = null; // Uniforms for compute
let vertexCountBuffer = null; // Per-shape vertex counts

// Path compute pipeline resources
let pathComputePipeline = null;
let pathComputeBindGroup = null;
let pathDescriptorBuffer = null; // Input: path descriptors from CPU
let pathPointsBuffer = null; // Input: flattened path points
let pathComputeVertexBuffer = null; // Output: generated path vertices
let pathComputeUniformBuffer = null; // Uniforms for path compute
let pathVertexCountBuffer = null; // Per-path vertex counts

const COMPUTE_MAX_SHAPES = 10000;
const COMPUTE_MAX_SEGMENTS = 128;
const COMPUTE_VERTICES_PER_SHAPE = COMPUTE_MAX_SEGMENTS * 6; // Max vertices per shape
const SHAPE_DESCRIPTOR_SIZE = 80; // 20 floats/u32s * 4 bytes (with vec4 alignment padding)

// Path compute constants
const PATH_MAX_PATHS = 64;
const PATH_MAX_POINTS = 128;
const PATH_DESCRIPTOR_SIZE = 96; // Updated: includes transform matrix + alignment padding
const PATH_BEZIER_SUBDIVISIONS = 128; // Must match shader's BEZIER_SUBDIVISIONS (max for adaptive)
const PATH_VERTICES_PER_PATH = PATH_MAX_POINTS * PATH_BEZIER_SUBDIVISIONS * 12; // 6 for quad + 6 for bevel join

// Debug tile rendering resources
let debugTilePipeline = null;
let debugTileVertexBuffer = null;
let debugTileUniformBuffer = null;
let debugTileBindGroup = null;
const DEBUG_TILE_MAX_VERTICES = 1024 * 6; // Enough for grid lines + tile highlights

// Maximum vertices - increased for complex scenes (up to 10000 shapes)
const MAX_SHAPES = 10000;
const MAX_VERTICES = MAX_SHAPES * 64 * 6 * 2; // Conservative estimate: 64 segments per shape average
const VERTEX_SIZE = 24; // 2 floats (position) + 4 floats (color) = 6 * 4 bytes
const PATH_VERTEX_SIZE = 44; // 2 floats (position) + 4 floats (color) + 2 floats (edge_info) + 1 float (arc_length) + 2 floats (dash_pattern) = 11 * 4 bytes

async function initWebGPU() {
  postStatus("Requesting GPU adapter...");

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Failed to get GPU adapter");
  }

  postStatus("Requesting GPU device...");

  // Request higher limits for large storage buffers
  const requiredLimits = {
    maxStorageBufferBindingSize: Math.min(
      1024 * 1024 * 1024, // 1GB
      adapter.limits.maxStorageBufferBindingSize
    ),
    maxBufferSize: Math.min(
      1024 * 1024 * 1024, // 1GB
      adapter.limits.maxBufferSize
    ),
  };

  device = await adapter.requestDevice({
    requiredLimits,
  });

  postStatus("Configuring canvas context...");
  ctx = canvas.getContext("webgpu");

  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({
    device: device,
    format: canvasFormat,
    alphaMode: "premultiplied",
  });

  // Create MSAA texture
  createMSAATexture();

  // Create uniform buffer
  uniformBuffer = device.createBuffer({
    size: 32, // 8 floats * 4 bytes (time, mouse_x, mouse_y, aspect_ratio, zoom, pan_x, pan_y, _padding)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create bind group layout
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  // Create bind group
  uniformBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: uniformBuffer },
      },
    ],
  });

  // Create pipeline layout
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  // Create background pipeline with MSAA
  postStatus("Creating background pipeline...");
  const bgShaderModule = device.createShaderModule({ code: bgShaderCode });

  const bgPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: bgShaderModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: bgShaderModule,
      entryPoint: "fs_main",
      targets: [{ format: canvasFormat }],
    },
    primitive: {
      topology: "triangle-list",
    },
    multisample: {
      count: SAMPLE_COUNT,
    },
  });

  // Create 2D shape pipeline with MSAA
  postStatus("Creating 2D shape pipeline...");
  const shaderModule = device.createShaderModule({ code: shaderCode });

  pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: VERTEX_SIZE,
          attributes: [
            {
              shaderLocation: 0,
              offset: 0,
              format: "float32x2", // position
            },
            {
              shaderLocation: 1,
              offset: 8,
              format: "float32x4", // color (RGBA)
            },
          ],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format: canvasFormat }],
    },
    primitive: {
      topology: "triangle-list",
    },
    multisample: {
      count: SAMPLE_COUNT,
    },
  });

  // Create wireframe pipeline for debugging path geometry
  wireframePipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: VERTEX_SIZE,
          attributes: [
            {
              shaderLocation: 0,
              offset: 0,
              format: "float32x2", // position
            },
            {
              shaderLocation: 1,
              offset: 8,
              format: "float32x4", // color (RGBA)
            },
          ],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format: canvasFormat }],
    },
    primitive: {
      topology: "line-list",
    },
    multisample: {
      count: SAMPLE_COUNT,
    },
  });

  // Create path-specific pipeline with extended vertex format for analytic AA
  postStatus("Creating path render pipeline with AA...");
  const pathShaderModule = device.createShaderModule({ code: pathShaderCode });

  pathRenderPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: pathShaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: PATH_VERTEX_SIZE, // 44 bytes: position + color + edge_info + arc_length + dash_pattern
          attributes: [
            {
              shaderLocation: 0,
              offset: 0,
              format: "float32x2", // position
            },
            {
              shaderLocation: 1,
              offset: 8,
              format: "float32x4", // color (RGBA)
            },
            {
              shaderLocation: 2,
              offset: 24,
              format: "float32x2", // edge_info (dist_to_center, half_width)
            },
            {
              shaderLocation: 3,
              offset: 32,
              format: "float32", // arc_length
            },
            {
              shaderLocation: 4,
              offset: 36,
              format: "float32x2", // dash_pattern (dash_length, gap_length)
            },
          ],
        },
      ],
    },
    fragment: {
      module: pathShaderModule,
      entryPoint: "fs_main",
      targets: [
        {
          format: canvasFormat,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
    },
    multisample: {
      count: SAMPLE_COUNT,
    },
  });

  // Create vertex buffer (large enough for all shapes)
  vertexBuffer = device.createBuffer({
    size: MAX_VERTICES * VERTEX_SIZE,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  // Create composite pipeline for tile atlas blitting
  postStatus("Creating composite pipeline...");
  const compositeShaderModule = device.createShaderModule({
    code: compositeShaderCode,
  });

  atlasSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  compositePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: compositeShaderModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: compositeShaderModule,
      entryPoint: "fs_main",
      targets: [{ format: canvasFormat }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  // Create debug tile pipeline
  postStatus("Creating debug tile pipeline...");
  const debugTileShaderModule = device.createShaderModule({
    code: debugTileShaderCode,
  });

  debugTileUniformBuffer = device.createBuffer({
    size: 16, // 4 floats: viewport_width, viewport_height, tile_size, padding
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  debugTileVertexBuffer = device.createBuffer({
    size: DEBUG_TILE_MAX_VERTICES * 6 * 4, // x, y, r, g, b, a per vertex (6 floats * 4 bytes)
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  debugTilePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: debugTileShaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 6 * 4, // 6 floats per vertex
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" }, // position
            { shaderLocation: 1, offset: 8, format: "float32x4" }, // color
          ],
        },
      ],
    },
    fragment: {
      module: debugTileShaderModule,
      entryPoint: "fs_main",
      targets: [
        {
          format: canvasFormat,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  debugTileBindGroup = device.createBindGroup({
    layout: debugTilePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: debugTileUniformBuffer } }],
  });

  // ========================================================================
  // Compute Pipeline Setup (GPU geometry generation)
  // ========================================================================
  postStatus("Creating compute pipeline for GPU tessellation...");

  const computeShaderModule = device.createShaderModule({
    code: computeShaderCode,
  });

  // Shape descriptor buffer (input)
  shapeDescriptorBuffer = device.createBuffer({
    size: COMPUTE_MAX_SHAPES * SHAPE_DESCRIPTOR_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Compute vertex buffer (output) - large enough for all shapes
  computeVertexBuffer = device.createBuffer({
    size: COMPUTE_MAX_SHAPES * COMPUTE_VERTICES_PER_SHAPE * 24, // 24 bytes per vertex (2 pos + 4 color)
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC,
  });

  // Per-shape vertex counts (for knowing how many vertices each shape generated)
  vertexCountBuffer = device.createBuffer({
    size: COMPUTE_MAX_SHAPES * 4, // u32 per shape
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  // Compute uniforms
  computeUniformBuffer = device.createBuffer({
    size: 16, // 4 floats: zoom, aspect_ratio, shape_count, max_segments
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create compute pipeline
  computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: computeShaderModule,
      entryPoint: "main",
    },
  });

  // Create compute bind group
  computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: computeUniformBuffer } },
      { binding: 1, resource: { buffer: shapeDescriptorBuffer } },
      { binding: 2, resource: { buffer: computeVertexBuffer } },
      { binding: 3, resource: { buffer: vertexCountBuffer } },
    ],
  });

  postStatus("Compute pipeline ready!");

  // ========================================================================
  // Path Compute Pipeline Setup (GPU path geometry generation)
  // ========================================================================
  postStatus("Creating path compute pipeline...");

  const pathComputeShaderModule = device.createShaderModule({
    code: pathComputeShaderCode,
  });

  // Path descriptor buffer (input)
  pathDescriptorBuffer = device.createBuffer({
    size: PATH_MAX_PATHS * PATH_DESCRIPTOR_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Path points buffer (input) - flattened point data for all paths
  // Format: x, y, ctrl_in_x, ctrl_in_y, ctrl_out_x, ctrl_out_y, segment_type (7 floats per point)
  pathPointsBuffer = device.createBuffer({
    size: PATH_MAX_PATHS * PATH_MAX_POINTS * 7 * 4, // 7 floats per point, 4 bytes each
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Path compute vertex buffer (output) - using PATH_VERTEX_SIZE (32 bytes) for edge_info
  pathComputeVertexBuffer = device.createBuffer({
    size: PATH_MAX_PATHS * PATH_VERTICES_PER_PATH * PATH_VERTEX_SIZE, // 44 bytes per vertex
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC,
  });

  // Per-path vertex counts
  pathVertexCountBuffer = device.createBuffer({
    size: PATH_MAX_PATHS * 4, // u32 per path
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  // Path compute uniforms
  pathComputeUniformBuffer = device.createBuffer({
    size: 16, // 4 floats/u32: zoom, aspect_ratio, path_count, max_points_per_path
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create path compute pipeline
  pathComputePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: pathComputeShaderModule,
      entryPoint: "main",
    },
  });

  // Create path compute bind group
  pathComputeBindGroup = device.createBindGroup({
    layout: pathComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: pathComputeUniformBuffer } },
      { binding: 1, resource: { buffer: pathDescriptorBuffer } },
      { binding: 2, resource: { buffer: pathPointsBuffer } },
      { binding: 3, resource: { buffer: pathComputeVertexBuffer } },
      { binding: 4, resource: { buffer: pathVertexCountBuffer } },
    ],
  });

  postStatus("Path compute pipeline ready!");

  return { bgPipeline, format: canvasFormat };
}

// Generate debug tile vertices for grid lines and dirty tile highlights
// Uses Zig's 16x16 NDC-based tile grid, not the pixel-based tilesX/tilesY
function generateDebugTileVertices(
  maxTilesX,
  maxTilesY,
  viewportWidth,
  viewportHeight
) {
  const vertices = [];
  const lineWidth = 2.0; // Thicker lines for visibility

  // Calculate tile size in pixels based on viewport and Zig's grid
  const tileSizeX = viewportWidth / maxTilesX;
  const tileSizeY = viewportHeight / maxTilesY;

  // Grid line color (dark gray, semi-transparent)
  const gridColor = [0.0, 0.0, 0.0, 0.5];
  // Dirty tile color (red with medium alpha)
  const dirtyColor = [1.0, 0.0, 0.0, 0.3];

  // Draw dirty tile backgrounds first (so grid lines appear on top)
  // Query Zig directly for each tile's dirty state
  // Note: Zig uses NDC where ty=0 is bottom, but screen Y=0 is top, so flip Y
  for (let ty = 0; ty < maxTilesY; ty++) {
    for (let tx = 0; tx < maxTilesX; tx++) {
      const isDirty = wasm.instance.exports.isTileDirty(tx, ty);

      if (isDirty) {
        const x0 = tx * tileSizeX;
        // Flip Y: ty=0 in Zig is bottom, so map to screen bottom
        const screenTy = maxTilesY - 1 - ty;
        const y0 = screenTy * tileSizeY;
        const x1 = x0 + tileSizeX;
        const y1 = y0 + tileSizeY;

        // Two triangles for the quad
        vertices.push(
          x0,
          y0,
          ...dirtyColor,
          x1,
          y0,
          ...dirtyColor,
          x0,
          y1,
          ...dirtyColor,
          x1,
          y0,
          ...dirtyColor,
          x1,
          y1,
          ...dirtyColor,
          x0,
          y1,
          ...dirtyColor
        );
      }
    }
  }

  // Draw vertical grid lines
  for (let tx = 0; tx <= maxTilesX; tx++) {
    const x = tx * tileSizeX;
    const x0 = x - lineWidth / 2;
    const x1 = x + lineWidth / 2;
    const y0 = 0;
    const y1 = viewportHeight;

    vertices.push(
      x0,
      y0,
      ...gridColor,
      x1,
      y0,
      ...gridColor,
      x0,
      y1,
      ...gridColor,
      x1,
      y0,
      ...gridColor,
      x1,
      y1,
      ...gridColor,
      x0,
      y1,
      ...gridColor
    );
  }

  // Draw horizontal grid lines
  for (let ty = 0; ty <= maxTilesY; ty++) {
    const y = ty * tileSizeY;
    const y0 = y - lineWidth / 2;
    const y1 = y + lineWidth / 2;
    const x0 = 0;
    const x1 = viewportWidth;

    vertices.push(
      x0,
      y0,
      ...gridColor,
      x1,
      y0,
      ...gridColor,
      x0,
      y1,
      ...gridColor,
      x1,
      y0,
      ...gridColor,
      x1,
      y1,
      ...gridColor,
      x0,
      y1,
      ...gridColor
    );
  }

  return new Float32Array(vertices);
}

function createMSAATexture() {
  if (msaaTexture) {
    msaaTexture.destroy();
  }
  msaaTexture = device.createTexture({
    size: { width, height },
    format: canvasFormat,
    sampleCount: SAMPLE_COUNT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  msaaTextureView = msaaTexture.createView();

  // Update tile grid dimensions
  tilesX = Math.ceil(width / TILE_SIZE_PX);
  tilesY = Math.ceil(height / TILE_SIZE_PX);

  const atlasWidth = tilesX * TILE_SIZE_PX;
  const atlasHeight = tilesY * TILE_SIZE_PX;

  // Create tile atlas MSAA texture for rendering
  if (tileAtlasMsaaTexture) {
    tileAtlasMsaaTexture.destroy();
  }
  tileAtlasMsaaTexture = device.createTexture({
    size: { width: atlasWidth, height: atlasHeight },
    format: canvasFormat,
    sampleCount: SAMPLE_COUNT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  tileAtlasMsaaView = tileAtlasMsaaTexture.createView();

  // Create tile atlas resolve texture (for sampling in composite pass)
  if (tileAtlasTexture) {
    tileAtlasTexture.destroy();
  }
  tileAtlasTexture = device.createTexture({
    size: { width: atlasWidth, height: atlasHeight },
    format: canvasFormat,
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });
  tileAtlasView = tileAtlasTexture.createView();

  // Create composite bind group for atlas blitting
  if (compositePipeline && atlasSampler) {
    compositeBindGroup = device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: tileAtlasView },
        { binding: 1, resource: atlasSampler },
      ],
    });
  }
}

async function loadWasm() {
  postStatus("Loading WASM module...");

  const response = await fetch("wiggle.wasm");
  const wasmBytes = await response.arrayBuffer();

  wasm = await WebAssembly.instantiate(wasmBytes, wasmImports);

  // Initialize Zig module
  wasm.instance.exports.init();

  postStatus("WASM loaded successfully");
}

let bgPipeline = null;

async function init(initCanvas, initWidth, initHeight) {
  try {
    canvas = initCanvas;
    width = initWidth;
    height = initHeight;

    // Set the canvas pixel dimensions BEFORE configuring WebGPU
    canvas.width = width;
    canvas.height = height;

    const { bgPipeline: bg } = await initWebGPU();
    bgPipeline = bg;

    await loadWasm();

    // Update canvas size in Zig
    wasm.instance.exports.updateCanvasSize(width, height);

    postStatus("Running");

    // Start render loop
    render();
  } catch (error) {
    console.error("Initialization error:", error);
    postError(error.message);
  }
}

let lastFpsTime = performance.now();
let frameCounter = 0;
let lastFps = 0;

// Run GPU compute shaders for geometry generation
// Returns array of {baseVertex, vertexCount} for each shape
function runComputeGeometry(commandEncoder) {
  if (!computePipeline) return null;

  // Build shape descriptors in Zig
  const descriptorCount = wasm.instance.exports.buildShapeDescriptors();
  if (descriptorCount === 0) return null;

  // Get descriptor data from Zig
  const descriptorPtr = wasm.instance.exports.getShapeDescriptorPtr();
  const descriptorSize = wasm.instance.exports.getShapeDescriptorSize();
  const descriptorData = new Uint8Array(
    wasm.instance.exports.memory.buffer,
    descriptorPtr,
    descriptorCount * descriptorSize
  );

  // Upload shape descriptors to GPU
  device.queue.writeBuffer(shapeDescriptorBuffer, 0, descriptorData);

  // Clear vertex counts
  const zeroCounts = new Uint32Array(descriptorCount);
  device.queue.writeBuffer(vertexCountBuffer, 0, zeroCounts);

  // Get current uniforms
  const zoom = wasm.instance.exports.getZoom();
  const aspectRatio = width / height;

  // Upload compute uniforms (mixed float/uint types)
  const uniformBuffer = new ArrayBuffer(16);
  const uniformFloatView = new Float32Array(uniformBuffer);
  const uniformUintView = new Uint32Array(uniformBuffer);
  uniformFloatView[0] = zoom;
  uniformFloatView[1] = aspectRatio;
  uniformUintView[2] = descriptorCount;
  uniformUintView[3] = COMPUTE_MAX_SEGMENTS;
  device.queue.writeBuffer(
    computeUniformBuffer,
    0,
    new Uint8Array(uniformBuffer)
  );

  // Run compute shader
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(computePipeline);
  computePass.setBindGroup(0, computeBindGroup);
  computePass.dispatchWorkgroups(descriptorCount); // One workgroup per shape
  computePass.end();

  // Calculate draw info for each shape based on descriptor data
  const descriptorView = new DataView(
    descriptorData.buffer,
    descriptorData.byteOffset,
    descriptorData.byteLength
  );
  const drawInfos = [];

  for (let i = 0; i < descriptorCount; i++) {
    const offset = i * descriptorSize;
    // ShapeDescriptor layout (80 bytes):
    // offset 0: shape_type (u32)
    // offset 4: render_mode (u32)
    // offset 8: width (f32)
    // offset 12: height (f32)
    // offset 16: stroke_width (f32)
    // offset 20: corner_radius (f32)
    // offset 24: shape_index (u32)
    // offset 28: _padding0 (u32)
    // offset 32: transform (24 bytes = 6 floats)
    // offset 56: _padding1 (u32)
    // offset 60: _padding2 (u32)
    // offset 64: color (16 bytes = vec4)
    const shapeType = descriptorView.getUint32(offset, true); // shape_type
    const renderMode = descriptorView.getUint32(offset + 4, true); // render_mode
    const shapeWidth = descriptorView.getFloat32(offset + 8, true); // width
    const shapeHeight = descriptorView.getFloat32(offset + 12, true); // height
    const cornerRadius = descriptorView.getFloat32(offset + 20, true); // corner_radius
    const shapeIndex = descriptorView.getUint32(offset + 24, true); // shape_index

    const baseVertex = i * COMPUTE_MAX_SEGMENTS * 6;
    let vertexCount = 0;

    if (shapeType === 1) {
      // Rectangle (type 1 in Zig)
      // Check if it has rounded corners
      const hasRoundedCorners = cornerRadius > 0.0001;
      const ARC_SEGMENTS = 8; // matches WGSL ARC_SEGMENTS

      if (renderMode === 0) {
        // Fill: path-based triangle fan from center
        // 4 edges + 4 arcs × ARC_SEGMENTS (starting from i=1) = 4 + 32 = 36 triangles
        if (hasRoundedCorners) {
          vertexCount = (4 + 4 * ARC_SEGMENTS) * 3; // 36 * 3 = 108
        } else {
          vertexCount = 6; // 2 triangles
        }
      } else {
        // Stroke: path-based quad strip
        // 4 edges + 4 arcs × ARC_SEGMENTS (starting from i=1) = 4 + 32 = 36 quads
        if (hasRoundedCorners) {
          vertexCount = (4 + 4 * ARC_SEGMENTS) * 6; // 36 * 6 = 216
        } else {
          vertexCount = 24; // 4 edges * 6 vertices
        }
      }
    } else if (shapeType === 0) {
      // Ellipse (type 0 in Zig)
      // Calculate segments like the shader does (must match WGSL calcSegments)
      const maxRadius = Math.max(shapeWidth, shapeHeight);
      const screenSize = maxRadius * zoom * 2.0;
      const baseSegments = Math.floor(screenSize * 500.0);
      const segments = Math.max(24, Math.min(128, baseSegments));

      if (renderMode === 0) {
        // Fill
        vertexCount = segments * 3; // triangle fan
      } else {
        // Stroke
        vertexCount = segments * 6; // ring
      }
    }

    if (vertexCount > 0) {
      drawInfos.push({ baseVertex, vertexCount, shapeIndex, type: "shape" });
    }
  }

  return drawInfos;
}

// Run GPU compute shaders for path geometry generation
// Returns array of {baseVertex, vertexCount} for each path
function runPathComputeGeometry(commandEncoder) {
  if (!pathComputePipeline) return null;

  // Build path descriptors in Zig
  const pathCount = wasm.instance.exports.buildPathDescriptors();
  if (pathCount === 0) return null;

  // Get path descriptor data from Zig
  const descriptorPtr = wasm.instance.exports.getPathDescriptorPtr();
  const descriptorSize = wasm.instance.exports.getPathDescriptorSize();
  const descriptorData = new Uint8Array(
    wasm.instance.exports.memory.buffer,
    descriptorPtr,
    pathCount * descriptorSize
  );

  // Get path points data from Zig
  const pointsPtr = wasm.instance.exports.getPathPointsPtr();
  const pointsCount = wasm.instance.exports.getPathPointsCount();
  const pointStride = wasm.instance.exports.getPathPointStride();
  const pointsData = new Uint8Array(
    wasm.instance.exports.memory.buffer,
    pointsPtr,
    pointsCount * pointStride * 4 // pointStride floats per point, 4 bytes each
  );

  // Upload path descriptors and points to GPU
  device.queue.writeBuffer(pathDescriptorBuffer, 0, descriptorData);
  device.queue.writeBuffer(pathPointsBuffer, 0, pointsData);

  // Clear vertex counts
  const zeroCounts = new Uint32Array(pathCount);
  device.queue.writeBuffer(pathVertexCountBuffer, 0, zeroCounts);

  // Note: We no longer clear the path vertex buffer here because:
  // 1. The Zig Path struct now zeros arrays by default, preventing stale data
  // 2. Clearing the buffer before compute was causing geometry corruption
  // The compute shader will overwrite only the vertices it needs.

  // Get current uniforms
  const zoom = wasm.instance.exports.getZoom();
  const aspectRatio = width / height;

  // Upload path compute uniforms
  const uniformBuffer = new ArrayBuffer(16);
  const uniformFloatView = new Float32Array(uniformBuffer);
  const uniformUintView = new Uint32Array(uniformBuffer);
  uniformFloatView[0] = zoom;
  uniformFloatView[1] = aspectRatio;
  uniformUintView[2] = pathCount;
  uniformUintView[3] = PATH_MAX_POINTS;
  device.queue.writeBuffer(
    pathComputeUniformBuffer,
    0,
    new Uint8Array(uniformBuffer)
  );

  // Run path compute shader
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(pathComputePipeline);
  computePass.setBindGroup(0, pathComputeBindGroup);
  computePass.dispatchWorkgroups(pathCount); // One workgroup per path
  computePass.end();

  // Calculate draw info for each path
  const descriptorView = new DataView(
    descriptorData.buffer,
    descriptorData.byteOffset,
    descriptorData.byteLength
  );
  const drawInfos = [];

  for (let i = 0; i < pathCount; i++) {
    const offset = i * descriptorSize;
    const pointCount = descriptorView.getUint32(offset + 4, true); // point_count
    const closed = descriptorView.getUint32(offset + 8, true); // closed
    const shapeIndex = descriptorView.getUint32(offset + 48, true); // shape_index

    // Each path segment can have up to BEZIER_SUBDIVISIONS sub-segments
    // Each sub-segment uses 6 vertices for quad + 6 for bevel join = 12 vertices
    const segmentCount = pointCount - 1 + (closed ? 1 : 0);
    const vertexCount = segmentCount * PATH_BEZIER_SUBDIVISIONS * 12;
    // Must match shader's base_vertex calculation
    const baseVertex = i * PATH_MAX_POINTS * PATH_BEZIER_SUBDIVISIONS * 12;

    if (vertexCount > 0) {
      drawInfos.push({ baseVertex, vertexCount, shapeIndex, type: "path" });
    }
  }

  return drawInfos;
}

function render() {
  if (!device || !wasm) return;

  const time = performance.now() / 1000.0;

  // Update Zig state
  wasm.instance.exports.update(time);

  // Get vertex data from Zig
  const vertexPtr = wasm.instance.exports.getVertexDataPtr();
  const vertexSize = wasm.instance.exports.getVertexDataSize();
  const vertexCount = wasm.instance.exports.getVertexCount();
  const vertexData = new Uint8Array(
    wasm.instance.exports.memory.buffer,
    vertexPtr,
    vertexSize
  );

  // Get uniform data from Zig
  const uniformPtr = wasm.instance.exports.getUniformDataPtr();
  const uniformSize = wasm.instance.exports.getUniformDataSize();
  const uniformData = new Uint8Array(
    wasm.instance.exports.memory.buffer,
    uniformPtr,
    uniformSize
  );

  // Get scene vs UI vertex counts
  // Compute shaders now generate ALL scene geometry (shapes AND paths)
  // Only UI overlay (selection boxes, handles) is CPU-generated
  const sceneVertexCount = wasm.instance.exports.getSceneVertexCount();
  const uiVertexCount = wasm.instance.exports.getUIVertexCount();

  // Upload UI vertex data only (paths are now GPU-generated)
  // UI vertex data is now at the start of the buffer (no scene vertices)
  if (uiVertexCount > 0) {
    // Note: UI vertices are still stored after scene vertices in Zig's buffer
    // We need to upload them to start of CPU vertex buffer for drawing
    device.queue.writeBuffer(
      vertexBuffer,
      0,
      vertexData,
      sceneVertexCount * VERTEX_SIZE, // Skip scene data in source
      uiVertexCount * VERTEX_SIZE
    );
  }
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // Create command encoder
  const commandEncoder = device.createCommandEncoder();

  // Run compute shaders for GPU geometry generation (shapes and paths)
  const computeDrawInfos = runComputeGeometry(commandEncoder);
  const pathDrawInfos = runPathComputeGeometry(commandEncoder);

  // Get current texture for resolve target
  const resolveTarget = ctx.getCurrentTexture().createView();

  // Get dirty tile count for deciding rendering strategy
  const dirtyTileCount = wasm.instance.exports.getDirtyTileCount();
  const totalTiles = tilesX * tilesY;

  // Use tile-based rendering when enabled and we have the resources
  if (
    useTileRendering &&
    compositeBindGroup &&
    dirtyTileCount > 0 &&
    dirtyTileCount < totalTiles
  ) {
    // TILE-BASED RENDERING PATH
    // Only re-render dirty tiles to the atlas

    // Render dirty tiles to atlas
    const atlasRenderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: tileAtlasMsaaView,
          resolveTarget: tileAtlasView,
          clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
          loadOp: "load", // Keep existing tile content
          storeOp: "discard",
        },
      ],
    });

    // Draw background for dirty tiles only using scissor
    atlasRenderPass.setPipeline(bgPipeline);

    const zigTilesX = wasm.instance.exports.getMaxTilesX();
    const zigTilesY = wasm.instance.exports.getMaxTilesY();

    for (let ty = 0; ty < Math.min(tilesY, zigTilesY); ty++) {
      for (let tx = 0; tx < Math.min(tilesX, zigTilesX); tx++) {
        if (wasm.instance.exports.isTileDirty(tx, ty) === 1) {
          // Set scissor rect for this tile
          const x = tx * TILE_SIZE_PX;
          const y = ty * TILE_SIZE_PX;
          atlasRenderPass.setScissorRect(x, y, TILE_SIZE_PX, TILE_SIZE_PX);
          atlasRenderPass.setViewport(
            0,
            0,
            tilesX * TILE_SIZE_PX,
            tilesY * TILE_SIZE_PX,
            0,
            1
          );
          atlasRenderPass.draw(6);
        }
      }
    }

    // Combine and sort all draw infos for proper layer order in tile rendering
    const allTileDrawInfos = [];
    if (computeDrawInfos) {
      allTileDrawInfos.push(...computeDrawInfos);
    }
    if (pathDrawInfos) {
      allTileDrawInfos.push(...pathDrawInfos);
    }
    allTileDrawInfos.sort((a, b) => a.shapeIndex - b.shapeIndex);

    // Draw all shapes and paths in layer order for dirty tiles
    if (allTileDrawInfos.length > 0) {
      for (let ty = 0; ty < Math.min(tilesY, zigTilesY); ty++) {
        for (let tx = 0; tx < Math.min(tilesX, zigTilesX); tx++) {
          if (wasm.instance.exports.isTileDirty(tx, ty) === 1) {
            const x = tx * TILE_SIZE_PX;
            const y = ty * TILE_SIZE_PX;
            atlasRenderPass.setScissorRect(x, y, TILE_SIZE_PX, TILE_SIZE_PX);

            let currentType = null;
            for (const drawInfo of allTileDrawInfos) {
              if (drawInfo.type === "shape") {
                if (currentType !== "shape") {
                  atlasRenderPass.setPipeline(pipeline);
                  atlasRenderPass.setBindGroup(0, uniformBindGroup);
                  atlasRenderPass.setVertexBuffer(0, computeVertexBuffer);
                  currentType = "shape";
                }
                atlasRenderPass.draw(
                  drawInfo.vertexCount,
                  1,
                  drawInfo.baseVertex,
                  0
                );
              } else if (drawInfo.type === "path") {
                if (currentType !== "path") {
                  atlasRenderPass.setPipeline(pathRenderPipeline);
                  atlasRenderPass.setBindGroup(0, uniformBindGroup);
                  atlasRenderPass.setVertexBuffer(0, pathComputeVertexBuffer);
                  currentType = "path";
                }
                atlasRenderPass.draw(
                  drawInfo.vertexCount,
                  1,
                  drawInfo.baseVertex,
                  0
                );
              }
            }
          }
        }
      }
    }

    atlasRenderPass.end();

    // Composite atlas to screen
    const compositePass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: resolveTarget,
          clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    compositePass.setPipeline(compositePipeline);
    compositePass.setBindGroup(0, compositeBindGroup);
    compositePass.draw(6);

    // Draw all shapes and paths on top of tiles in layer order
    if (allTileDrawInfos.length > 0) {
      let currentType = null;
      for (const drawInfo of allTileDrawInfos) {
        if (drawInfo.type === "shape") {
          if (currentType !== "shape") {
            compositePass.setPipeline(pipeline);
            compositePass.setBindGroup(0, uniformBindGroup);
            compositePass.setVertexBuffer(0, computeVertexBuffer);
            currentType = "shape";
          }
          compositePass.draw(drawInfo.vertexCount, 1, drawInfo.baseVertex, 0);
        } else if (drawInfo.type === "path") {
          if (currentType !== "path") {
            compositePass.setPipeline(pathRenderPipeline);
            compositePass.setBindGroup(0, uniformBindGroup);
            compositePass.setVertexBuffer(0, pathComputeVertexBuffer);
            currentType = "path";
          }
          compositePass.draw(drawInfo.vertexCount, 1, drawInfo.baseVertex, 0);
        }
      }
    }

    // Draw UI overlay on top
    if (uiVertexCount > 0) {
      // For UI, we need to use the regular pipeline with MSAA
      // This is a simplified approach - UI goes on final output
      compositePass.setPipeline(pipeline);
      compositePass.setBindGroup(0, uniformBindGroup);
      compositePass.setVertexBuffer(0, vertexBuffer);
      compositePass.draw(uiVertexCount, 1, 0, 0); // UI now at index 0 in vertex buffer
    }

    compositePass.end();
  } else {
    // STANDARD RENDERING PATH (all tiles dirty or tile rendering disabled)
    // Render everything to screen directly

    // Begin render pass with MSAA
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: msaaTextureView,
          resolveTarget: resolveTarget,
          clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "discard",
        },
      ],
    });

    // Draw background (white)
    renderPass.setPipeline(bgPipeline);
    renderPass.draw(6);

    // Combine shape and path draw infos, sort by shapeIndex for proper layer order
    const allDrawInfos = [];
    if (computeDrawInfos) {
      allDrawInfos.push(...computeDrawInfos);
    }
    if (pathDrawInfos) {
      allDrawInfos.push(...pathDrawInfos);
    }
    // Sort by shapeIndex to respect layer order
    allDrawInfos.sort((a, b) => a.shapeIndex - b.shapeIndex);

    // Draw all shapes and paths in layer order
    let currentType = null;
    for (const drawInfo of allDrawInfos) {
      if (drawInfo.type === "shape") {
        if (currentType !== "shape") {
          renderPass.setPipeline(pipeline);
          renderPass.setBindGroup(0, uniformBindGroup);
          renderPass.setVertexBuffer(0, computeVertexBuffer);
          currentType = "shape";
        }
        renderPass.draw(drawInfo.vertexCount, 1, drawInfo.baseVertex, 0);
      } else if (drawInfo.type === "path") {
        if (currentType !== "path") {
          renderPass.setPipeline(
            wireframeMode ? wireframePipeline : pathRenderPipeline
          );
          renderPass.setBindGroup(0, uniformBindGroup);
          renderPass.setVertexBuffer(0, pathComputeVertexBuffer);
          currentType = "path";
        }
        renderPass.draw(drawInfo.vertexCount, 1, drawInfo.baseVertex, 0);
      }
    }

    // Draw UI overlay (selection boxes, handles, previews) - always fresh
    if (uiVertexCount > 0) {
      renderPass.setPipeline(pipeline);
      renderPass.setBindGroup(0, uniformBindGroup);
      renderPass.setVertexBuffer(0, vertexBuffer);
      renderPass.draw(uiVertexCount, 1, 0, 0); // UI now at index 0 in vertex buffer
    }

    renderPass.end();

    // If tile rendering is enabled, also update the atlas for next frame
    if (
      useTileRendering &&
      tileAtlasMsaaView &&
      totalTiles === dirtyTileCount
    ) {
      // Full atlas update when all tiles are dirty
      const atlasPass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: tileAtlasMsaaView,
            resolveTarget: tileAtlasView,
            clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
            loadOp: "clear",
            storeOp: "discard",
          },
        ],
      });

      atlasPass.setPipeline(bgPipeline);
      atlasPass.draw(6);

      // Combine and sort all draw infos for proper layer order
      const allAtlasDrawInfos = [];
      if (computeDrawInfos) {
        allAtlasDrawInfos.push(...computeDrawInfos);
      }
      if (pathDrawInfos) {
        allAtlasDrawInfos.push(...pathDrawInfos);
      }
      allAtlasDrawInfos.sort((a, b) => a.shapeIndex - b.shapeIndex);

      // Draw all shapes and paths in layer order
      if (allAtlasDrawInfos.length > 0) {
        let currentType = null;
        for (const drawInfo of allAtlasDrawInfos) {
          if (drawInfo.type === "shape") {
            if (currentType !== "shape") {
              atlasPass.setPipeline(pipeline);
              atlasPass.setBindGroup(0, uniformBindGroup);
              atlasPass.setVertexBuffer(0, computeVertexBuffer);
              currentType = "shape";
            }
            atlasPass.draw(drawInfo.vertexCount, 1, drawInfo.baseVertex, 0);
          } else if (drawInfo.type === "path") {
            if (currentType !== "path") {
              atlasPass.setPipeline(pathRenderPipeline);
              atlasPass.setBindGroup(0, uniformBindGroup);
              atlasPass.setVertexBuffer(0, pathComputeVertexBuffer);
              currentType = "path";
            }
            atlasPass.draw(drawInfo.vertexCount, 1, drawInfo.baseVertex, 0);
          }
        }
      }

      atlasPass.end();
    }
  }

  // Debug tile overlay rendering (render to final canvas, not MSAA)
  if (debugTiles && debugTilePipeline) {
    // Get Zig's tile grid dimensions (16x16 in NDC space)
    const maxTilesX = wasm.instance.exports.getMaxTilesX();
    const maxTilesY = wasm.instance.exports.getMaxTilesY();

    // Generate debug vertices - queries Zig directly for dirty tiles
    const debugVertices = generateDebugTileVertices(
      maxTilesX,
      maxTilesY,
      width,
      height
    );
    const debugVertexCount = debugVertices.length / 6; // 6 floats per vertex

    if (debugVertexCount > 0 && debugTileVertexBuffer && debugTileBindGroup) {
      // Update debug uniform buffer with viewport dimensions
      device.queue.writeBuffer(
        debugTileUniformBuffer,
        0,
        new Float32Array([width, height, TILE_SIZE_PX, 0])
      );

      // Update debug vertex buffer
      device.queue.writeBuffer(debugTileVertexBuffer, 0, debugVertices);

      // Render debug overlay directly to canvas
      const debugPass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: resolveTarget,
            loadOp: "load", // Keep existing content
            storeOp: "store",
          },
        ],
      });

      debugPass.setPipeline(debugTilePipeline);
      debugPass.setBindGroup(0, debugTileBindGroup);
      debugPass.setVertexBuffer(0, debugTileVertexBuffer);
      debugPass.draw(debugVertexCount);
      debugPass.end();
    }
  }

  // Submit commands
  device.queue.submit([commandEncoder.finish()]);

  // Clear tile dirty flags after rendering (tiles have been "rendered")
  // This allows accurate tracking of which tiles change between frames
  wasm.instance.exports.clearAllTiles();

  // FPS calculation
  frameCounter++;
  const now = performance.now();
  if (now - lastFpsTime >= 500) {
    lastFps = Math.round((frameCounter * 1000) / (now - lastFpsTime));
    lastFpsTime = now;
    frameCounter = 0;
    // Include tile stats in FPS message (use dirtyTileCount captured before clear)
    const maxTilesX = wasm.instance.exports.getMaxTilesX();
    const maxTilesY = wasm.instance.exports.getMaxTilesY();
    postMessage({
      type: "fps",
      data: lastFps,
      tiles: {
        dirty: dirtyTileCount, // Use the count captured at start of render
        total: maxTilesX * maxTilesY,
        gridX: maxTilesX,
        gridY: maxTilesY,
      },
    });
  }

  // Check if shapes changed and send update
  if (wasm && wasm.instance.exports.isShapesDirty() === 1) {
    wasm.instance.exports.clearShapesDirty();
    sendShapesUpdate();
  }

  // Request next frame
  requestAnimationFrame(render);
}

// Send shapes update to main thread
function sendShapesUpdate() {
  if (!wasm) return;

  const count = wasm.instance.exports.getShapeCount();
  const shapes = [];
  for (let i = 0; i < count; i++) {
    shapes.push({
      id: i,
      type: wasm.instance.exports.getShapeType(i),
      parentId: wasm.instance.exports.getShapeParent(i),
      visible: wasm.instance.exports.getShapeVisible(i) === 1,
      selected: wasm.instance.exports.isShapeSelected(i) === 1,
      x: wasm.instance.exports.getShapeX(i),
      y: wasm.instance.exports.getShapeY(i),
      width: wasm.instance.exports.getShapeWidth(i),
      height: wasm.instance.exports.getShapeHeight(i),
      fillEnabled: wasm.instance.exports.getShapeFillEnabled(i) === 1,
      fillColor: [
        wasm.instance.exports.getShapeFillR(i),
        wasm.instance.exports.getShapeFillG(i),
        wasm.instance.exports.getShapeFillB(i),
      ],
      strokeEnabled: wasm.instance.exports.getShapeStrokeEnabled(i) === 1,
      strokeColor: [
        wasm.instance.exports.getShapeStrokeR(i),
        wasm.instance.exports.getShapeStrokeG(i),
        wasm.instance.exports.getShapeStrokeB(i),
      ],
      strokeWidth: wasm.instance.exports.getShapeStrokeWidth(i),
    });
  }
  const selectedId = wasm.instance.exports.getSelectedShape();
  const hasSelection = wasm.instance.exports.hasSelection() === 1;
  const currentToolFromZig = wasm.instance.exports.getCurrentTool();
  self.postMessage({
    type: "shapes",
    shapes,
    selectedId,
    hasSelection,
    currentTool: currentToolFromZig,
  });
}

// Current tool
let currentTool = 2; // 0 = circle, 1 = rect, 2 = select, 3 = pen

// Handle messages from main thread
self.onmessage = async (e) => {
  const { type, ...data } = e.data;

  switch (type) {
    case "init":
      await init(data.canvas, data.width, data.height);
      break;

    case "resize":
      width = data.width;
      height = data.height;
      // Resize the offscreen canvas
      if (canvas) {
        canvas.width = width;
        canvas.height = height;
      }
      // Recreate MSAA texture for new size
      if (device && canvasFormat) {
        createMSAATexture();
      }
      if (wasm) {
        wasm.instance.exports.updateCanvasSize(width, height);
      }
      break;

    case "mouse":
      mouseX = data.x;
      mouseY = data.y;
      mousePressed = data.pressed;
      if (wasm) {
        wasm.instance.exports.updateMousePosition(mouseX, mouseY);
        wasm.instance.exports.updateMousePressed(mousePressed);
      }
      break;

    case "tool":
      currentTool =
        data.tool === "circle"
          ? 0
          : data.tool === "rect"
          ? 1
          : data.tool === "pen"
          ? 3
          : 2; // select = 2
      if (wasm) {
        wasm.instance.exports.setTool(currentTool);
      }
      break;

    case "finishPenPath":
      if (wasm) {
        wasm.instance.exports.finishPenPath();
      }
      break;

    case "cancelPenPath":
      if (wasm) {
        wasm.instance.exports.cancelPenPath();
      }
      break;

    case "style":
      if (wasm) {
        // Convert pixel stroke width to NDC (approximate: 2px at 1024px canvas = 0.01 NDC)
        const strokeWidthNDC = (data.strokeWidth || 2) / 200;
        // Convert pixel corner radius to NDC
        const cornerRadiusNDC = (data.cornerRadius || 0) / 200;
        wasm.instance.exports.setStyle(
          data.fillEnabled ? 1 : 0,
          data.fillR,
          data.fillG,
          data.fillB,
          data.strokeEnabled ? 1 : 0,
          data.strokeR,
          data.strokeG,
          data.strokeB,
          strokeWidthNDC,
          cornerRadiusNDC
        );
      }
      break;

    case "zoom":
      if (wasm) {
        wasm.instance.exports.setZoom(data.zoom, data.panX, data.panY);
      }
      break;

    case "metaKey":
      if (wasm) {
        wasm.instance.exports.setMetaKey(data.pressed ? 1 : 0);
      }
      break;

    case "shiftKey":
      if (wasm) {
        wasm.instance.exports.setShiftKey(data.pressed ? 1 : 0);
      }
      break;

    case "keydown":
      break;

    case "keyup":
      break;

    case "getShapes":
      sendShapesUpdate();
      break;

    case "setShapeVisible":
      if (wasm) {
        wasm.instance.exports.setShapeVisible(data.index, data.visible ? 1 : 0);
      }
      break;

    case "setShapeParent":
      if (wasm) {
        wasm.instance.exports.setShapeParent(data.index, data.parentId);
      }
      break;

    case "createGroup":
      if (wasm) {
        const groupId = wasm.instance.exports.createGroup();
        // Set selected shapes as children of this group
        if (data.children && data.children.length > 0) {
          for (const childId of data.children) {
            wasm.instance.exports.setShapeParent(childId, groupId);
          }
        }
        self.postMessage({ type: "groupCreated", groupId });
      }
      break;

    case "selectShape":
      if (wasm) {
        wasm.instance.exports.setSelectedShape(data.index);
      }
      break;

    case "deleteShape":
      if (wasm) {
        wasm.instance.exports.deleteShape(data.index);
      }
      break;

    case "reorderShape":
      if (wasm) {
        wasm.instance.exports.reorderShape(data.fromIndex, data.toIndex);
      }
      break;

    case "groupSelected":
      if (wasm) {
        const groupId = wasm.instance.exports.groupSelectedShapes();
        self.postMessage({ type: "groupCreated", groupId });
      }
      break;

    case "ungroupShape":
      if (wasm) {
        wasm.instance.exports.ungroupShape(data.index);
      }
      break;

    case "updateSelectedStyle":
      if (wasm) {
        wasm.instance.exports.updateSelectedStyle(
          data.fillEnabled ? 1 : 0,
          data.fillR,
          data.fillG,
          data.fillB,
          data.strokeEnabled ? 1 : 0,
          data.strokeR,
          data.strokeG,
          data.strokeB,
          data.strokeWidth || 0.01, // Default to 0.01 NDC (2px)
          data.cornerRadius || 0,
          data.strokeCap || 0, // 0=butt, 1=round, 2=square
          data.strokeJoin || 0, // 0=miter, 1=round, 2=bevel
          data.dashLength || 0, // 0 = solid line
          data.dashGap || 0
        );
      }
      break;

    case "moveSelectedShapes":
      if (wasm) {
        wasm.instance.exports.moveSelectedShapes(data.dx, data.dy);
      }
      break;

    case "setShapeTransform":
      if (wasm) {
        wasm.instance.exports.setShapeTransform(
          data.index,
          data.x,
          data.y,
          data.width,
          data.height
        );
      }
      break;

    case "undo":
      if (wasm) {
        wasm.instance.exports.undo();
      }
      break;

    case "redo":
      if (wasm) {
        wasm.instance.exports.redo();
      }
      break;

    case "setDebugTiles":
      debugTiles = data.enabled;
      console.log("Debug tiles set to:", debugTiles);
      break;

    case "setWireframeMode":
      wireframeMode = data.enabled;
      console.log("Wireframe mode set to:", wireframeMode);
      break;
  }
};
