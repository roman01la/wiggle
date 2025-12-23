const std = @import("std");

// ============================================================================
// External JavaScript functions
// ============================================================================

extern fn jsLog(ptr: [*]const u8, len: usize) void;

// ============================================================================
// Logging helper
// ============================================================================

fn log(comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const message = std.fmt.bufPrint(&buf, fmt, args) catch return;
    jsLog(message.ptr, message.len);
}

// ============================================================================
// 2D Transform System (Scene Graph Style)
// ============================================================================

// 2D affine transform represented as a 3x2 matrix (6 floats)
// | m00 m01 tx |   | a c tx |
// | m10 m11 ty | = | b d ty |
// | 0   0   1  |   (implicit row for homogeneous coords)
//
// Transform order: Scale -> Rotate -> Translate
// Applying to point: [x', y'] = [a*x + c*y + tx, b*x + d*y + ty]
const Mat3x2 = struct {
    // Column-major storage for GPU compatibility
    // [a, b, c, d, tx, ty] = [m00, m10, m01, m11, m02, m12]
    m: [6]f32,

    // Identity matrix
    pub const identity = Mat3x2{ .m = .{ 1, 0, 0, 1, 0, 0 } };

    // Create translation matrix
    pub fn translation(tx: f32, ty: f32) Mat3x2 {
        return .{ .m = .{ 1, 0, 0, 1, tx, ty } };
    }

    // Create rotation matrix (angle in radians)
    pub fn rotation(angle: f32) Mat3x2 {
        const c = @cos(angle);
        const s = @sin(angle);
        return .{ .m = .{ c, s, -s, c, 0, 0 } };
    }

    // Create scale matrix
    pub fn scale(sx: f32, sy: f32) Mat3x2 {
        return .{ .m = .{ sx, 0, 0, sy, 0, 0 } };
    }

    // Multiply two matrices: self * other
    pub fn mul(self: Mat3x2, other: Mat3x2) Mat3x2 {
        const a = self.m;
        const b = other.m;
        return .{
            .m = .{
                a[0] * b[0] + a[2] * b[1], // m00
                a[1] * b[0] + a[3] * b[1], // m10
                a[0] * b[2] + a[2] * b[3], // m01
                a[1] * b[2] + a[3] * b[3], // m11
                a[0] * b[4] + a[2] * b[5] + a[4], // tx
                a[1] * b[4] + a[3] * b[5] + a[5], // ty
            },
        };
    }

    // Transform a point
    pub fn transformPoint(self: Mat3x2, x: f32, y: f32) [2]f32 {
        return .{
            self.m[0] * x + self.m[2] * y + self.m[4],
            self.m[1] * x + self.m[3] * y + self.m[5],
        };
    }

    // Get translation component
    pub fn getTranslation(self: Mat3x2) [2]f32 {
        return .{ self.m[4], self.m[5] };
    }

    // Get rotation angle (assumes uniform scale or no shear)
    pub fn getRotation(self: Mat3x2) f32 {
        return std.math.atan2(self.m[1], self.m[0]);
    }

    // Get scale (assumes no shear)
    pub fn getScale(self: Mat3x2) [2]f32 {
        return .{
            @sqrt(self.m[0] * self.m[0] + self.m[1] * self.m[1]),
            @sqrt(self.m[2] * self.m[2] + self.m[3] * self.m[3]),
        };
    }
};

// ============================================================================
// Input state (updated from JavaScript)
// ============================================================================

var mouse_x: f32 = 0.5;
var mouse_y: f32 = 0.5;
var mouse_pressed: bool = false;
var prev_mouse_pressed: bool = false;
var canvas_width: f32 = 800;
var canvas_height: f32 = 600;
var current_tool: u32 = 2; // 0 = circle, 1 = rect, 2 = select, 3 = pen

// Pen tool state
var pen_drawing: bool = false; // Currently adding points to a path
var pen_path_index: usize = 0; // Index of path being drawn

// Bezier handle editing state (Figma-style)
var pen_selected_point: i32 = -1; // Currently selected point for handle editing (-1 = none)
var pen_selected_path: i32 = -1; // Path containing the selected point
var pen_dragging_handle: bool = false; // Currently dragging a bezier handle
var pen_handle_type: u32 = 0; // 0 = out handle, 1 = in handle
var pen_symmetric_handles: bool = true; // Whether to mirror handle movement

// Selection state
var selection_box_active: bool = false;
var selection_box_start_x: f32 = 0.0;
var selection_box_start_y: f32 = 0.0;
var multi_selection: [MAX_SHAPES]bool = [_]bool{false} ** MAX_SHAPES;

// Dirty flag for signaling changes to the UI (layers panel sync)
var shapes_dirty: bool = false;

// ============================================================================
// Tile-Based Rendering
// ============================================================================

// Tile configuration (in NDC space, -1 to 1)
const TILE_SIZE: f32 = 0.25; // Each tile covers 0.25 NDC units (about 256px at 1024px canvas)
const MAX_TILES_X: usize = 16; // Max tiles horizontally
const MAX_TILES_Y: usize = 16; // Max tiles vertically
const MAX_TILES: usize = MAX_TILES_X * MAX_TILES_Y;

// Tile dirty state - true if tile needs re-rendering
var tile_dirty: [MAX_TILES]bool = [_]bool{true} ** MAX_TILES;
var tiles_dirty_count: usize = MAX_TILES; // How many tiles are dirty

// Mark a specific tile as dirty (tile coords in grid space)
fn markTileDirty(tile_x: usize, tile_y: usize) void {
    if (tile_x < MAX_TILES_X and tile_y < MAX_TILES_Y) {
        const idx = tile_y * MAX_TILES_X + tile_x;
        if (!tile_dirty[idx]) {
            tile_dirty[idx] = true;
            tiles_dirty_count += 1;
        }
    }
}

// Mark all tiles that a shape overlaps as dirty
fn markShapeTilesDirty(shape_x: f32, shape_y: f32, shape_w: f32, shape_h: f32) void {
    // Convert shape bounds from NDC (-1 to 1) to tile grid coords
    const min_x = shape_x - shape_w;
    const max_x = shape_x + shape_w;
    const min_y = shape_y - shape_h;
    const max_y = shape_y + shape_h;

    // Convert to normalized 0-1 range, then to tile indices
    // Clamp to valid range to avoid integer overflow
    const norm_min_x = @max(0.0, @min(1.0, (min_x + 1.0) / 2.0));
    const norm_max_x = @max(0.0, @min(1.0, (max_x + 1.0) / 2.0));
    const norm_min_y = @max(0.0, @min(1.0, (min_y + 1.0) / 2.0));
    const norm_max_y = @max(0.0, @min(1.0, (max_y + 1.0) / 2.0));

    const tile_min_x = @as(usize, @intFromFloat(norm_min_x * @as(f32, MAX_TILES_X - 1)));
    const tile_max_x = @as(usize, @intFromFloat(norm_max_x * @as(f32, MAX_TILES_X - 1)));
    const tile_min_y = @as(usize, @intFromFloat(norm_min_y * @as(f32, MAX_TILES_Y - 1)));
    const tile_max_y = @as(usize, @intFromFloat(norm_max_y * @as(f32, MAX_TILES_Y - 1)));

    // Mark all overlapping tiles as dirty
    var ty = tile_min_y;
    while (ty <= tile_max_y) : (ty += 1) {
        var tx = tile_min_x;
        while (tx <= tile_max_x) : (tx += 1) {
            markTileDirty(tx, ty);
        }
    }
}

// Mark all tiles as dirty (full redraw)
fn markAllTilesDirty() void {
    for (0..MAX_TILES) |i| {
        tile_dirty[i] = true;
    }
    tiles_dirty_count = MAX_TILES;
}

// Clear all tile dirty flags
fn clearAllTilesDirty() void {
    for (0..MAX_TILES) |i| {
        tile_dirty[i] = false;
    }
    tiles_dirty_count = 0;
}

// Exports for tile system
export fn getTileSize() f32 {
    return TILE_SIZE;
}

export fn getMaxTilesX() usize {
    return MAX_TILES_X;
}

export fn getMaxTilesY() usize {
    return MAX_TILES_Y;
}

export fn isTileDirty(tile_x: usize, tile_y: usize) u32 {
    if (tile_x >= MAX_TILES_X or tile_y >= MAX_TILES_Y) return 0;
    const idx = tile_y * MAX_TILES_X + tile_x;
    return if (tile_dirty[idx]) 1 else 0;
}

export fn getDirtyTileCount() usize {
    return tiles_dirty_count;
}

export fn clearTileDirty(tile_x: usize, tile_y: usize) void {
    if (tile_x >= MAX_TILES_X or tile_y >= MAX_TILES_Y) return;
    const idx = tile_y * MAX_TILES_X + tile_x;
    if (tile_dirty[idx]) {
        tile_dirty[idx] = false;
        if (tiles_dirty_count > 0) tiles_dirty_count -= 1;
    }
}

export fn clearAllTiles() void {
    clearAllTilesDirty();
}

// Track previous zoom for detecting zoom changes (zoom affects vertex generation)
var prev_zoom: f32 = 1.0;

// Track number of scene vertices vs UI overlay vertices (for debugging/metrics)
var scene_vertex_count: usize = 0;
var ui_vertex_count: usize = 0;

// Flag to indicate scene was regenerated this frame (for JS buffer upload optimization)
var scene_was_regenerated: bool = true;

fn markDirty() void {
    shapes_dirty = true;
    scene_cache_valid = false;
    markAllTilesDirty(); // All tiles need redraw when shapes change
}

// Mark dirty but only for specific shape's tiles (more granular)
fn markDirtyForShape(idx: usize) void {
    shapes_dirty = true;
    scene_cache_valid = false;
    const shape = shapes[idx];
    markShapeTilesDirty(shape.x, shape.y, shape.width, shape.height);
}

// Returns the number of scene vertices (shapes) vs overlay vertices (UI)
export fn getSceneVertexCount() usize {
    return scene_vertex_count;
}

export fn getUIVertexCount() usize {
    return ui_vertex_count;
}

export fn isSceneCacheValid() u32 {
    return if (scene_cache_valid) 1 else 0;
}

export fn wasSceneRegenerated() u32 {
    return if (scene_was_regenerated) 1 else 0;
}

export fn getCachedSceneVertexCount() usize {
    return cached_scene_vertex_count;
}

// Batch info exports for potential multi-draw-call optimization
export fn getBatchRectFillStart() usize {
    return rect_fill_batch.start_vertex;
}
export fn getBatchRectFillCount() usize {
    return rect_fill_batch.vertex_count;
}
export fn getBatchEllipseFillStart() usize {
    return ellipse_fill_batch.start_vertex;
}
export fn getBatchEllipseFillCount() usize {
    return ellipse_fill_batch.vertex_count;
}
export fn getBatchRectStrokeStart() usize {
    return rect_stroke_batch.start_vertex;
}
export fn getBatchRectStrokeCount() usize {
    return rect_stroke_batch.vertex_count;
}
export fn getBatchEllipseStrokeStart() usize {
    return ellipse_stroke_batch.start_vertex;
}
export fn getBatchEllipseStrokeCount() usize {
    return ellipse_stroke_batch.vertex_count;
}

// ============================================================================
// GPU Compute Shape Descriptors
// ============================================================================

// Shape descriptor format for GPU compute (80 bytes)
// Matches the WGSL ShapeDescriptor struct - must account for vec4 alignment
// Uses a full 3x2 transform matrix for position, rotation, and scale
const ShapeDescriptor = extern struct {
    shape_type: u32, // 0=ellipse, 1=rect, 2=group (matches Shape struct)
    render_mode: u32, // 0=fill, 1=stroke
    width: f32, // half-width (in local space)
    height: f32, // half-height (in local space)
    stroke_width: f32, // stroke thickness
    corner_radius: f32, // corner radius for rectangles
    shape_index: u32, // original shape index for layer ordering
    _padding0: u32 = 0,
    // Transform matrix (3x2 affine): [a, b, c, d, tx, ty] - at offset 32
    // Transforms local coords to world: world = mat * local
    transform: [6]f32, // 24 bytes, ends at offset 56
    // Padding for vec4 alignment (vec4 requires 16-byte alignment, so offset 64)
    _padding1: u32 = 0,
    _padding2: u32 = 0,
    // Color at offset 64 (16-byte aligned for vec4<f32>)
    color_r: f32,
    color_g: f32,
    color_b: f32,
    color_a: f32,
};

// Path descriptor format for GPU compute (80 bytes = 20 floats/u32s)
// Paths are separate from shapes because they have variable-length point data
// Note: Larger than shapes due to additional path-specific fields
const PathDescriptor = extern struct {
    point_start: u32, // Start index in path_points buffer
    point_count: u32, // Number of anchor points
    closed: u32, // 1 if closed, 0 if open
    stroke_width: f32, // Stroke thickness
    color_r: f32, // RGBA color
    color_g: f32,
    color_b: f32,
    color_a: f32,
    stroke_cap: u32, // 0=butt, 1=round, 2=square
    stroke_join: u32, // 0=miter, 1=round, 2=bevel
    dash_length: f32, // Dash length (0 = solid)
    dash_gap: f32, // Gap length
    shape_index: u32, // original shape index for layer ordering
    _padding0: u32 = 0,
    // Transform matrix (3x2 affine): [a, b, c, d, tx, ty] at offset 56
    transform: [6]f32,
    // Padding to reach 96 bytes (aligned to 16 for vec4)
    _padding1: u32 = 0,
    _padding2: u32 = 0,
    _padding3: u32 = 0,
    _padding4: u32 = 0,
};

// Buffer for GPU shape descriptors
var shape_descriptors: [MAX_SHAPES * 2]ShapeDescriptor = undefined; // *2 for fill+stroke
var descriptor_count: usize = 0;

// Buffer for GPU path descriptors
var path_descriptors: [MAX_PATHS]PathDescriptor = undefined;
var path_descriptor_count: usize = 0;

// Flattened path data buffers for GPU (all paths' data concatenated)
// Each anchor point has: x, y, ctrl_in_x, ctrl_in_y, ctrl_out_x, ctrl_out_y, segment_type (7 floats)
// segment_type: 0.0 = line, 1.0 = cubic bezier (as float for GPU compatibility)
const PATH_POINT_STRIDE: usize = 7; // floats per point
var path_points_buffer: [MAX_PATHS * MAX_PATH_POINTS * PATH_POINT_STRIDE]f32 = undefined;
var path_points_count: usize = 0; // Total number of anchor points (not floats)

// Build shape descriptors for GPU compute
// Returns number of descriptors written
export fn buildShapeDescriptors() usize {
    descriptor_count = 0;

    for (0..shape_count) |i| {
        const shape = shapes[i];

        // Skip groups (type 2), paths (type 3), and invisible shapes
        // Paths are rendered via buildPathDescriptors instead
        if (shape.shape_type == 2 or shape.shape_type == 3 or !shape.visible) continue;

        // Note: Shapes with parent_id >= 0 are part of a group but still render individually
        // The group just provides logical grouping for selection/transform operations

        // Compute world transform (composing with parent transforms)
        const world_transform = shape.getWorldTransform(shapes[0..shape_count]);
        const world_pos = world_transform.getTranslation();

        // Check if visible on screen (basic culling using world position)
        const screen_min_x = (world_pos[0] - shape.width) * zoom + pan_x;
        const screen_max_x = (world_pos[0] + shape.width) * zoom + pan_x;
        const screen_min_y = (world_pos[1] - shape.height) * zoom + pan_y;
        const screen_max_y = (world_pos[1] + shape.height) * zoom + pan_y;

        if (screen_max_x < -1.5 or screen_min_x > 1.5 or
            screen_max_y < -1.5 or screen_min_y > 1.5)
        {
            continue;
        }

        // Stroke width is in world space (NDC), no zoom adjustment needed
        // It will scale naturally with the shape when zoomed
        const adjusted_stroke_width = shape.stroke_width;

        // Corner radius is in the same coordinate space as width/height, no zoom adjustment needed
        const adjusted_corner_radius = shape.corner_radius;

        // Add fill descriptor if enabled
        if (shape.fill_enabled) {
            shape_descriptors[descriptor_count] = .{
                .shape_type = shape.shape_type,
                .render_mode = 0, // fill
                .width = shape.width,
                .height = shape.height,
                .stroke_width = 0.0,
                .corner_radius = adjusted_corner_radius,
                .shape_index = @intCast(i),
                .transform = world_transform.m,
                .color_r = shape.fill_color[0],
                .color_g = shape.fill_color[1],
                .color_b = shape.fill_color[2],
                .color_a = 1.0,
            };
            descriptor_count += 1;
        }

        // Add stroke descriptor if enabled
        if (shape.stroke_enabled) {
            shape_descriptors[descriptor_count] = .{
                .shape_type = shape.shape_type,
                .render_mode = 1, // stroke
                .width = shape.width,
                .height = shape.height,
                .stroke_width = adjusted_stroke_width,
                .corner_radius = adjusted_corner_radius,
                .shape_index = @intCast(i),
                .transform = world_transform.m,
                .color_r = shape.stroke_color[0],
                .color_g = shape.stroke_color[1],
                .color_b = shape.stroke_color[2],
                .color_a = 1.0,
            };
            descriptor_count += 1;
        }
    }

    return descriptor_count;
}

// Build path descriptors for GPU compute
// Iterates over shapes that reference paths to get their style settings
export fn buildPathDescriptors() usize {
    path_descriptor_count = 0;
    path_points_count = 0;

    for (0..shape_count) |i| {
        const shape = shapes[i];
        // Skip non-path shapes, invisible shapes, and shapes without stroke
        if (shape.shape_type != 3 or !shape.visible or !shape.stroke_enabled) continue;

        const path_idx = shape.path_index;
        if (path_idx >= path_count) {
            log("Path shape {d} has invalid path_index {d} >= path_count {d}", .{ i, path_idx, path_count });
            continue;
        }

        const path = paths[path_idx];
        if (path.point_count < 2) {
            log("Path {d} has only {d} points, skipping", .{ path_idx, path.point_count });
            continue; // Need at least 2 points
        }

        // Copy points to the flattened buffer with bezier data
        // Format: x, y, ctrl_in_x, ctrl_in_y, ctrl_out_x, ctrl_out_y, segment_type
        const point_start = path_points_count;
        for (0..path.point_count) |pt_idx| {
            const base = path_points_count * PATH_POINT_STRIDE;
            path_points_buffer[base + 0] = path.points[pt_idx][0]; // x
            path_points_buffer[base + 1] = path.points[pt_idx][1]; // y
            path_points_buffer[base + 2] = path.control_points[pt_idx][0][0]; // ctrl_in_x
            path_points_buffer[base + 3] = path.control_points[pt_idx][0][1]; // ctrl_in_y
            path_points_buffer[base + 4] = path.control_points[pt_idx][1][0]; // ctrl_out_x
            path_points_buffer[base + 5] = path.control_points[pt_idx][1][1]; // ctrl_out_y
            path_points_buffer[base + 6] = @floatFromInt(@intFromEnum(path.segment_types[pt_idx])); // segment_type
            path_points_count += 1;
        }

        // Stroke width is in world space (NDC), no zoom adjustment needed
        // It will scale naturally with the path when zoomed
        const adjusted_stroke_width = shape.stroke_width;

        // Compute world transform (composing with parent transforms)
        const world_transform = shape.getWorldTransform(shapes[0..shape_count]);

        path_descriptors[path_descriptor_count] = .{
            .point_start = @intCast(point_start),
            .point_count = path.point_count,
            .closed = if (path.closed) @as(u32, 1) else 0,
            .stroke_width = adjusted_stroke_width,
            .color_r = shape.stroke_color[0],
            .color_g = shape.stroke_color[1],
            .color_b = shape.stroke_color[2],
            .color_a = 1.0,
            // Stroke styling
            .stroke_cap = @intFromEnum(shape.stroke_cap),
            .stroke_join = @intFromEnum(shape.stroke_join),
            .dash_length = shape.dash_array[0],
            .dash_gap = shape.dash_array[1],
            .shape_index = @intCast(i),
            .transform = world_transform.m,
        };
        path_descriptor_count += 1;
    }

    return path_descriptor_count;
}

// Get pointer to shape descriptor buffer for JS to read
export fn getShapeDescriptorPtr() [*]const u8 {
    return @ptrCast(&shape_descriptors);
}

// Get number of shape descriptors
export fn getShapeDescriptorCount() usize {
    return descriptor_count;
}

// Size of each descriptor in bytes
export fn getShapeDescriptorSize() usize {
    return @sizeOf(ShapeDescriptor);
}

// Path descriptor exports
export fn getPathDescriptorPtr() [*]const u8 {
    return @ptrCast(&path_descriptors);
}

export fn getPathDescriptorCount() usize {
    return path_descriptor_count;
}

export fn getPathDescriptorSize() usize {
    return @sizeOf(PathDescriptor);
}

// Path points buffer exports
export fn getPathPointsPtr() [*]const u8 {
    return @ptrCast(&path_points_buffer);
}

export fn getPathPointsCount() usize {
    return path_points_count;
}

export fn getPathPointStride() usize {
    return PATH_POINT_STRIDE;
}

export fn isShapesDirty() u32 {
    return if (shapes_dirty) 1 else 0;
}

export fn clearShapesDirty() void {
    shapes_dirty = false;
}

// Current style settings
var fill_enabled: bool = false;
var fill_color: [3]f32 = .{ 1.0, 1.0, 1.0 };
var stroke_enabled: bool = true;
var stroke_color: [3]f32 = .{ 0.0, 0.0, 0.0 };
var stroke_width: f32 = 0.01; // Stroke width in NDC (default ~2px at 1024px canvas)
var corner_radius: f32 = 0.0; // Corner radius in NDC (for rectangles)

// Zoom and pan state
var zoom: f32 = 1.0;
var pan_x: f32 = 0.0;
var pan_y: f32 = 0.0;
var meta_key_pressed: bool = false;
var shift_key_pressed: bool = false;

// Drag state for shape creation
var is_dragging: bool = false;
var drag_start_x: f32 = 0.0;
var drag_start_y: f32 = 0.0;

// Drag state for moving selected shapes
var is_moving_shapes: bool = false;
var move_start_x: f32 = 0.0;
var move_start_y: f32 = 0.0;
var prev_move_x: f32 = 0.0;
var prev_move_y: f32 = 0.0;

// Resize state
// Handle: 0=none, 1=left, 2=right, 3=top, 4=bottom, 5=top-left, 6=top-right, 7=bottom-left, 8=bottom-right
var is_resizing: bool = false;
var resize_handle: u8 = 0;
var resize_shape_idx: usize = 0;
var resize_anchor_x: f32 = 0.0; // The edge that stays fixed horizontally
var resize_anchor_y: f32 = 0.0; // The edge that stays fixed vertically

// Rotation state
var is_rotating: bool = false;
var rotate_shape_idx: usize = 0;
var rotate_start_angle: f32 = 0.0; // Initial angle when rotation started
var rotate_shape_initial_rotation: f32 = 0.0; // Shape's rotation when drag started

export fn updateMousePosition(x: f32, y: f32) void {
    mouse_x = x / canvas_width;
    mouse_y = y / canvas_height;
}

export fn updateMousePressed(pressed: bool) void {
    mouse_pressed = pressed;
}

export fn updateCanvasSize(width: f32, height: f32) void {
    canvas_width = width;
    canvas_height = height;
}

export fn setTool(tool: u32) void {
    // If switching away from pen tool while drawing, finish the path
    if (current_tool == 3 and pen_drawing and tool != 3) {
        pen_drawing = false;
    }
    current_tool = tool;
}

export fn getCurrentTool() u32 {
    return current_tool;
}

// Finish current pen path (called on Enter or double-click)
export fn finishPenPath() void {
    if (pen_drawing) {
        pen_drawing = false;

        // Find the shape that references this path and select it
        for (0..shape_count) |i| {
            if (shapes[i].shape_type == 3 and shapes[i].path_index == pen_path_index) {
                clearSelection();
                multi_selection[i] = true;
                selected_shape = @intCast(i);
                break;
            }
        }

        // Switch to select tool
        current_tool = 2;

        markDirty();
        // Log path info for debugging
        if (pen_path_index < path_count) {
            const path = paths[pen_path_index];
            log("Pen path finished: index={d}, points={d}, closed={}", .{ pen_path_index, path.point_count, path.closed });
        }
        log("Total paths={d}, shapes={d}", .{ path_count, shape_count });
    }
}

// Cancel current pen path (called on Escape)
export fn cancelPenPath() void {
    if (pen_drawing) {
        // Remove the last path and shape
        if (path_count > 0) {
            path_count -= 1;
        }
        if (shape_count > 0) {
            shape_count -= 1;
        }
        pen_drawing = false;
        markDirty();
        log("Pen path cancelled", .{});
    }
}

// Check if currently drawing a pen path
export fn isPenDrawing() u32 {
    return if (pen_drawing) 1 else 0;
}

// Get current pen path index (for setting control points during drawing)
export fn getPenPathIndex() i32 {
    if (pen_drawing and pen_path_index < path_count) {
        return @intCast(pen_path_index);
    }
    return -1;
}

// Get selected point for bezier handle editing
export fn getSelectedPathPoint() i32 {
    return pen_selected_point;
}

export fn getSelectedPath() i32 {
    return pen_selected_path;
}

// Clear bezier point selection
export fn clearPathPointSelection() void {
    pen_selected_point = -1;
    pen_selected_path = -1;
    markDirty();
}

// Set control points for a point in a path (for bezier curves)
// ctrl_in: incoming control handle (from previous point)
// ctrl_out: outgoing control handle (to next point)
export fn setPathPointControlPoints(path_idx: u32, point_idx: u32, ctrl_in_x: f32, ctrl_in_y: f32, ctrl_out_x: f32, ctrl_out_y: f32) void {
    if (path_idx >= path_count) return;
    const path = &paths[path_idx];
    if (point_idx >= path.point_count) return;

    path.control_points[point_idx][0] = .{ ctrl_in_x, ctrl_in_y };
    path.control_points[point_idx][1] = .{ ctrl_out_x, ctrl_out_y };
    markDirty();
}

// Set segment type for a point (determines how segment from this point to next is drawn)
// 0 = line, 1 = cubic bezier
export fn setPathSegmentType(path_idx: u32, point_idx: u32, seg_type: u32) void {
    if (path_idx >= path_count) return;
    const path = &paths[path_idx];
    if (point_idx >= path.point_count) return;

    path.segment_types[point_idx] = if (seg_type == 1) .cubic_bezier else .line;
    markDirty();
}

// Get path data for GPU rendering
export fn getPathPtr() [*]const u8 {
    return @ptrCast(&paths);
}

export fn getPathCount() usize {
    return path_count;
}

export fn getPathSize() usize {
    return @sizeOf(Path);
}

export fn setStyle(fill_en: u32, fill_r: f32, fill_g: f32, fill_b: f32, stroke_en: u32, stroke_r: f32, stroke_g: f32, stroke_b: f32, stroke_w: f32, corner_r: f32) void {
    fill_enabled = fill_en != 0;
    fill_color = .{ fill_r, fill_g, fill_b };
    stroke_enabled = stroke_en != 0;
    stroke_color = .{ stroke_r, stroke_g, stroke_b };
    stroke_width = stroke_w;
    corner_radius = corner_r;
}

export fn setZoom(z: f32, px: f32, py: f32) void {
    zoom = z;
    pan_x = px;
    pan_y = py;
}

export fn getZoom() f32 {
    return zoom;
}

export fn setMetaKey(pressed: u32) void {
    meta_key_pressed = pressed != 0;
}

export fn setShiftKey(pressed: u32) void {
    shift_key_pressed = pressed != 0;
}

// ============================================================================
// 2D Shape Rendering
// ============================================================================

const Vertex = extern struct {
    position: [2]f32,
    color: [4]f32,
};

// Circle segments - base and max for dynamic resolution
const MIN_CIRCLE_SEGMENTS: usize = 64;
const MAX_CIRCLE_SEGMENTS: usize = 256;

// Rectangle: 2 triangles = 6 vertices
const RECT_VERTICES: usize = 6;

// Support multiple shapes - increased for complex scenes
const MAX_SHAPES: usize = 10000;
// Each shape can have fill + stroke with up to MAX_CIRCLE_SEGMENTS
// For memory efficiency, we use a more conservative estimate
const MAX_VERTICES: usize = MAX_SHAPES * 64 * 6 * 2 + MAX_CIRCLE_SEGMENTS * 6 * 2;

// Main vertex buffer (scene + UI overlays)
var vertices: [MAX_VERTICES]Vertex = undefined;
var vertex_count: usize = 0;

// Cached scene vertices - reused when only pan changes
var cached_scene_vertices: [MAX_VERTICES]Vertex = undefined;
var cached_scene_vertex_count: usize = 0;
var scene_cache_valid: bool = false;

// Batch tracking for draw call optimization
// Vertices are organized: [rect fills | rect strokes | ellipse fills | ellipse strokes]
const BatchInfo = struct {
    start_vertex: usize,
    vertex_count: usize,
};
var rect_fill_batch: BatchInfo = .{ .start_vertex = 0, .vertex_count = 0 };
var rect_stroke_batch: BatchInfo = .{ .start_vertex = 0, .vertex_count = 0 };
var ellipse_fill_batch: BatchInfo = .{ .start_vertex = 0, .vertex_count = 0 };
var ellipse_stroke_batch: BatchInfo = .{ .start_vertex = 0, .vertex_count = 0 };

// Stroke cap styles (how line endpoints are drawn)
const StrokeCap = enum(u8) {
    butt = 0, // Flat end at the exact endpoint
    round = 1, // Semicircle extending past endpoint
    square = 2, // Rectangle extending past endpoint
};

// Stroke join styles (how corners between segments are drawn)
const StrokeJoin = enum(u8) {
    miter = 0, // Sharp corner (with miter limit)
    round = 1, // Rounded corner
    bevel = 2, // Flat corner
};

// Stored shapes - using transform-based architecture
// Position and rotation are stored as local transform relative to parent
const Shape = struct {
    shape_type: u32, // 0 = circle, 1 = rect, 2 = group, 3 = path
    // Local transform components (relative to parent, or world if no parent)
    x: f32, // center x position
    y: f32, // center y position
    rotation: f32 = 0.0, // rotation in radians (counter-clockwise)
    scale_x: f32 = 1.0, // scale x (for future use)
    scale_y: f32 = 1.0, // scale y (for future use)
    // Geometry dimensions (in local space, centered at origin)
    width: f32, // half-width (or radius for circle)
    height: f32, // half-height (ignored for circle)
    fill_enabled: bool,
    fill_color: [3]f32,
    stroke_enabled: bool,
    stroke_color: [3]f32,
    stroke_width: f32, // stroke width in NDC (each shape has its own)
    parent_id: i32, // -1 = no parent, otherwise index of parent shape
    visible: bool, // whether the shape is visible in the layers panel
    corner_radius: f32, // corner radius for rectangles (in NDC)
    path_index: u32, // index into paths array (for shape_type == 3)
    // Stroke styling for paths
    stroke_cap: StrokeCap = .butt, // How line endpoints are drawn
    stroke_join: StrokeJoin = .miter, // How corners are drawn
    // Dash pattern: [dash_length, gap_length] in NDC units. [0,0] = solid
    dash_array: [2]f32 = .{ 0, 0 },
    dash_offset: f32 = 0, // Offset into dash pattern

    // Convert local transform to matrix: T * R * S
    pub fn getLocalTransform(self: Shape) Mat3x2 {
        // For paths (shape_type 3), points are stored in absolute world coordinates,
        // so we only apply rotation/scale around the shape's x,y origin (not translation)
        // The x,y for paths represents the transform origin, not an offset
        if (self.shape_type == 3) {
            // For paths: translate to origin, apply rotation/scale, translate back
            // This allows rotation around the path's reference point without offsetting
            if (self.rotation == 0 and self.scale_x == 1 and self.scale_y == 1) {
                return Mat3x2.identity;
            }
            const to_origin = Mat3x2.translation(-self.x, -self.y);
            const s = Mat3x2.scale(self.scale_x, self.scale_y);
            const r = Mat3x2.rotation(self.rotation);
            const from_origin = Mat3x2.translation(self.x, self.y);
            return from_origin.mul(r.mul(s.mul(to_origin)));
        }
        // For other shapes: standard T * R * S composition
        const s = Mat3x2.scale(self.scale_x, self.scale_y);
        const r = Mat3x2.rotation(self.rotation);
        const t = Mat3x2.translation(self.x, self.y);
        return t.mul(r.mul(s));
    }

    // Get world transform by composing with parent transforms
    // Note: In this system, all shapes store ABSOLUTE positions, not relative to parent.
    // So we only inherit rotation/scale from parent, not translation.
    pub fn getWorldTransform(self: Shape, all_shapes: []const Shape) Mat3x2 {
        const local = self.getLocalTransform();
        if (self.parent_id < 0) {
            return local;
        }
        const parent_idx = @as(usize, @intCast(self.parent_id));
        if (parent_idx >= all_shapes.len) {
            return local;
        }
        const parent = all_shapes[parent_idx];

        // Get parent's world rotation and scale (without translation)
        // We need to apply parent's rotation/scale around the parent's world position
        if (parent.rotation == 0 and parent.scale_x == 1 and parent.scale_y == 1) {
            // No rotation/scale from parent, just use local transform
            return local;
        }

        // Get parent's world transform to extract its position
        const parent_world = parent.getWorldTransform(all_shapes);
        const parent_pos = parent_world.getTranslation();

        // Build transform: translate child to parent origin, apply parent rot/scale, translate back
        // Then apply child's local transform
        const to_parent = Mat3x2.translation(-parent_pos[0], -parent_pos[1]);
        const parent_rs = Mat3x2.rotation(parent.rotation).mul(Mat3x2.scale(parent.scale_x, parent.scale_y));
        const from_parent = Mat3x2.translation(parent_pos[0], parent_pos[1]);

        // Compose: from_parent * parent_rs * to_parent * local
        return from_parent.mul(parent_rs.mul(to_parent.mul(local)));
    }

    // Get world position (convenience method)
    pub fn getWorldPosition(self: Shape, all_shapes: []const Shape) [2]f32 {
        const world = self.getWorldTransform(all_shapes);
        return world.getTranslation();
    }
};

// Path data structure - stores points and segments for pen tool paths
// Supports both line segments and cubic bezier curves
const MAX_PATH_POINTS: usize = 256; // Max anchor points per path
const MAX_PATHS: usize = 128; // Max number of paths

// Segment types for path rendering
const SegmentType = enum(u8) {
    line = 0, // Straight line to next point
    cubic_bezier = 1, // Cubic bezier curve (uses control points)
};

const Path = struct {
    // Anchor points (on-curve points)
    // Zeroed by default to prevent stale data from appearing after delete/create cycles
    points: [MAX_PATH_POINTS][2]f32 = [_][2]f32{.{ 0, 0 }} ** MAX_PATH_POINTS,
    // Control points for bezier curves (2 per anchor point: in-handle and out-handle)
    // control_points[i][0] = in-handle for point i (from previous segment)
    // control_points[i][1] = out-handle for point i (to next segment)
    control_points: [MAX_PATH_POINTS][2][2]f32 = [_][2][2]f32{.{ .{ 0, 0 }, .{ 0, 0 } }} ** MAX_PATH_POINTS,
    // Segment type for each segment (segment i connects point i to point i+1)
    segment_types: [MAX_PATH_POINTS]SegmentType = [_]SegmentType{.line} ** MAX_PATH_POINTS,
    point_count: u32 = 0,
    closed: bool = false, // Whether path is closed (connects last point to first)
};

var paths: [MAX_PATHS]Path = [_]Path{.{}} ** MAX_PATHS;
var path_count: usize = 0;

var shapes: [MAX_SHAPES]Shape = undefined;
var shape_count: usize = 0;
var selected_shape: i32 = -1; // Currently selected shape index

// ============================================================================
// Undo/Redo History
// ============================================================================

// Reduced history depth for memory efficiency with large shape counts
const MAX_HISTORY: usize = 100;

const HistorySnapshot = struct {
    shapes: [MAX_SHAPES]Shape,
    shape_count: usize,
    selected_shape: i32,
    multi_selection: [MAX_SHAPES]bool,
};

var history: [MAX_HISTORY]HistorySnapshot = undefined;
var history_count: usize = 0;
var history_index: usize = 0; // Points to the current state (last undo point)

// Track if we're in the middle of an operation (to avoid creating snapshots during move/resize)
var operation_in_progress: bool = false;

fn pushHistorySnapshot() void {
    // If we're in the middle of the history (after some undos), discard redo states
    if (history_index < history_count) {
        history_count = history_index;
    }

    // If history is full, shift everything down
    if (history_count >= MAX_HISTORY) {
        for (0..MAX_HISTORY - 1) |i| {
            history[i] = history[i + 1];
        }
        history_count = MAX_HISTORY - 1;
    }

    // Save current state
    history[history_count] = .{
        .shapes = shapes,
        .shape_count = shape_count,
        .selected_shape = selected_shape,
        .multi_selection = multi_selection,
    };
    history_count += 1;
    history_index = history_count;
}

fn restoreSnapshot(snapshot: *const HistorySnapshot) void {
    shapes = snapshot.shapes;
    shape_count = snapshot.shape_count;
    selected_shape = snapshot.selected_shape;
    multi_selection = snapshot.multi_selection;
    markDirty();
}

export fn undo() void {
    if (history_index > 1) {
        history_index -= 1;
        restoreSnapshot(&history[history_index - 1]);
    }
}

export fn redo() void {
    if (history_index < history_count) {
        restoreSnapshot(&history[history_index]);
        history_index += 1;
    }
}

export fn canUndo() u32 {
    return if (history_index > 1) 1 else 0;
}

export fn canRedo() u32 {
    return if (history_index < history_count) 1 else 0;
}

// Call this before starting an operation (move, resize, etc.)
export fn beginOperation() void {
    if (!operation_in_progress) {
        operation_in_progress = true;
        pushHistorySnapshot();
    }
}

// Call this after ending an operation
export fn endOperation() void {
    operation_in_progress = false;
}

// Uniform data for the shader
const Uniforms = extern struct {
    time: f32,
    mouse_x: f32,
    mouse_y: f32,
    aspect_ratio: f32,
    // View transform uniforms
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    _padding: f32, // Align to 16 bytes for WebGPU
};

var uniforms: Uniforms = .{
    .time = 0.0,
    .mouse_x = 0.5,
    .mouse_y = 0.5,
    .aspect_ratio = 1.0,
    .zoom = 1.0,
    .pan_x = 0.0,
    .pan_y = 0.0,
    ._padding = 0.0,
};

// ============================================================================
// Shape generation functions
// ============================================================================

// Calculate dynamic segment count based on shape size and zoom
fn calcSegments(radius_x: f32, radius_y: f32) usize {
    // Larger radius = more segments, more zoom = more segments
    const max_radius = @max(radius_x, radius_y);
    const screen_size = max_radius * zoom * 2.0;
    const base_segments: f32 = screen_size * 200.0;
    const clamped = @max(@as(f32, MIN_CIRCLE_SEGMENTS), @min(@as(f32, MAX_CIRCLE_SEGMENTS), base_segments));
    return @as(usize, @intFromFloat(clamped));
}

// Transform a point from world space to screen space (for bounds checking only)
fn transformToScreen(x: f32, y: f32) [2]f32 {
    const aspect = canvas_width / canvas_height;
    const screen_x = (x * zoom + pan_x) / aspect;
    const screen_y = y * zoom + pan_y;
    return .{ screen_x, screen_y };
}

// Output vertices in world space - shader will apply zoom/pan transform
fn worldToVertex(x: f32, y: f32) [2]f32 {
    return .{ x, y };
}

fn addEllipseFill(cx: f32, cy: f32, radius_x: f32, radius_y: f32, r: f32, g: f32, b: f32, a: f32) void {
    const segments = calcSegments(radius_x, radius_y);
    const angle_step = (2.0 * std.math.pi) / @as(f32, @floatFromInt(segments));

    const center = worldToVertex(cx, cy);

    for (0..segments) |i| {
        const angle1 = @as(f32, @floatFromInt(i)) * angle_step;
        const angle2 = @as(f32, @floatFromInt(i + 1)) * angle_step;

        const p1 = worldToVertex(cx + @cos(angle1) * radius_x, cy + @sin(angle1) * radius_y);
        const p2 = worldToVertex(cx + @cos(angle2) * radius_x, cy + @sin(angle2) * radius_y);

        // Center vertex
        vertices[vertex_count] = .{
            .position = center,
            .color = .{ r, g, b, a },
        };
        vertex_count += 1;

        // First edge vertex
        vertices[vertex_count] = .{
            .position = p1,
            .color = .{ r, g, b, a },
        };
        vertex_count += 1;

        // Second edge vertex
        vertices[vertex_count] = .{
            .position = p2,
            .color = .{ r, g, b, a },
        };
        vertex_count += 1;
    }
}

fn addRectangle(x1: f32, y1: f32, x2: f32, y2: f32, r: f32, g: f32, b: f32, a: f32) void {
    const p1 = worldToVertex(x1, y1);
    const p2 = worldToVertex(x2, y1);
    const p3 = worldToVertex(x2, y2);
    const p4 = worldToVertex(x1, y2);

    // Triangle 1
    vertices[vertex_count] = .{ .position = p1, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = p2, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = p3, .color = .{ r, g, b, a } };
    vertex_count += 1;

    // Triangle 2
    vertices[vertex_count] = .{ .position = p1, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = p3, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = p4, .color = .{ r, g, b, a } };
    vertex_count += 1;
}

// Stroke rendering for ellipse (ring) with uniform thickness
fn addEllipseStroke(cx: f32, cy: f32, radius_x: f32, radius_y: f32, thickness: f32, r: f32, g: f32, b: f32, a: f32) void {
    const segments = calcSegments(radius_x, radius_y);
    const angle_step = (2.0 * std.math.pi) / @as(f32, @floatFromInt(segments));

    for (0..segments) |i| {
        const angle1 = @as(f32, @floatFromInt(i)) * angle_step;
        const angle2 = @as(f32, @floatFromInt(i + 1)) * angle_step;

        const cos1 = @cos(angle1);
        const sin1 = @sin(angle1);
        const cos2 = @cos(angle2);
        const sin2 = @sin(angle2);

        // Outer vertices on the ellipse
        const outer_x1 = cx + cos1 * radius_x;
        const outer_y1 = cy + sin1 * radius_y;
        const outer_x2 = cx + cos2 * radius_x;
        const outer_y2 = cy + sin2 * radius_y;

        // For an ellipse (x/a)² + (y/b)² = 1, the outward normal at point (x,y) is
        // proportional to (x/a², y/b²). We need to normalize this.
        // At parametric point (a*cos(t), b*sin(t)), the unnormalized normal is (cos(t)/a, sin(t)/b)
        const nx1_unnorm = cos1 / radius_x;
        const ny1_unnorm = sin1 / radius_y;
        const len1 = @sqrt(nx1_unnorm * nx1_unnorm + ny1_unnorm * ny1_unnorm);
        const nx1 = if (len1 > 0.0001) nx1_unnorm / len1 else 0.0;
        const ny1 = if (len1 > 0.0001) ny1_unnorm / len1 else 0.0;

        const nx2_unnorm = cos2 / radius_x;
        const ny2_unnorm = sin2 / radius_y;
        const len2 = @sqrt(nx2_unnorm * nx2_unnorm + ny2_unnorm * ny2_unnorm);
        const nx2 = if (len2 > 0.0001) nx2_unnorm / len2 else 0.0;
        const ny2 = if (len2 > 0.0001) ny2_unnorm / len2 else 0.0;

        // Inner vertices (offset inward by thickness along the true normal)
        const inner_x1 = outer_x1 - nx1 * thickness;
        const inner_y1 = outer_y1 - ny1 * thickness;
        const inner_x2 = outer_x2 - nx2 * thickness;
        const inner_y2 = outer_y2 - ny2 * thickness;

        // Transform all points to vertex positions (world space)
        const outer1 = worldToVertex(outer_x1, outer_y1);
        const outer2 = worldToVertex(outer_x2, outer_y2);
        const inner1 = worldToVertex(inner_x1, inner_y1);
        const inner2 = worldToVertex(inner_x2, inner_y2);

        // Triangle 1: outer1, outer2, inner1
        vertices[vertex_count] = .{ .position = outer1, .color = .{ r, g, b, a } };
        vertex_count += 1;
        vertices[vertex_count] = .{ .position = outer2, .color = .{ r, g, b, a } };
        vertex_count += 1;
        vertices[vertex_count] = .{ .position = inner1, .color = .{ r, g, b, a } };
        vertex_count += 1;

        // Triangle 2: inner1, outer2, inner2
        vertices[vertex_count] = .{ .position = inner1, .color = .{ r, g, b, a } };
        vertex_count += 1;
        vertices[vertex_count] = .{ .position = outer2, .color = .{ r, g, b, a } };
        vertex_count += 1;
        vertices[vertex_count] = .{ .position = inner2, .color = .{ r, g, b, a } };
        vertex_count += 1;
    }
}

// Stroke rendering for rectangle (hollow) - uses addRectangle which handles transform
fn addRectangleStroke(x1: f32, y1: f32, x2: f32, y2: f32, thickness: f32, r: f32, g: f32, b: f32, a: f32) void {
    // Inner rectangle coords (adjusted for edge rendering)
    const iy1 = y1 + thickness;
    const iy2 = y2 - thickness;

    // Top edge
    addRectangle(x1, y2 - thickness, x2, y2, r, g, b, a);
    // Bottom edge
    addRectangle(x1, y1, x2, y1 + thickness, r, g, b, a);
    // Left edge (between top and bottom)
    addRectangle(x1, iy1, x1 + thickness, iy2, r, g, b, a);
    // Right edge (between top and bottom)
    addRectangle(x2 - thickness, iy1, x2, iy2, r, g, b, a);
}

// Add a single line segment with thickness (as a quad)
fn addLineSegment(x1: f32, y1: f32, x2: f32, y2: f32, thickness: f32, r: f32, g: f32, b: f32, a: f32) void {
    // Calculate perpendicular direction for line thickness
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = @sqrt(dx * dx + dy * dy);
    if (len < 0.0001) return;

    // Perpendicular unit vector
    const px = -dy / len * thickness * 0.5;
    const py = dx / len * thickness * 0.5;

    // Four corners of the line quad
    const p1 = worldToVertex(x1 + px, y1 + py);
    const p2 = worldToVertex(x1 - px, y1 - py);
    const p3 = worldToVertex(x2 + px, y2 + py);
    const p4 = worldToVertex(x2 - px, y2 - py);

    // Two triangles for the quad
    vertices[vertex_count] = .{ .position = p1, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = p2, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = p3, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = p2, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = p4, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = p3, .color = .{ r, g, b, a } };
    vertex_count += 1;
}

// Render a path as connected line segments
fn renderPath(path_idx: usize, r: f32, g: f32, b: f32, a: f32, thickness: f32) void {
    const path = paths[path_idx];
    if (path.point_count < 2) return;

    const start_verts = vertex_count;

    // Draw line segments between consecutive points
    for (0..path.point_count - 1) |i| {
        const p1 = path.points[i];
        const p2 = path.points[i + 1];
        addLineSegment(p1[0], p1[1], p2[0], p2[1], thickness, r, g, b, a);
    }

    // If closed, draw segment from last to first
    if (path.closed and path.point_count >= 2) {
        const last = path.points[path.point_count - 1];
        const first = path.points[0];
        addLineSegment(last[0], last[1], first[0], first[1], thickness, r, g, b, a);
    }

    log("renderPath: path_idx={d}, points={d}, added {d} vertices", .{ path_idx, path.point_count, vertex_count - start_verts });
}

// Render a point marker (small circle) for path vertices
fn renderPathPoint(x: f32, y: f32, size: f32, r: f32, g: f32, b: f32, a: f32) void {
    // Simple diamond shape for the point
    const left = worldToVertex(x - size, y);
    const top = worldToVertex(x, y - size);
    const right = worldToVertex(x + size, y);
    const bottom = worldToVertex(x, y + size);

    vertices[vertex_count] = .{ .position = left, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = top, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = right, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = left, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = right, .color = .{ r, g, b, a } };
    vertex_count += 1;
    vertices[vertex_count] = .{ .position = bottom, .color = .{ r, g, b, a } };
    vertex_count += 1;
}

// ============================================================================
// Memory access for JavaScript
// ============================================================================

export fn getVertexDataPtr() [*]const u8 {
    return @ptrCast(&vertices);
}

export fn getVertexDataSize() usize {
    return vertex_count * @sizeOf(Vertex);
}

export fn getVertexCount() usize {
    return vertex_count;
}

export fn getUniformDataPtr() [*]const u8 {
    return @ptrCast(&uniforms);
}

export fn getUniformDataSize() usize {
    return @sizeOf(Uniforms);
}

// ============================================================================
// Hit testing for selection
// ============================================================================

fn pointInShape(px: f32, py: f32, shape: Shape) bool {
    if (shape.shape_type == 0) {
        // Ellipse - check if point is inside ellipse equation
        const dx = px - shape.x;
        const dy = py - shape.y;
        const nx = dx / shape.width;
        const ny = dy / shape.height;
        return (nx * nx + ny * ny) <= 1.0;
    } else if (shape.shape_type == 3) {
        // Path - check if point is inside bounding box
        if (shape.path_index >= path_count) return false;

        const bounds = getPathBounds(shape.path_index);
        if (!bounds.has_points) return false;

        // Add small padding to make selection easier
        const padding: f32 = 0.02;
        return px >= bounds.min_x - padding and px <= bounds.max_x + padding and
            py >= bounds.min_y - padding and py <= bounds.max_y + padding;
    } else {
        // Rectangle - simple bounds check
        const x1 = shape.x - shape.width;
        const y1 = shape.y - shape.height;
        const x2 = shape.x + shape.width;
        const y2 = shape.y + shape.height;
        return px >= x1 and px <= x2 and py >= y1 and py <= y2;
    }
}

// Calculate distance from point (px, py) to line segment (x1, y1) - (x2, y2)
fn pointToSegmentDistance(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) f32 {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len_sq = dx * dx + dy * dy;

    if (len_sq < 0.0001) {
        // Segment is basically a point
        const dpx = px - x1;
        const dpy = py - y1;
        return @sqrt(dpx * dpx + dpy * dpy);
    }

    // Project point onto line, clamped to segment
    var t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    t = @max(0.0, @min(1.0, t));

    // Find closest point on segment
    const closest_x = x1 + t * dx;
    const closest_y = y1 + t * dy;

    // Return distance
    const dist_x = px - closest_x;
    const dist_y = py - closest_y;
    return @sqrt(dist_x * dist_x + dist_y * dist_y);
}

fn shapeIntersectsBox(shape: Shape, box_min_x: f32, box_min_y: f32, box_max_x: f32, box_max_y: f32) bool {
    // Get shape bounds
    const shape_min_x = shape.x - shape.width;
    const shape_min_y = shape.y - shape.height;
    const shape_max_x = shape.x + shape.width;
    const shape_max_y = shape.y + shape.height;

    // Check AABB intersection
    return shape_max_x >= box_min_x and shape_min_x <= box_max_x and
        shape_max_y >= box_min_y and shape_min_y <= box_max_y;
}

// Get the bounding box of a path
fn getPathBounds(path_index: usize) struct { min_x: f32, min_y: f32, max_x: f32, max_y: f32, has_points: bool } {
    if (path_index >= path_count) {
        return .{ .min_x = 0, .min_y = 0, .max_x = 0, .max_y = 0, .has_points = false };
    }

    const path = paths[path_index];
    if (path.point_count == 0) {
        return .{ .min_x = 0, .min_y = 0, .max_x = 0, .max_y = 0, .has_points = false };
    }

    var min_x: f32 = std.math.inf(f32);
    var min_y: f32 = std.math.inf(f32);
    var max_x: f32 = -std.math.inf(f32);
    var max_y: f32 = -std.math.inf(f32);

    for (0..path.point_count) |i| {
        const pt = path.points[i];
        min_x = @min(min_x, pt[0]);
        min_y = @min(min_y, pt[1]);
        max_x = @max(max_x, pt[0]);
        max_y = @max(max_y, pt[1]);

        // Also include control points for bezier curves
        // control_points[i][0] = in-handle, control_points[i][1] = out-handle
        const ctrl_in = path.control_points[i][0];
        const ctrl_out = path.control_points[i][1];
        min_x = @min(min_x, @min(ctrl_in[0], ctrl_out[0]));
        min_y = @min(min_y, @min(ctrl_in[1], ctrl_out[1]));
        max_x = @max(max_x, @max(ctrl_in[0], ctrl_out[0]));
        max_y = @max(max_y, @max(ctrl_in[1], ctrl_out[1]));
    }

    return .{ .min_x = min_x, .min_y = min_y, .max_x = max_x, .max_y = max_y, .has_points = true };
}

// Get the bounding box of a group (computed from its children, recursively)
fn getGroupBounds(group_index: usize) struct { min_x: f32, min_y: f32, max_x: f32, max_y: f32, has_children: bool } {
    const group_id = @as(i32, @intCast(group_index));
    var min_x: f32 = std.math.inf(f32);
    var min_y: f32 = std.math.inf(f32);
    var max_x: f32 = -std.math.inf(f32);
    var max_y: f32 = -std.math.inf(f32);
    var has_children = false;

    for (0..shape_count) |i| {
        if (shapes[i].parent_id == group_id) {
            has_children = true;
            const child = shapes[i];

            // Handle nested groups recursively
            if (child.shape_type == 2) {
                const nested_bounds = getGroupBounds(i);
                if (nested_bounds.has_children) {
                    min_x = @min(min_x, nested_bounds.min_x);
                    min_y = @min(min_y, nested_bounds.min_y);
                    max_x = @max(max_x, nested_bounds.max_x);
                    max_y = @max(max_y, nested_bounds.max_y);
                }
            } else if (child.shape_type == 3 and child.path_index < path_count) {
                // Handle paths - they use path points for bounds
                const path_bounds = getPathBounds(child.path_index);
                if (path_bounds.has_points) {
                    min_x = @min(min_x, path_bounds.min_x);
                    min_y = @min(min_y, path_bounds.min_y);
                    max_x = @max(max_x, path_bounds.max_x);
                    max_y = @max(max_y, path_bounds.max_y);
                }
            } else {
                const child_min_x = child.x - child.width;
                const child_min_y = child.y - child.height;
                const child_max_x = child.x + child.width;
                const child_max_y = child.y + child.height;
                min_x = @min(min_x, child_min_x);
                min_y = @min(min_y, child_min_y);
                max_x = @max(max_x, child_max_x);
                max_y = @max(max_y, child_max_y);
            }
        }
    }

    return .{ .min_x = min_x, .min_y = min_y, .max_x = max_x, .max_y = max_y, .has_children = has_children };
}

// Recursively move all descendants of a group
fn moveGroupDescendants(group_index: usize, dx: f32, dy: f32) void {
    const group_id = @as(i32, @intCast(group_index));
    for (0..shape_count) |j| {
        if (shapes[j].parent_id == group_id) {
            shapes[j].x += dx;
            shapes[j].y += dy;

            // If child is a nested group, recursively move its descendants
            if (shapes[j].shape_type == 2) {
                moveGroupDescendants(j, dx, dy);
            }

            // If child is a path, also move its path points
            if (shapes[j].shape_type == 3 and shapes[j].path_index < path_count) {
                const child_path = &paths[shapes[j].path_index];
                for (0..child_path.point_count) |pt_idx| {
                    child_path.points[pt_idx][0] += dx;
                    child_path.points[pt_idx][1] += dy;
                    child_path.control_points[pt_idx][0][0] += dx;
                    child_path.control_points[pt_idx][0][1] += dy;
                    child_path.control_points[pt_idx][1][0] += dx;
                    child_path.control_points[pt_idx][1][1] += dy;
                }
            }

            // Mark tiles dirty
            markShapeTilesDirty(shapes[j].x, shapes[j].y, shapes[j].width, shapes[j].height);
        }
    }
}

// Check if a shape is a descendant of a group (at any nesting level)
fn isDescendantOf(shape_index: usize, ancestor_id: i32) bool {
    var current_parent = shapes[shape_index].parent_id;
    while (current_parent >= 0) {
        if (current_parent == ancestor_id) return true;
        const parent_idx = @as(usize, @intCast(current_parent));
        if (parent_idx >= shape_count) return false;
        current_parent = shapes[parent_idx].parent_id;
    }
    return false;
}

// Get the top-level parent group for a shape (or its own id if no parent)
fn getTopLevelParent(shape_index: usize) i32 {
    var current_idx = shape_index;
    while (shapes[current_idx].parent_id >= 0) {
        const parent_idx = @as(usize, @intCast(shapes[current_idx].parent_id));
        if (parent_idx >= shape_count) break;
        current_idx = parent_idx;
    }
    return @as(i32, @intCast(current_idx));
}

// Recursively mark tiles dirty for all descendants of a group
fn markGroupDescendantTilesDirty(group_index: usize) void {
    const group_id = @as(i32, @intCast(group_index));
    for (0..shape_count) |j| {
        if (shapes[j].parent_id == group_id) {
            markShapeTilesDirty(shapes[j].x, shapes[j].y, shapes[j].width, shapes[j].height);
            // Recursively mark nested group descendants
            if (shapes[j].shape_type == 2) {
                markGroupDescendantTilesDirty(j);
            }
        }
    }
}

// Check if a point is inside a group (checks against children bounds)
fn pointInGroup(px: f32, py: f32, group_index: usize) bool {
    const bounds = getGroupBounds(group_index);
    if (!bounds.has_children) return false;
    return px >= bounds.min_x and px <= bounds.max_x and py >= bounds.min_y and py <= bounds.max_y;
}

// Check if a group intersects a selection box
fn groupIntersectsBox(group_index: usize, box_min_x: f32, box_min_y: f32, box_max_x: f32, box_max_y: f32) bool {
    const bounds = getGroupBounds(group_index);
    if (!bounds.has_children) return false;
    return bounds.max_x >= box_min_x and bounds.min_x <= box_max_x and
        bounds.max_y >= box_min_y and bounds.min_y <= box_max_y;
}

// Check if cursor is on a resize handle of a shape
// Returns: 0=none, 1=left, 2=right, 3=top, 4=bottom, 5=top-left, 6=top-right, 7=bottom-left, 8=bottom-right
fn getResizeHandle(px: f32, py: f32, shape: Shape, current_zoom: f32) u8 {
    const handle_size = 0.02 / current_zoom; // Size of the handle hit area

    const min_x = shape.x - shape.width;
    const max_x = shape.x + shape.width;
    const min_y = shape.y - shape.height;
    const max_y = shape.y + shape.height;

    // Check corners first (they have priority)
    // Top-left
    if (@abs(px - min_x) < handle_size and @abs(py - max_y) < handle_size) return 5;
    // Top-right
    if (@abs(px - max_x) < handle_size and @abs(py - max_y) < handle_size) return 6;
    // Bottom-left
    if (@abs(px - min_x) < handle_size and @abs(py - min_y) < handle_size) return 7;
    // Bottom-right
    if (@abs(px - max_x) < handle_size and @abs(py - min_y) < handle_size) return 8;

    // Check edges (near the midpoints)
    // Left edge
    if (@abs(px - min_x) < handle_size and py > min_y and py < max_y) return 1;
    // Right edge
    if (@abs(px - max_x) < handle_size and py > min_y and py < max_y) return 2;
    // Top edge
    if (@abs(py - max_y) < handle_size and px > min_x and px < max_x) return 3;
    // Bottom edge
    if (@abs(py - min_y) < handle_size and px > min_x and px < max_x) return 4;

    return 0;
}

// Check if cursor is on the rotation handle of a shape
// Rotation handle is a circle above the top edge of the shape (in rotated space)
fn isOnRotationHandle(px: f32, py: f32, shape: Shape, current_zoom: f32) bool {
    const handle_size = 0.02 / current_zoom;
    const rotation_handle_offset = 0.06 / current_zoom; // Distance above top edge
    const hh = shape.height + 0.01 / current_zoom; // half-height with padding (matching overlay)

    // Rotation handle position in local space (before rotation)
    const local_handle_x: f32 = 0;
    const local_handle_y: f32 = hh + rotation_handle_offset;

    // Rotate handle position around shape center
    const cos_r = @cos(shape.rotation);
    const sin_r = @sin(shape.rotation);
    const handle_x = shape.x + local_handle_x * cos_r - local_handle_y * sin_r;
    const handle_y = shape.y + local_handle_x * sin_r + local_handle_y * cos_r;

    // Check if point is within handle radius
    const dx = px - handle_x;
    const dy = py - handle_y;
    return (dx * dx + dy * dy) < (handle_size * handle_size);
}

// ============================================================================
// Update logic
// ============================================================================

var frame_count: u32 = 0;

export fn init() void {
    log("Zig 2D Drawing initialized!", .{});
}

export fn update(time: f32) void {
    frame_count += 1;
    vertex_count = 0;

    // Update uniforms
    uniforms.time = time;
    uniforms.mouse_x = mouse_x;
    uniforms.mouse_y = mouse_y;
    uniforms.aspect_ratio = canvas_width / canvas_height;
    uniforms.zoom = zoom;
    uniforms.pan_x = pan_x;
    uniforms.pan_y = pan_y;

    // Aspect ratio for coordinate conversion
    const aspect = canvas_width / canvas_height;

    // Current mouse position in NDC (account for zoom/pan and aspect ratio)
    const current_x = (((mouse_x - 0.5) * 2.0 * aspect) - pan_x) / zoom;
    const current_y = ((0.5 - mouse_y) * 2.0 - pan_y) / zoom;

    // Detect mouse state changes
    const mouse_just_pressed = mouse_pressed and !prev_mouse_pressed;
    const mouse_just_released = !mouse_pressed and prev_mouse_pressed;

    // Handle Cmd+click for path point selection (works in select tool mode)
    if (current_tool == 2 and meta_key_pressed and mouse_just_pressed) {
        const hit_threshold = 0.015 / zoom;
        var found_point = false;

        // Look for a point to select on SELECTED path shapes only
        for (0..shape_count) |shape_idx| {
            const shape = shapes[shape_idx];
            if (!multi_selection[shape_idx] or !shape.visible or shape.shape_type != 3) continue;
            if (shape.path_index >= path_count) continue;

            const path = &paths[shape.path_index];
            for (0..path.point_count) |pt_idx| {
                const pt = path.points[pt_idx];
                const dx = current_x - pt[0];
                const dy = current_y - pt[1];
                const dist = @sqrt(dx * dx + dy * dy);

                if (dist < hit_threshold) {
                    pen_selected_path = @intCast(shape.path_index);
                    pen_selected_point = @intCast(pt_idx);
                    found_point = true;
                    markDirty();
                    log("Selected point {d} on path {d} for handle editing (select tool)", .{ pt_idx, shape.path_index });
                    break;
                }
            }
            if (found_point) break;
        }

        // If no point found, deselect
        if (!found_point) {
            pen_selected_point = -1;
            pen_selected_path = -1;
            markDirty();
        }
    }

    // Handle drag start (only when not zooming)
    if (mouse_just_pressed and !meta_key_pressed) {
        // First check if clicking on a bezier handle (works in select tool mode)
        if (current_tool == 2 and pen_selected_point >= 0 and pen_selected_path >= 0) {
            const hit_threshold = 0.015 / zoom;
            const path_idx: usize = @intCast(pen_selected_path);
            const pt_idx: usize = @intCast(pen_selected_point);
            if (path_idx < path_count) {
                const path = &paths[path_idx];
                if (pt_idx < path.point_count) {
                    const ctrl_out = path.control_points[pt_idx][1];
                    const ctrl_in = path.control_points[pt_idx][0];

                    // Check out handle
                    const dx_out = current_x - ctrl_out[0];
                    const dy_out = current_y - ctrl_out[1];
                    const dist_out = @sqrt(dx_out * dx_out + dy_out * dy_out);

                    // Check in handle
                    const dx_in = current_x - ctrl_in[0];
                    const dy_in = current_y - ctrl_in[1];
                    const dist_in = @sqrt(dx_in * dx_in + dy_in * dy_in);

                    if (dist_out < hit_threshold) {
                        pen_dragging_handle = true;
                        pen_handle_type = 0; // out handle
                        log("Started dragging out handle (select tool)", .{});
                    } else if (dist_in < hit_threshold) {
                        pen_dragging_handle = true;
                        pen_handle_type = 1; // in handle
                        log("Started dragging in handle (select tool)", .{});
                    }
                }
            }
        }

        if (current_tool == 2 and !pen_dragging_handle) {
            // Selection tool - first check for rotation handle on selected shapes
            var found_rotation_handle = false;
            var rotation_idx: usize = 0;
            for (0..shape_count) |i| {
                if (multi_selection[i] and shapes[i].visible and shapes[i].shape_type != 2 and shapes[i].shape_type != 3) {
                    if (isOnRotationHandle(current_x, current_y, shapes[i], zoom)) {
                        found_rotation_handle = true;
                        rotation_idx = i;
                        break;
                    }
                }
            }

            if (found_rotation_handle) {
                // Start rotating
                pushHistorySnapshot();
                is_rotating = true;
                rotate_shape_idx = rotation_idx;
                const rs = shapes[rotation_idx];
                // Calculate initial angle from shape center to cursor
                rotate_start_angle = std.math.atan2(current_y - rs.y, current_x - rs.x);
                rotate_shape_initial_rotation = rs.rotation;
            } else {
                // Check for resize handles on selected shapes
                var found_resize_handle: u8 = 0;
                var resize_idx: usize = 0;
                for (0..shape_count) |i| {
                    if (multi_selection[i] and shapes[i].visible and shapes[i].shape_type != 2) {
                        const handle = getResizeHandle(current_x, current_y, shapes[i], zoom);
                        if (handle != 0) {
                            found_resize_handle = handle;
                            resize_idx = i;
                            break;
                        }
                    }
                }

                if (found_resize_handle != 0) {
                    // Start resizing
                    pushHistorySnapshot(); // Save state before resizing
                    is_resizing = true;
                    resize_handle = found_resize_handle;
                    resize_shape_idx = resize_idx;

                    // Set anchor edges (the edges that stay fixed)
                    const rs = shapes[resize_idx];
                    const s_min_x = rs.x - rs.width;
                    const s_max_x = rs.x + rs.width;
                    const s_min_y = rs.y - rs.height;
                    const s_max_y = rs.y + rs.height;

                    // Horizontal anchor: opposite edge of the handle
                    switch (found_resize_handle) {
                        1, 5, 7 => {
                            resize_anchor_x = s_max_x;
                        }, // Left handles -> anchor right
                        2, 6, 8 => {
                            resize_anchor_x = s_min_x;
                        }, // Right handles -> anchor left
                        else => {
                            resize_anchor_x = rs.x;
                        }, // Top/bottom only -> anchor center
                    }
                    // Vertical anchor: opposite edge of the handle
                    switch (found_resize_handle) {
                        3, 5, 6 => {
                            resize_anchor_y = s_min_y;
                        }, // Top handles -> anchor bottom
                        4, 7, 8 => {
                            resize_anchor_y = s_max_y;
                        }, // Bottom handles -> anchor top
                        else => {
                            resize_anchor_y = rs.y;
                        }, // Left/right only -> anchor center
                    }
                } else {
                    // Check if clicking on a selected shape/group to move
                    var clicked_on_selected = false;
                    for (0..shape_count) |i| {
                        if (multi_selection[i] and shapes[i].visible) {
                            if (shapes[i].shape_type == 2) {
                                // Group - check against group bounds
                                if (pointInGroup(current_x, current_y, i)) {
                                    clicked_on_selected = true;
                                    break;
                                }
                            } else {
                                if (pointInShape(current_x, current_y, shapes[i])) {
                                    clicked_on_selected = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (clicked_on_selected) {
                        // Start moving selected shapes
                        pushHistorySnapshot(); // Save state before moving
                        is_moving_shapes = true;
                        move_start_x = current_x;
                        move_start_y = current_y;
                        prev_move_x = current_x;
                        prev_move_y = current_y;
                    } else {
                        // Check if clicking on any unselected shape - select it and start moving immediately
                        var clicked_shape: i32 = -1;
                        for (0..shape_count) |i| {
                            const shape = shapes[i];
                            if (!shape.visible) continue;

                            // Skip children of groups - they should be selected via their parent
                            if (shape.parent_id >= 0) continue;

                            if (shape.shape_type == 2) {
                                if (pointInGroup(current_x, current_y, i)) {
                                    clicked_shape = @as(i32, @intCast(i));
                                }
                            } else {
                                if (pointInShape(current_x, current_y, shape)) {
                                    clicked_shape = @as(i32, @intCast(i));
                                }
                            }
                        }

                        if (clicked_shape >= 0) {
                            // Select this shape (clear others unless shift held)
                            if (!shift_key_pressed) {
                                for (0..MAX_SHAPES) |i| {
                                    multi_selection[i] = false;
                                }
                            }
                            multi_selection[@as(usize, @intCast(clicked_shape))] = true;
                            selected_shape = clicked_shape;
                            // Clear bezier point selection when selecting a different shape
                            pen_selected_point = -1;
                            pen_selected_path = -1;
                            markDirty();

                            // Start moving immediately
                            pushHistorySnapshot(); // Save state before moving
                            is_moving_shapes = true;
                            move_start_x = current_x;
                            move_start_y = current_y;
                            prev_move_x = current_x;
                            prev_move_y = current_y;
                        } else {
                            // Start selection box (clicked on empty space)
                            selection_box_active = true;
                            selection_box_start_x = current_x;
                            selection_box_start_y = current_y;
                        }
                    }
                }
            } // End of else block for rotation handle
        } else if (current_tool != 2 and !pen_dragging_handle) {
            // Not select tool and not dragging a bezier handle - start shape creation drag
            is_dragging = true;
            drag_start_x = current_x;
            drag_start_y = current_y;
        }
    }

    // Handle bezier handle dragging in select tool mode
    if (current_tool == 2 and pen_dragging_handle and mouse_pressed and pen_selected_point >= 0 and pen_selected_path >= 0) {
        const path_idx: usize = @intCast(pen_selected_path);
        const pt_idx: usize = @intCast(pen_selected_point);
        if (path_idx < path_count) {
            const path = &paths[path_idx];
            if (pt_idx < path.point_count) {
                const anchor = path.points[pt_idx];

                if (pen_handle_type == 0) {
                    // Dragging out handle - affects segment FROM this point
                    path.control_points[pt_idx][1] = .{ current_x, current_y };
                    // Mirror to in handle if symmetric
                    if (pen_symmetric_handles) {
                        const dx = current_x - anchor[0];
                        const dy = current_y - anchor[1];
                        path.control_points[pt_idx][0] = .{ anchor[0] - dx, anchor[1] - dy };
                        // Also mark the segment TO this point as bezier if it exists
                        if (pt_idx > 0) {
                            path.segment_types[pt_idx - 1] = .cubic_bezier;
                            // ALWAYS initialize the OTHER endpoint (pt_idx-1) out-handle
                            const prev_pt = path.points[pt_idx - 1];
                            const dir_x = anchor[0] - prev_pt[0];
                            const dir_y = anchor[1] - prev_pt[1];
                            const dir_len = @sqrt(dir_x * dir_x + dir_y * dir_y);
                            if (dir_len > 0.001) {
                                const handle_len = dir_len / 3.0;
                                path.control_points[pt_idx - 1][1] = .{
                                    prev_pt[0] + (dir_x / dir_len) * handle_len,
                                    prev_pt[1] + (dir_y / dir_len) * handle_len,
                                };
                            }
                        }
                    }
                    // Mark segment from this point as bezier
                    if (pt_idx < path.point_count - 1) {
                        path.segment_types[pt_idx] = .cubic_bezier;
                        // ALWAYS initialize the other endpoint's in-handle for this segment
                        const next_pt = path.points[pt_idx + 1];
                        // Initialize next point's in-handle to point toward this anchor
                        const dir_x = anchor[0] - next_pt[0];
                        const dir_y = anchor[1] - next_pt[1];
                        const dir_len = @sqrt(dir_x * dir_x + dir_y * dir_y);
                        if (dir_len > 0.001) {
                            // Use 1/3 of segment length as handle length (standard bezier convention)
                            const handle_len = dir_len / 3.0;
                            path.control_points[pt_idx + 1][0] = .{
                                next_pt[0] + (dir_x / dir_len) * handle_len,
                                next_pt[1] + (dir_y / dir_len) * handle_len,
                            };
                        }
                    } else if (path.closed) {
                        path.segment_types[pt_idx] = .cubic_bezier;
                    }
                } else {
                    // Dragging in handle - affects segment TO this point
                    path.control_points[pt_idx][0] = .{ current_x, current_y };
                    // Mirror to out handle if symmetric
                    if (pen_symmetric_handles) {
                        const dx = current_x - anchor[0];
                        const dy = current_y - anchor[1];
                        path.control_points[pt_idx][1] = .{ anchor[0] - dx, anchor[1] - dy };
                        // Also mark the segment FROM this point as bezier if it exists
                        if (pt_idx < path.point_count - 1) {
                            path.segment_types[pt_idx] = .cubic_bezier;
                            // ALWAYS initialize the OTHER endpoint (pt_idx+1) in-handle
                            const next_pt = path.points[pt_idx + 1];
                            const dir_x = anchor[0] - next_pt[0];
                            const dir_y = anchor[1] - next_pt[1];
                            const dir_len = @sqrt(dir_x * dir_x + dir_y * dir_y);
                            if (dir_len > 0.001) {
                                const handle_len = dir_len / 3.0;
                                path.control_points[pt_idx + 1][0] = .{
                                    next_pt[0] + (dir_x / dir_len) * handle_len,
                                    next_pt[1] + (dir_y / dir_len) * handle_len,
                                };
                            }
                        }
                    }
                    // Mark segment TO this point as bezier (previous segment)
                    if (pt_idx > 0) {
                        path.segment_types[pt_idx - 1] = .cubic_bezier;
                        // ALWAYS initialize the other endpoint's out-handle
                        const prev_pt = path.points[pt_idx - 1];
                        const dir_x = anchor[0] - prev_pt[0];
                        const dir_y = anchor[1] - prev_pt[1];
                        const dir_len = @sqrt(dir_x * dir_x + dir_y * dir_y);
                        if (dir_len > 0.0001) {
                            const handle_len = dir_len / 3.0;
                            path.control_points[pt_idx - 1][1] = .{
                                prev_pt[0] + (dir_x / dir_len) * handle_len,
                                prev_pt[1] + (dir_y / dir_len) * handle_len,
                            };
                        }
                    } else if (path.closed and path.point_count > 0) {
                        path.segment_types[path.point_count - 1] = .cubic_bezier;
                    }
                }
                markDirty();
            }
        }
    }

    // Release bezier handle drag (works in all tool modes)
    if (!mouse_pressed and pen_dragging_handle) {
        pen_dragging_handle = false;
    }

    // Handle moving shapes while dragging
    if (is_moving_shapes and mouse_pressed and !meta_key_pressed) {
        const dx = current_x - prev_move_x;
        const dy = current_y - prev_move_y;
        if (dx != 0 or dy != 0) {
            // Mark tiles at OLD positions dirty (before move)
            for (0..shape_count) |i| {
                if (multi_selection[i]) {
                    markShapeTilesDirty(shapes[i].x, shapes[i].y, shapes[i].width, shapes[i].height);
                    // Also recursively mark descendants if this is a group
                    if (shapes[i].shape_type == 2) {
                        markGroupDescendantTilesDirty(i);
                    }
                }
            }

            // Move all selected shapes (and their descendants if groups, and path points if paths)
            for (0..shape_count) |i| {
                if (multi_selection[i]) {
                    shapes[i].x += dx;
                    shapes[i].y += dy;

                    // If this is a group, recursively move all descendants
                    if (shapes[i].shape_type == 2) {
                        moveGroupDescendants(i, dx, dy);
                    }

                    // If this is a path, also move all its points
                    if (shapes[i].shape_type == 3 and shapes[i].path_index < path_count) {
                        const path = &paths[shapes[i].path_index];
                        for (0..path.point_count) |pt_idx| {
                            path.points[pt_idx][0] += dx;
                            path.points[pt_idx][1] += dy;
                            // Also move control points
                            path.control_points[pt_idx][0][0] += dx;
                            path.control_points[pt_idx][0][1] += dy;
                            path.control_points[pt_idx][1][0] += dx;
                            path.control_points[pt_idx][1][1] += dy;
                        }
                    }

                    // Mark tiles at NEW positions dirty (after move)
                    markShapeTilesDirty(shapes[i].x, shapes[i].y, shapes[i].width, shapes[i].height);
                }
            }
            prev_move_x = current_x;
            prev_move_y = current_y;
            // Use light dirty marking (tiles already marked above)
            shapes_dirty = true;
            scene_cache_valid = false;
        }
    }

    // Handle resizing shape while dragging
    if (is_resizing and mouse_pressed and !meta_key_pressed) {
        const shape = &shapes[resize_shape_idx];

        // Mark tiles at OLD bounds dirty (before resize)
        markShapeTilesDirty(shape.x, shape.y, shape.width, shape.height);

        // Calculate new bounds using anchors
        // The anchor is the fixed edge, current position is the moving edge
        switch (resize_handle) {
            1, 2 => { // Left or Right (horizontal only)
                shape.x = (resize_anchor_x + current_x) / 2.0;
                shape.width = @abs(current_x - resize_anchor_x) / 2.0;
            },
            3, 4 => { // Top or Bottom (vertical only)
                shape.y = (resize_anchor_y + current_y) / 2.0;
                shape.height = @abs(current_y - resize_anchor_y) / 2.0;
            },
            5, 6, 7, 8 => { // Corner handles (both dimensions)
                shape.x = (resize_anchor_x + current_x) / 2.0;
                shape.width = @abs(current_x - resize_anchor_x) / 2.0;
                shape.y = (resize_anchor_y + current_y) / 2.0;
                shape.height = @abs(current_y - resize_anchor_y) / 2.0;
            },
            else => {},
        }

        // Mark tiles at NEW bounds dirty (after resize)
        markShapeTilesDirty(shape.x, shape.y, shape.width, shape.height);

        // Use light dirty marking (tiles already marked above)
        shapes_dirty = true;
        scene_cache_valid = false;
    }

    // Stop moving shapes on release
    if (mouse_just_released and is_moving_shapes) {
        is_moving_shapes = false;
    }

    // Stop resizing on release
    if (mouse_just_released and is_resizing) {
        is_resizing = false;
        resize_handle = 0;
    }

    // Handle rotating shape while dragging
    if (is_rotating and mouse_pressed and !meta_key_pressed) {
        const shape = &shapes[rotate_shape_idx];

        // Mark tiles at shape bounds dirty
        markShapeTilesDirty(shape.x, shape.y, shape.width, shape.height);

        // Calculate current angle from shape center to cursor
        const current_angle = std.math.atan2(current_y - shape.y, current_x - shape.x);
        const angle_delta = current_angle - rotate_start_angle;
        shape.rotation = rotate_shape_initial_rotation + angle_delta;

        // Mark dirty
        shapes_dirty = true;
        scene_cache_valid = false;
    }

    // Stop rotating on release
    if (mouse_just_released and is_rotating) {
        is_rotating = false;
    }

    // Cancel drag if meta key was pressed during drag
    if (meta_key_pressed and is_dragging) {
        is_dragging = false;
    }
    if (meta_key_pressed and selection_box_active) {
        selection_box_active = false;
    }
    if (meta_key_pressed and is_moving_shapes) {
        is_moving_shapes = false;
    }
    if (meta_key_pressed and is_resizing) {
        is_resizing = false;
        resize_handle = 0;
    }
    if (meta_key_pressed and is_rotating) {
        is_rotating = false;
    }

    // Handle selection tool release
    if (mouse_just_released and selection_box_active and !meta_key_pressed) {
        selection_box_active = false;

        const sel_min_x = @min(selection_box_start_x, current_x);
        const sel_max_x = @max(selection_box_start_x, current_x);
        const sel_min_y = @min(selection_box_start_y, current_y);
        const sel_max_y = @max(selection_box_start_y, current_y);

        const sel_width = sel_max_x - sel_min_x;
        const sel_height = sel_max_y - sel_min_y;

        // Clear previous selection only if shift is not held
        if (!shift_key_pressed) {
            for (0..MAX_SHAPES) |i| {
                multi_selection[i] = false;
            }
            selected_shape = -1;
            markDirty();
        }

        // Check if it's a click (small drag) or box selection
        if (sel_width < 0.02 and sel_height < 0.02) {
            // Click selection - find topmost shape/group at click point
            var top_hit: i32 = -1;
            for (0..shape_count) |i| {
                const shape = shapes[i];
                if (!shape.visible) continue;

                // Skip children of groups - they should be selected via their parent
                if (shape.parent_id >= 0) continue;

                if (shape.shape_type == 2) {
                    // Group - check against group bounds
                    if (pointInGroup(current_x, current_y, i)) {
                        top_hit = @as(i32, @intCast(i));
                    }
                } else {
                    if (pointInShape(current_x, current_y, shape)) {
                        top_hit = @as(i32, @intCast(i));
                    }
                }
            }
            if (top_hit >= 0) {
                const hit_idx = @as(usize, @intCast(top_hit));
                if (shift_key_pressed) {
                    // Toggle selection with shift
                    multi_selection[hit_idx] = !multi_selection[hit_idx];
                } else {
                    multi_selection[hit_idx] = true;
                }
                selected_shape = top_hit;
                markDirty();
            }
        } else {
            // Box selection - select all shapes/groups that intersect the box
            var first_selected: i32 = -1;
            for (0..shape_count) |i| {
                const shape = shapes[i];
                if (!shape.visible) continue;

                // Skip children of groups - they should be selected via their parent
                if (shape.parent_id >= 0) continue;

                var intersects = false;
                if (shape.shape_type == 2) {
                    // Group - check against group bounds
                    intersects = groupIntersectsBox(i, sel_min_x, sel_min_y, sel_max_x, sel_max_y);
                } else {
                    intersects = shapeIntersectsBox(shape, sel_min_x, sel_min_y, sel_max_x, sel_max_y);
                }
                if (intersects) {
                    multi_selection[i] = true;
                    markDirty();
                    if (first_selected == -1) {
                        first_selected = @as(i32, @intCast(i));
                    }
                }
            }
            if (first_selected >= 0) {
                selected_shape = first_selected;
            }
        }
    }

    // Handle drag end - create shape (only if not meta key and using shape tools)
    if (mouse_just_released and is_dragging and !meta_key_pressed and current_tool != 2 and current_tool != 3) {
        is_dragging = false;

        // Calculate shape bounds
        const min_x = @min(drag_start_x, current_x);
        const max_x = @max(drag_start_x, current_x);
        const min_y = @min(drag_start_y, current_y);
        const max_y = @max(drag_start_y, current_y);

        const width = max_x - min_x;
        const height = max_y - min_y;

        // Only create shape if it has some size (not just a click)
        if (width > 0.01 or height > 0.01) {
            if (shape_count < MAX_SHAPES) {
                pushHistorySnapshot(); // Save state before creating shape
                const new_shape_index = shape_count;
                shapes[shape_count] = .{
                    .shape_type = current_tool,
                    .x = (min_x + max_x) / 2.0, // center
                    .y = (min_y + max_y) / 2.0,
                    .width = width / 2.0, // half-width (radius)
                    .height = height / 2.0,
                    .fill_enabled = fill_enabled,
                    .fill_color = fill_color,
                    .stroke_enabled = stroke_enabled,
                    .stroke_color = stroke_color,
                    .stroke_width = stroke_width, // Store current stroke width
                    .parent_id = -1,
                    .visible = true,
                    .corner_radius = if (current_tool == 1) corner_radius else 0.0, // Only for rectangles
                    .path_index = 0, // Not used for rect/circle
                };
                shape_count += 1;

                // Select the new shape and switch to select tool
                clearSelection();
                multi_selection[new_shape_index] = true;
                selected_shape = @intCast(new_shape_index);
                current_tool = 2; // Switch to select tool

                markDirty();
                log("Shape created! count={d}, zoom={d:.2}", .{ shape_count, zoom });
            }
        }
    }

    // Handle pen tool - click to add points, Cmd+click to edit handles
    if (current_tool == 3) {
        const hit_threshold = 0.015 / zoom; // Threshold for hitting points/handles

        // Cmd+click: Select point for handle editing (only on selected paths)
        if (meta_key_pressed and mouse_just_pressed) {
            // Look for a point to select on SELECTED path shapes only
            var found_point = false;
            for (0..shape_count) |shape_idx| {
                const shape = shapes[shape_idx];
                if (!multi_selection[shape_idx] or !shape.visible or shape.shape_type != 3) continue;
                if (shape.path_index >= path_count) continue;

                const path = &paths[shape.path_index];
                for (0..path.point_count) |pt_idx| {
                    const pt = path.points[pt_idx];
                    const dx = current_x - pt[0];
                    const dy = current_y - pt[1];
                    const dist = @sqrt(dx * dx + dy * dy);

                    if (dist < hit_threshold) {
                        pen_selected_path = @intCast(shape.path_index);
                        pen_selected_point = @intCast(pt_idx);
                        found_point = true;
                        markDirty();
                        log("Selected point {d} on path {d} for handle editing", .{ pt_idx, shape.path_index });
                        break;
                    }
                }
                if (found_point) break;
            }

            // If no point found, deselect
            if (!found_point) {
                pen_selected_point = -1;
                pen_selected_path = -1;
                markDirty();
            }
        }

        // Check if clicking on a handle of the selected point
        if (!meta_key_pressed and mouse_just_pressed and pen_selected_point >= 0 and pen_selected_path >= 0) {
            const path_idx: usize = @intCast(pen_selected_path);
            const pt_idx: usize = @intCast(pen_selected_point);
            if (path_idx < path_count) {
                const path = &paths[path_idx];
                if (pt_idx < path.point_count) {
                    const ctrl_out = path.control_points[pt_idx][1];
                    const ctrl_in = path.control_points[pt_idx][0];

                    // Check out handle
                    const dx_out = current_x - ctrl_out[0];
                    const dy_out = current_y - ctrl_out[1];
                    const dist_out = @sqrt(dx_out * dx_out + dy_out * dy_out);

                    // Check in handle
                    const dx_in = current_x - ctrl_in[0];
                    const dy_in = current_y - ctrl_in[1];
                    const dist_in = @sqrt(dx_in * dx_in + dy_in * dy_in);

                    if (dist_out < hit_threshold) {
                        pen_dragging_handle = true;
                        pen_handle_type = 0; // out handle
                        log("Started dragging out handle", .{});
                    } else if (dist_in < hit_threshold) {
                        pen_dragging_handle = true;
                        pen_handle_type = 1; // in handle
                        log("Started dragging in handle", .{});
                    }
                }
            }
        }

        // Dragging a handle - update control point
        if (pen_dragging_handle and mouse_pressed and pen_selected_point >= 0 and pen_selected_path >= 0) {
            const path_idx: usize = @intCast(pen_selected_path);
            const pt_idx: usize = @intCast(pen_selected_point);
            if (path_idx < path_count) {
                const path = &paths[path_idx];
                if (pt_idx < path.point_count) {
                    const anchor = path.points[pt_idx];

                    if (pen_handle_type == 0) {
                        // Dragging out handle
                        path.control_points[pt_idx][1] = .{ current_x, current_y };
                        // Mirror to in handle if symmetric
                        if (pen_symmetric_handles) {
                            const dx = current_x - anchor[0];
                            const dy = current_y - anchor[1];
                            path.control_points[pt_idx][0] = .{ anchor[0] - dx, anchor[1] - dy };
                            // Also mark the segment TO this point as bezier if it exists
                            if (pt_idx > 0) {
                                path.segment_types[pt_idx - 1] = .cubic_bezier;
                                // ALWAYS initialize the OTHER endpoint (pt_idx-1) out-handle
                                const prev_pt = path.points[pt_idx - 1];
                                const dir_x = anchor[0] - prev_pt[0];
                                const dir_y = anchor[1] - prev_pt[1];
                                const dir_len = @sqrt(dir_x * dir_x + dir_y * dir_y);
                                if (dir_len > 0.001) {
                                    const handle_len = dir_len / 3.0;
                                    path.control_points[pt_idx - 1][1] = .{
                                        prev_pt[0] + (dir_x / dir_len) * handle_len,
                                        prev_pt[1] + (dir_y / dir_len) * handle_len,
                                    };
                                }
                            }
                        }
                        // Mark segment from this point as bezier
                        if (pt_idx < path.point_count - 1) {
                            path.segment_types[pt_idx] = .cubic_bezier;
                            // ALWAYS initialize the other endpoint's in-handle for this segment
                            const next_pt = path.points[pt_idx + 1];
                            const dir_x = anchor[0] - next_pt[0];
                            const dir_y = anchor[1] - next_pt[1];
                            const dir_len = @sqrt(dir_x * dir_x + dir_y * dir_y);
                            if (dir_len > 0.001) {
                                const handle_len = dir_len / 3.0;
                                path.control_points[pt_idx + 1][0] = .{
                                    next_pt[0] + (dir_x / dir_len) * handle_len,
                                    next_pt[1] + (dir_y / dir_len) * handle_len,
                                };
                            }
                        } else if (path.closed) {
                            path.segment_types[pt_idx] = .cubic_bezier;
                        }
                    } else {
                        // Dragging in handle
                        path.control_points[pt_idx][0] = .{ current_x, current_y };
                        // Mirror to out handle if symmetric
                        if (pen_symmetric_handles) {
                            const dx = current_x - anchor[0];
                            const dy = current_y - anchor[1];
                            path.control_points[pt_idx][1] = .{ anchor[0] - dx, anchor[1] - dy };
                            // Also mark the segment FROM this point as bezier if it exists
                            if (pt_idx < path.point_count - 1) {
                                path.segment_types[pt_idx] = .cubic_bezier;
                                // ALWAYS initialize the OTHER endpoint (pt_idx+1) in-handle
                                const next_pt = path.points[pt_idx + 1];
                                const dir_x = anchor[0] - next_pt[0];
                                const dir_y = anchor[1] - next_pt[1];
                                const dir_len = @sqrt(dir_x * dir_x + dir_y * dir_y);
                                if (dir_len > 0.001) {
                                    const handle_len = dir_len / 3.0;
                                    path.control_points[pt_idx + 1][0] = .{
                                        next_pt[0] + (dir_x / dir_len) * handle_len,
                                        next_pt[1] + (dir_y / dir_len) * handle_len,
                                    };
                                }
                            }
                        }
                        // Mark segment TO this point as bezier (previous segment)
                        if (pt_idx > 0) {
                            path.segment_types[pt_idx - 1] = .cubic_bezier;
                            // ALWAYS initialize the other endpoint's out-handle
                            const prev_pt = path.points[pt_idx - 1];
                            // Initialize previous point's out-handle to point toward this anchor
                            const dir_x = anchor[0] - prev_pt[0];
                            const dir_y = anchor[1] - prev_pt[1];
                            const dir_len = @sqrt(dir_x * dir_x + dir_y * dir_y);
                            if (dir_len > 0.0001) {
                                // Use 1/3 of segment length as handle length (standard bezier convention)
                                const handle_len = dir_len / 3.0;
                                path.control_points[pt_idx - 1][1] = .{
                                    prev_pt[0] + (dir_x / dir_len) * handle_len,
                                    prev_pt[1] + (dir_y / dir_len) * handle_len,
                                };
                            }
                        } else if (path.closed and path.point_count > 0) {
                            // If closed path, the "previous" segment is from last point
                            path.segment_types[path.point_count - 1] = .cubic_bezier;
                        }
                    }
                    markDirty();
                }
            }
        }

        // Release handle drag
        if (!mouse_pressed and pen_dragging_handle) {
            pen_dragging_handle = false;
        }

        // Normal click (no Cmd): Add points to path
        if (!meta_key_pressed and mouse_just_pressed and !pen_dragging_handle) {
            if (!pen_drawing) {
                // Start a new path
                if (path_count < MAX_PATHS and shape_count < MAX_SHAPES) {
                    pushHistorySnapshot();
                    pen_drawing = true;
                    pen_path_index = path_count;
                    pen_selected_point = -1; // Clear handle selection when starting new path
                    pen_selected_path = -1;

                    // Reset path to clean state before initializing
                    paths[path_count] = .{};

                    // Initialize new path with first point
                    paths[path_count].point_count = 1;
                    paths[path_count].points[0] = .{ current_x, current_y };
                    // Initialize control points for first point (straight line by default)
                    paths[path_count].control_points[0][0] = .{ current_x, current_y }; // in-handle
                    paths[path_count].control_points[0][1] = .{ current_x, current_y }; // out-handle
                    paths[path_count].segment_types[0] = .line;
                    paths[path_count].closed = false;

                    // Create shape for this path
                    shapes[shape_count] = .{
                        .shape_type = 3, // path
                        .x = current_x,
                        .y = current_y,
                        .width = 0,
                        .height = 0,
                        .fill_enabled = false, // Paths default to stroke only
                        .fill_color = fill_color,
                        .stroke_enabled = true,
                        .stroke_color = stroke_color,
                        .stroke_width = stroke_width, // Store current stroke width
                        .parent_id = -1,
                        .visible = true,
                        .corner_radius = 0,
                        .path_index = @as(u32, @intCast(path_count)),
                    };
                    shape_count += 1;
                    path_count += 1;
                    markDirty();
                    log("Started new path, count={d}", .{path_count});
                }
            } else {
                // Add point to current path
                const path = &paths[pen_path_index];
                if (path.point_count < MAX_PATH_POINTS) {
                    // Check if clicking near first point to close path
                    const first_pt = path.points[0];
                    const dist_to_first = @sqrt((current_x - first_pt[0]) * (current_x - first_pt[0]) +
                        (current_y - first_pt[1]) * (current_y - first_pt[1]));
                    const close_threshold = 0.02 / zoom; // Adjust based on zoom

                    if (path.point_count >= 2 and dist_to_first < close_threshold) {
                        // Close the path
                        path.closed = true;
                        pen_drawing = false;
                        markDirty();
                        log("Path closed with {d} points", .{path.point_count});
                    } else {
                        // Add new point (straight line segment)
                        const idx = path.point_count;
                        path.points[idx] = .{ current_x, current_y };
                        // Initialize control points to be at the anchor point (straight line by default)
                        path.control_points[idx][0] = .{ current_x, current_y }; // in-handle
                        path.control_points[idx][1] = .{ current_x, current_y }; // out-handle
                        path.segment_types[idx] = .line;
                        path.point_count += 1;

                        markDirty();
                        log("Added point {d} to path", .{path.point_count});
                    }
                }
            }
        }
    }

    // Cancel pen drawing on Escape (handled via finishPenPath export)
    if (pen_drawing and is_dragging) {
        is_dragging = false; // Prevent drag behavior during pen tool
    }

    // Update previous state at end of frame
    prev_mouse_pressed = mouse_pressed;

    // ========================================================================
    // SCENE GEOMETRY - Draw all stored shapes (with caching)
    // Zoom changes affect stroke widths and segment counts, so invalidate cache
    // Pan changes are handled in shader, so cached vertices remain valid
    // ========================================================================

    // Invalidate cache if zoom changed
    const zoom_changed = zoom != prev_zoom;
    if (zoom_changed) {
        prev_zoom = zoom;
        scene_cache_valid = false;
    }

    const scene_start_vertex = vertex_count;

    // Check if we can use cached scene vertices
    if (scene_cache_valid and cached_scene_vertex_count > 0) {
        // Copy cached scene vertices to main buffer
        for (0..cached_scene_vertex_count) |i| {
            vertices[vertex_count + i] = cached_scene_vertices[i];
        }
        vertex_count += cached_scene_vertex_count;
        scene_was_regenerated = false;
    } else {
        // Regenerate scene geometry with shape batching
        // Batching groups similar shapes together for better GPU cache utilization
        // Order: rect fills -> rect strokes -> ellipse fills -> ellipse strokes
        const gen_start = vertex_count;

        // Pass 1: Rectangle fills
        rect_fill_batch.start_vertex = vertex_count;
        for (0..shape_count) |i| {
            const shape = shapes[i];
            if (!shape.visible or shape.shape_type != 1 or !shape.fill_enabled) continue;
            const bounds_min_x = shape.x - shape.width;
            const bounds_min_y = shape.y - shape.height;
            const bounds_max_x = shape.x + shape.width;
            const bounds_max_y = shape.y + shape.height;
            if (!isBoundsVisible(bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y)) continue;
            addRectangle(bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y, shape.fill_color[0], shape.fill_color[1], shape.fill_color[2], 1.0);
        }
        rect_fill_batch.vertex_count = vertex_count - rect_fill_batch.start_vertex;

        // Pass 2: Ellipse fills
        ellipse_fill_batch.start_vertex = vertex_count;
        for (0..shape_count) |i| {
            const shape = shapes[i];
            if (!shape.visible or shape.shape_type != 0 or !shape.fill_enabled) continue;
            const bounds_min_x = shape.x - shape.width;
            const bounds_min_y = shape.y - shape.height;
            const bounds_max_x = shape.x + shape.width;
            const bounds_max_y = shape.y + shape.height;
            if (!isBoundsVisible(bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y)) continue;
            addEllipseFill(shape.x, shape.y, shape.width, shape.height, shape.fill_color[0], shape.fill_color[1], shape.fill_color[2], 1.0);
        }
        ellipse_fill_batch.vertex_count = vertex_count - ellipse_fill_batch.start_vertex;

        // Pass 3: Rectangle strokes (on top of fills)
        rect_stroke_batch.start_vertex = vertex_count;
        for (0..shape_count) |i| {
            const shape = shapes[i];
            if (!shape.visible or shape.shape_type != 1 or !shape.stroke_enabled) continue;
            const bounds_min_x = shape.x - shape.width;
            const bounds_min_y = shape.y - shape.height;
            const bounds_max_x = shape.x + shape.width;
            const bounds_max_y = shape.y + shape.height;
            if (!isBoundsVisible(bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y)) continue;
            addRectangleStroke(bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y, shape.stroke_width, shape.stroke_color[0], shape.stroke_color[1], shape.stroke_color[2], 1.0);
        }
        rect_stroke_batch.vertex_count = vertex_count - rect_stroke_batch.start_vertex;

        // Pass 4: Ellipse strokes (on top of fills)
        ellipse_stroke_batch.start_vertex = vertex_count;
        for (0..shape_count) |i| {
            const shape = shapes[i];
            if (!shape.visible or shape.shape_type != 0 or !shape.stroke_enabled) continue;
            const bounds_min_x = shape.x - shape.width;
            const bounds_min_y = shape.y - shape.height;
            const bounds_max_x = shape.x + shape.width;
            const bounds_max_y = shape.y + shape.height;
            if (!isBoundsVisible(bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y)) continue;
            addEllipseStroke(shape.x, shape.y, shape.width, shape.height, shape.stroke_width, shape.stroke_color[0], shape.stroke_color[1], shape.stroke_color[2], 1.0);
        }
        ellipse_stroke_batch.vertex_count = vertex_count - ellipse_stroke_batch.start_vertex;

        // Note: Paths are now rendered via GPU compute shader (buildPathDescriptors + runPathComputeGeometry)
        // No CPU path rendering needed here

        // Cache the generated scene vertices
        cached_scene_vertex_count = vertex_count - gen_start;
        for (0..cached_scene_vertex_count) |i| {
            cached_scene_vertices[i] = vertices[gen_start + i];
        }
        scene_cache_valid = true;
        scene_was_regenerated = true;
    }

    // Track scene vertex count (shapes only, no UI overlays)
    scene_vertex_count = vertex_count - scene_start_vertex;

    // ========================================================================
    // UI OVERLAY GEOMETRY - Selection boxes, handles, previews
    // ========================================================================
    const ui_start_vertex = vertex_count;

    // Draw selection highlights and resize handles for selected shapes (not paths or groups)
    for (0..shape_count) |i| {
        const shape = shapes[i];
        // Skip groups (type 2) and paths (type 3) - they have special handling
        if (!shape.visible or shape.shape_type == 2 or shape.shape_type == 3) continue;
        if (!multi_selection[i]) continue;

        // Get rotation sin/cos for transforming points
        const cos_r = @cos(shape.rotation);
        const sin_r = @sin(shape.rotation);
        const cx = shape.x;
        const cy = shape.y;

        // Helper to rotate a point around shape center
        const rotatePoint = struct {
            fn f(px: f32, py: f32, center_x: f32, center_y: f32, cos_rot: f32, sin_rot: f32) [2]f32 {
                const dx = px - center_x;
                const dy = py - center_y;
                return .{
                    center_x + dx * cos_rot - dy * sin_rot,
                    center_y + dx * sin_rot + dy * cos_rot,
                };
            }
        }.f;

        // Define the 8 control points in local space (before rotation)
        const handle_size = 0.012 / zoom;
        const hw = shape.width + 0.01 / zoom; // half-width with padding
        const hh = shape.height + 0.01 / zoom; // half-height with padding

        // Corner positions (local space relative to center)
        const tl = rotatePoint(cx - hw, cy + hh, cx, cy, cos_r, sin_r); // top-left
        const tr = rotatePoint(cx + hw, cy + hh, cx, cy, cos_r, sin_r); // top-right
        const bl = rotatePoint(cx - hw, cy - hh, cx, cy, cos_r, sin_r); // bottom-left
        const br = rotatePoint(cx + hw, cy - hh, cx, cy, cos_r, sin_r); // bottom-right

        // Edge midpoint positions
        const ml = rotatePoint(cx - hw, cy, cx, cy, cos_r, sin_r); // mid-left
        const mr = rotatePoint(cx + hw, cy, cx, cy, cos_r, sin_r); // mid-right
        const mt = rotatePoint(cx, cy + hh, cx, cy, cos_r, sin_r); // mid-top
        const mb = rotatePoint(cx, cy - hh, cx, cy, cos_r, sin_r); // mid-bottom

        // Draw selection highlight (rotated rectangle as 4 lines)
        const stroke_w = 0.005 / zoom;
        addLineSegment(tl[0], tl[1], tr[0], tr[1], stroke_w, 0.0, 0.478, 1.0, 1.0); // top
        addLineSegment(tr[0], tr[1], br[0], br[1], stroke_w, 0.0, 0.478, 1.0, 1.0); // right
        addLineSegment(br[0], br[1], bl[0], bl[1], stroke_w, 0.0, 0.478, 1.0, 1.0); // bottom
        addLineSegment(bl[0], bl[1], tl[0], tl[1], stroke_w, 0.0, 0.478, 1.0, 1.0); // left

        // Draw corner handles (axis-aligned squares at rotated positions)
        const handle_stroke = 0.003 / zoom;
        // Top-left
        addRectangle(tl[0] - handle_size, tl[1] - handle_size, tl[0] + handle_size, tl[1] + handle_size, 1.0, 1.0, 1.0, 1.0);
        addRectangleStroke(tl[0] - handle_size, tl[1] - handle_size, tl[0] + handle_size, tl[1] + handle_size, handle_stroke, 0.0, 0.478, 1.0, 1.0);
        // Top-right
        addRectangle(tr[0] - handle_size, tr[1] - handle_size, tr[0] + handle_size, tr[1] + handle_size, 1.0, 1.0, 1.0, 1.0);
        addRectangleStroke(tr[0] - handle_size, tr[1] - handle_size, tr[0] + handle_size, tr[1] + handle_size, handle_stroke, 0.0, 0.478, 1.0, 1.0);
        // Bottom-left
        addRectangle(bl[0] - handle_size, bl[1] - handle_size, bl[0] + handle_size, bl[1] + handle_size, 1.0, 1.0, 1.0, 1.0);
        addRectangleStroke(bl[0] - handle_size, bl[1] - handle_size, bl[0] + handle_size, bl[1] + handle_size, handle_stroke, 0.0, 0.478, 1.0, 1.0);
        // Bottom-right
        addRectangle(br[0] - handle_size, br[1] - handle_size, br[0] + handle_size, br[1] + handle_size, 1.0, 1.0, 1.0, 1.0);
        addRectangleStroke(br[0] - handle_size, br[1] - handle_size, br[0] + handle_size, br[1] + handle_size, handle_stroke, 0.0, 0.478, 1.0, 1.0);

        // Edge handles (at rotated midpoints)
        // Left
        addRectangle(ml[0] - handle_size, ml[1] - handle_size, ml[0] + handle_size, ml[1] + handle_size, 1.0, 1.0, 1.0, 1.0);
        addRectangleStroke(ml[0] - handle_size, ml[1] - handle_size, ml[0] + handle_size, ml[1] + handle_size, handle_stroke, 0.0, 0.478, 1.0, 1.0);
        // Right
        addRectangle(mr[0] - handle_size, mr[1] - handle_size, mr[0] + handle_size, mr[1] + handle_size, 1.0, 1.0, 1.0, 1.0);
        addRectangleStroke(mr[0] - handle_size, mr[1] - handle_size, mr[0] + handle_size, mr[1] + handle_size, handle_stroke, 0.0, 0.478, 1.0, 1.0);
        // Top
        addRectangle(mt[0] - handle_size, mt[1] - handle_size, mt[0] + handle_size, mt[1] + handle_size, 1.0, 1.0, 1.0, 1.0);
        addRectangleStroke(mt[0] - handle_size, mt[1] - handle_size, mt[0] + handle_size, mt[1] + handle_size, handle_stroke, 0.0, 0.478, 1.0, 1.0);
        // Bottom
        addRectangle(mb[0] - handle_size, mb[1] - handle_size, mb[0] + handle_size, mb[1] + handle_size, 1.0, 1.0, 1.0, 1.0);
        addRectangleStroke(mb[0] - handle_size, mb[1] - handle_size, mb[0] + handle_size, mb[1] + handle_size, handle_stroke, 0.0, 0.478, 1.0, 1.0);

        // Rotation handle (circle above top edge, in rotated space)
        const rotation_handle_offset = 0.06 / zoom;
        const rot_handle = rotatePoint(cx, cy + hh + rotation_handle_offset, cx, cy, cos_r, sin_r);
        // Draw connecting line from top-center to rotation handle
        addLineSegment(mt[0], mt[1], rot_handle[0], rot_handle[1], handle_stroke, 0.0, 0.478, 1.0, 1.0);
        // Draw rotation handle circle (white fill with blue stroke)
        addEllipseFill(rot_handle[0], rot_handle[1], handle_size, handle_size, 1.0, 1.0, 1.0, 1.0);
        addEllipseStroke(rot_handle[0], rot_handle[1], handle_size, handle_size, handle_stroke, 0.0, 0.478, 1.0, 1.0);
    }

    // Draw selection highlight for groups
    for (0..shape_count) |j| {
        const shape = shapes[j];
        if (shape.shape_type != 2 or !shape.visible) continue;
        if (!multi_selection[j]) continue;

        const bounds = getGroupBounds(j);
        if (bounds.has_children) {
            const sel_x1 = bounds.min_x - 0.01 / zoom;
            const sel_y1 = bounds.min_y - 0.01 / zoom;
            const sel_x2 = bounds.max_x + 0.01 / zoom;
            const sel_y2 = bounds.max_y + 0.01 / zoom;
            addRectangleStroke(sel_x1, sel_y1, sel_x2, sel_y2, 0.005 / zoom, 0.0, 0.478, 1.0, 1.0);
        }
    }

    // Draw selection highlight and points for selected path shapes
    for (0..shape_count) |i| {
        const shape = shapes[i];
        if (!shape.visible or shape.shape_type != 3) continue;
        if (!multi_selection[i]) continue;

        // This is a selected path - show bounding box and points
        if (shape.path_index < path_count) {
            const path = paths[shape.path_index];

            // Draw bounding box
            const bounds = getPathBounds(shape.path_index);
            if (bounds.has_points) {
                const sel_x1 = bounds.min_x - 0.01 / zoom;
                const sel_y1 = bounds.min_y - 0.01 / zoom;
                const sel_x2 = bounds.max_x + 0.01 / zoom;
                const sel_y2 = bounds.max_y + 0.01 / zoom;
                addRectangleStroke(sel_x1, sel_y1, sel_x2, sel_y2, 0.005 / zoom, 0.0, 0.478, 1.0, 1.0);
            }

            // Draw path points
            const point_size = 0.01 / zoom;
            for (0..path.point_count) |pt_idx| {
                const pt = path.points[pt_idx];
                renderPathPoint(pt[0], pt[1], point_size, 0.0, 0.478, 1.0, 1.0);
            }
        }
    }

    // Draw selection marquee while selecting
    if (selection_box_active and current_tool == 2) {
        const sel_min_x = @min(selection_box_start_x, current_x);
        const sel_max_x = @max(selection_box_start_x, current_x);
        const sel_min_y = @min(selection_box_start_y, current_y);
        const sel_max_y = @max(selection_box_start_y, current_y);

        // Blue stroke only (transparent box)
        addRectangleStroke(sel_min_x, sel_min_y, sel_max_x, sel_max_y, 0.003 / zoom, 0.0, 0.478, 1.0, 0.8);
    }

    // Draw preview shape while dragging (not when zooming, only for shape tools)
    if (is_dragging and !meta_key_pressed and current_tool != 2 and current_tool != 3) {
        const min_x = @min(drag_start_x, current_x);
        const max_x = @max(drag_start_x, current_x);
        const min_y = @min(drag_start_y, current_y);
        const max_y = @max(drag_start_y, current_y);

        if (current_tool == 0) {
            // Ellipse preview
            const cx = (min_x + max_x) / 2.0;
            const cy = (min_y + max_y) / 2.0;
            const rx = (max_x - min_x) / 2.0;
            const ry = (max_y - min_y) / 2.0;
            if (fill_enabled) {
                addEllipseFill(cx, cy, rx, ry, fill_color[0], fill_color[1], fill_color[2], 0.5);
            }
            if (stroke_enabled) {
                addEllipseStroke(cx, cy, rx, ry, stroke_width / zoom, stroke_color[0], stroke_color[1], stroke_color[2], 0.7);
            }
        } else {
            // Rectangle preview
            if (fill_enabled) {
                addRectangle(min_x, min_y, max_x, max_y, fill_color[0], fill_color[1], fill_color[2], 0.5);
            }
            if (stroke_enabled) {
                addRectangleStroke(min_x, min_y, max_x, max_y, stroke_width / zoom, stroke_color[0], stroke_color[1], stroke_color[2], 0.7);
            }
        }
    }

    // Draw pen path preview while drawing
    if (pen_drawing and pen_path_index < path_count) {
        const path = paths[pen_path_index];

        // Draw the path so far
        if (path.point_count >= 1) {
            // Draw path segments
            for (0..path.point_count) |i| {
                if (i > 0) {
                    const p1 = path.points[i - 1];
                    const p2 = path.points[i];
                    addLineSegment(p1[0], p1[1], p2[0], p2[1], stroke_width / zoom, stroke_color[0], stroke_color[1], stroke_color[2], 1.0);
                }
            }

            // Draw line from last point to cursor
            const last_pt = path.points[path.point_count - 1];
            addLineSegment(last_pt[0], last_pt[1], current_x, current_y, stroke_width / zoom, stroke_color[0], stroke_color[1], stroke_color[2], 0.5);

            // Draw point markers
            const point_size = 0.01 / zoom;
            for (0..path.point_count) |i| {
                const pt = path.points[i];
                // First point is highlighted (can close path by clicking it)
                if (i == 0 and path.point_count >= 2) {
                    renderPathPoint(pt[0], pt[1], point_size * 1.5, 0.0, 0.8, 0.2, 1.0);
                } else {
                    renderPathPoint(pt[0], pt[1], point_size, 0.0, 0.478, 1.0, 1.0);
                }
            }
        }
    }

    // Draw bezier handles for selected point (Cmd+click selection)
    if (pen_selected_point >= 0 and pen_selected_path >= 0) {
        const path_idx: usize = @intCast(pen_selected_path);
        const pt_idx: usize = @intCast(pen_selected_point);
        if (path_idx < path_count) {
            const path = paths[path_idx];
            if (pt_idx < path.point_count) {
                const anchor = path.points[pt_idx];
                const ctrl_in = path.control_points[pt_idx][0];
                const ctrl_out = path.control_points[pt_idx][1];

                const handle_size = 0.008 / zoom;
                const line_width = 0.002 / zoom;

                // Draw handle lines (from anchor to control points)
                addLineSegment(anchor[0], anchor[1], ctrl_in[0], ctrl_in[1], line_width, 0.6, 0.6, 0.6, 0.8);
                addLineSegment(anchor[0], anchor[1], ctrl_out[0], ctrl_out[1], line_width, 0.6, 0.6, 0.6, 0.8);

                // Draw anchor point (larger, highlighted)
                renderPathPoint(anchor[0], anchor[1], handle_size * 1.5, 0.0, 0.478, 1.0, 1.0);

                // Draw control handle points (smaller, different color)
                // In handle - affects curve coming INTO this point
                renderPathPoint(ctrl_in[0], ctrl_in[1], handle_size, 1.0, 0.4, 0.0, 1.0);
                // Out handle - affects curve going OUT from this point
                renderPathPoint(ctrl_out[0], ctrl_out[1], handle_size, 1.0, 0.4, 0.0, 1.0);
            }
        }
    }

    // Track UI overlay vertex count
    ui_vertex_count = vertex_count - ui_start_vertex;
}

// Returns true if the world-space bounding box is at least partially visible in the viewport
fn isBoundsVisible(world_min_x: f32, world_min_y: f32, world_max_x: f32, world_max_y: f32) bool {
    // Transform bounds to screen space (NDC)
    const p1 = transformToScreen(world_min_x, world_min_y);
    const p2 = transformToScreen(world_max_x, world_max_y);
    const screen_min_x = @min(p1[0], p2[0]);
    const screen_max_x = @max(p1[0], p2[0]);
    const screen_min_y = @min(p1[1], p2[1]);
    const screen_max_y = @max(p1[1], p2[1]);
    // NDC viewport is [-1, 1] in both axes
    return screen_max_x >= -1.0 and screen_min_x <= 1.0 and screen_max_y >= -1.0 and screen_min_y <= 1.0;
}

export fn getFrameCount() u32 {
    return frame_count;
}

// ============================================================================
// Layer management exports
// ============================================================================

export fn getShapeCount() usize {
    return shape_count;
}

export fn getShapeType(index: usize) u32 {
    if (index >= shape_count) return 0;
    return shapes[index].shape_type;
}

export fn getShapeParent(index: usize) i32 {
    if (index >= shape_count) return -1;
    return shapes[index].parent_id;
}

export fn getShapeVisible(index: usize) u32 {
    if (index >= shape_count) return 0;
    return if (shapes[index].visible) 1 else 0;
}

export fn getShapeX(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].x;
}

export fn getShapeY(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].y;
}

export fn getShapeWidth(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].width * 2.0; // Return full width, not half-width
}

export fn getShapeHeight(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].height * 2.0; // Return full height, not half-height
}

export fn setShapeTransform(index: usize, x: f32, y: f32, w: f32, h: f32) void {
    if (index >= shape_count) return;
    pushHistorySnapshot(); // Save state before transform
    shapes[index].x = x;
    shapes[index].y = y;
    shapes[index].width = w / 2.0; // Store as half-width
    shapes[index].height = h / 2.0; // Store as half-height
    markDirty();
}

export fn setShapeVisible(index: usize, visible: u32) void {
    if (index >= shape_count) return;
    pushHistorySnapshot(); // Save state before visibility change
    shapes[index].visible = visible != 0;
    markDirty();
}

export fn setShapeParent(index: usize, parent: i32) void {
    if (index >= shape_count) return;
    pushHistorySnapshot(); // Save state before parent change
    shapes[index].parent_id = parent;
    markDirty();
}

export fn setSelectedShape(index: i32) void {
    selected_shape = index;
}

export fn getSelectedShape() i32 {
    return selected_shape;
}

export fn isShapeSelected(index: usize) u32 {
    if (index >= MAX_SHAPES) return 0;
    return if (multi_selection[index]) 1 else 0;
}

export fn clearSelection() void {
    for (0..MAX_SHAPES) |i| {
        multi_selection[i] = false;
    }
    selected_shape = -1;
    markDirty();
}

// ============================================================================
// Shape style and transform updates
// ============================================================================

// Update style for all selected shapes
export fn updateSelectedStyle(fill_en: u32, fill_r: f32, fill_g: f32, fill_b: f32, stroke_en: u32, stroke_r: f32, stroke_g: f32, stroke_b: f32, stroke_w: f32, corner_r: f32, stroke_cap: u32, stroke_join: u32, dash_length: f32, dash_gap: f32) void {
    pushHistorySnapshot(); // Save state before style change
    for (0..shape_count) |i| {
        if (multi_selection[i]) {
            shapes[i].fill_enabled = fill_en != 0;
            shapes[i].fill_color = .{ fill_r, fill_g, fill_b };
            shapes[i].stroke_enabled = stroke_en != 0;
            shapes[i].stroke_color = .{ stroke_r, stroke_g, stroke_b };
            shapes[i].stroke_width = stroke_w;
            // Only update corner radius for rectangles
            if (shapes[i].shape_type == 1) {
                shapes[i].corner_radius = corner_r;
            }
            // Update stroke cap and join styles
            shapes[i].stroke_cap = @enumFromInt(stroke_cap);
            shapes[i].stroke_join = @enumFromInt(stroke_join);
            // Update dash pattern
            shapes[i].dash_array[0] = dash_length;
            shapes[i].dash_array[1] = dash_gap;
        }
    }
    markDirty();
}

// Move all selected shapes by delta
export fn moveSelectedShapes(dx: f32, dy: f32) void {
    for (0..shape_count) |i| {
        if (multi_selection[i]) {
            shapes[i].x += dx;
            shapes[i].y += dy;

            // Also move children
            const parent_id = @as(i32, @intCast(i));
            for (0..shape_count) |j| {
                if (shapes[j].parent_id == parent_id and !multi_selection[j]) {
                    shapes[j].x += dx;
                    shapes[j].y += dy;
                }
            }
        }
    }
}

// Get shape style info for UI
export fn getShapeFillEnabled(index: usize) u32 {
    if (index >= shape_count) return 0;
    return if (shapes[index].fill_enabled) 1 else 0;
}

export fn getShapeFillR(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].fill_color[0];
}

export fn getShapeFillG(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].fill_color[1];
}

export fn getShapeFillB(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].fill_color[2];
}

export fn getShapeStrokeEnabled(index: usize) u32 {
    if (index >= shape_count) return 0;
    return if (shapes[index].stroke_enabled) 1 else 0;
}

export fn getShapeStrokeR(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].stroke_color[0];
}

export fn getShapeStrokeG(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].stroke_color[1];
}

export fn getShapeStrokeB(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].stroke_color[2];
}

export fn getShapeStrokeWidth(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].stroke_width;
}

export fn setShapeStrokeWidth(index: usize, width: f32) void {
    if (index >= shape_count) return;
    shapes[index].stroke_width = width;
    markDirty();
}

// Stroke cap style exports
export fn getShapeStrokeCap(index: usize) u32 {
    if (index >= shape_count) return 0;
    return @intFromEnum(shapes[index].stroke_cap);
}

export fn setShapeStrokeCap(index: usize, cap: u32) void {
    if (index >= shape_count) return;
    shapes[index].stroke_cap = @enumFromInt(@as(u8, @intCast(cap % 3)));
    markDirty();
}

// Stroke join style exports
export fn getShapeStrokeJoin(index: usize) u32 {
    if (index >= shape_count) return 0;
    return @intFromEnum(shapes[index].stroke_join);
}

export fn setShapeStrokeJoin(index: usize, join: u32) void {
    if (index >= shape_count) return;
    shapes[index].stroke_join = @enumFromInt(@as(u8, @intCast(join % 3)));
    markDirty();
}

// Dash pattern exports
export fn getShapeDashLength(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].dash_array[0];
}

export fn getShapeDashGap(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].dash_array[1];
}

export fn setShapeDashPattern(index: usize, dash_length: f32, dash_gap: f32) void {
    if (index >= shape_count) return;
    shapes[index].dash_array[0] = dash_length;
    shapes[index].dash_array[1] = dash_gap;
    markDirty();
}

export fn getShapeDashOffset(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].dash_offset;
}

export fn setShapeDashOffset(index: usize, offset: f32) void {
    if (index >= shape_count) return;
    shapes[index].dash_offset = offset;
    markDirty();
}

// Rotation exports
export fn getShapeRotation(index: usize) f32 {
    if (index >= shape_count) return 0;
    return shapes[index].rotation;
}

export fn setShapeRotation(index: usize, rotation: f32) void {
    if (index >= shape_count) return;
    // Mark tiles at old bounds dirty
    markShapeTilesDirty(shapes[index].x, shapes[index].y, shapes[index].width, shapes[index].height);
    shapes[index].rotation = rotation;
    // Mark tiles at new bounds dirty (rotation might expand bounds)
    markShapeTilesDirty(shapes[index].x, shapes[index].y, shapes[index].width, shapes[index].height);
    markDirty();
}

// Rotate selected shapes by delta (in radians)
export fn rotateSelectedShapes(delta: f32) void {
    var has_rotation = false;
    for (0..shape_count) |i| {
        if (multi_selection[i]) {
            has_rotation = true;
            break;
        }
    }
    if (!has_rotation) return;

    pushHistorySnapshot();
    for (0..shape_count) |i| {
        if (multi_selection[i]) {
            markShapeTilesDirty(shapes[i].x, shapes[i].y, shapes[i].width, shapes[i].height);
            shapes[i].rotation += delta;
            markShapeTilesDirty(shapes[i].x, shapes[i].y, shapes[i].width, shapes[i].height);
        }
    }
    markDirty();
}

// Check if any shape is selected
export fn hasSelection() u32 {
    for (0..shape_count) |i| {
        if (multi_selection[i]) return 1;
    }
    return 0;
}

// Create a group from selected shapes
export fn createGroup() i32 {
    if (shape_count >= MAX_SHAPES) return -1;

    pushHistorySnapshot(); // Save state before creating group
    // Create a group shape at the center of selected shapes
    shapes[shape_count] = .{
        .shape_type = 2, // group
        .x = 0,
        .y = 0,
        .width = 0,
        .height = 0,
        .fill_enabled = false,
        .fill_color = .{ 0, 0, 0 },
        .stroke_enabled = false,
        .stroke_color = .{ 0, 0, 0 },
        .stroke_width = 0.0,
        .parent_id = -1,
        .visible = true,
        .corner_radius = 0.0,
        .path_index = 0,
    };
    const group_id = @as(i32, @intCast(shape_count));
    shape_count += 1;
    markDirty();
    return group_id;
}

// Move a shape by delta (also moves children if it's a group)
export fn moveShape(index: usize, dx: f32, dy: f32) void {
    if (index >= shape_count) return;

    shapes[index].x += dx;
    shapes[index].y += dy;

    // Move all children recursively
    const parent_id = @as(i32, @intCast(index));
    for (0..shape_count) |i| {
        if (shapes[i].parent_id == parent_id) {
            moveShape(i, dx, dy);
        }
    }
}

// Delete a shape (and unparent its children)
export fn deleteShape(index: usize) void {
    if (index >= shape_count) return;

    pushHistorySnapshot(); // Save state before deleting
    const deleted_id = @as(i32, @intCast(index));
    const deleted_shape = shapes[index];

    // If deleting a path shape, also delete the path data
    var deleted_path_index: i32 = -1;
    if (deleted_shape.shape_type == 3) {
        deleted_path_index = @as(i32, @intCast(deleted_shape.path_index));
        const path_idx = deleted_shape.path_index;
        if (path_idx < path_count) {
            // Shift paths down
            var p = path_idx;
            while (p < path_count - 1) : (p += 1) {
                paths[p] = paths[p + 1];
            }
            // Clear the last slot to prevent stale data
            paths[path_count - 1] = .{};
            path_count -= 1;
        }
    }

    // Update path_index references in other shapes (for paths after deleted one)
    if (deleted_path_index >= 0) {
        for (0..shape_count) |i| {
            if (shapes[i].shape_type == 3 and shapes[i].path_index > @as(u32, @intCast(deleted_path_index))) {
                shapes[i].path_index -= 1;
            }
        }
    }

    // Unparent children and update parent references
    for (0..shape_count) |i| {
        if (shapes[i].parent_id == deleted_id) {
            shapes[i].parent_id = -1;
        }
        // Update parent references for shapes after deleted one
        if (shapes[i].parent_id > deleted_id) {
            shapes[i].parent_id -= 1;
        }
    }

    // Shift shapes down
    var i = index;
    while (i < shape_count - 1) : (i += 1) {
        shapes[i] = shapes[i + 1];
    }
    shape_count -= 1;

    // Shift multi_selection array down
    var j = index;
    while (j < shape_count) : (j += 1) {
        multi_selection[j] = multi_selection[j + 1];
    }
    // Clear the last slot
    if (shape_count < MAX_SHAPES) {
        multi_selection[shape_count] = false;
    }

    // Clear selection if deleted
    if (selected_shape == deleted_id) {
        selected_shape = -1;
    } else if (selected_shape > deleted_id) {
        selected_shape -= 1;
    }
    markDirty();
}

// Reorder a shape to a new index (for drag and drop in layers panel)
export fn reorderShape(from_index: usize, to_index: usize) void {
    if (from_index >= shape_count or to_index >= shape_count) return;
    if (from_index == to_index) return;

    pushHistorySnapshot(); // Save state before reordering
    const shape_to_move = shapes[from_index];
    const from_id = @as(i32, @intCast(from_index));
    const to_id = @as(i32, @intCast(to_index));

    // Update parent references before moving
    for (0..shape_count) |i| {
        if (shapes[i].parent_id == from_id) {
            // Will be updated after move
            shapes[i].parent_id = -999; // Temp marker
        }
    }

    if (from_index < to_index) {
        // Moving down - shift shapes up
        var i = from_index;
        while (i < to_index) : (i += 1) {
            shapes[i] = shapes[i + 1];
        }
        // Update parent refs for shapes that shifted
        for (0..shape_count) |j| {
            if (shapes[j].parent_id > from_id and shapes[j].parent_id <= to_id) {
                shapes[j].parent_id -= 1;
            }
        }
    } else {
        // Moving up - shift shapes down
        var i = from_index;
        while (i > to_index) : (i -= 1) {
            shapes[i] = shapes[i - 1];
        }
        // Update parent refs for shapes that shifted
        for (0..shape_count) |j| {
            if (shapes[j].parent_id >= to_id and shapes[j].parent_id < from_id) {
                shapes[j].parent_id += 1;
            }
        }
    }

    shapes[to_index] = shape_to_move;

    // Fix temp markers - children now point to new position
    for (0..shape_count) |i| {
        if (shapes[i].parent_id == -999) {
            shapes[i].parent_id = to_id;
        }
    }

    // Update selection
    if (selected_shape == from_id) {
        selected_shape = to_id;
    }

    // Update multi_selection array
    const was_selected = multi_selection[from_index];
    if (from_index < to_index) {
        var i = from_index;
        while (i < to_index) : (i += 1) {
            multi_selection[i] = multi_selection[i + 1];
        }
    } else {
        var i = from_index;
        while (i > to_index) : (i -= 1) {
            multi_selection[i] = multi_selection[i - 1];
        }
    }
    multi_selection[to_index] = was_selected;
    markDirty();
}

// Group all currently selected shapes
export fn groupSelectedShapes() i32 {
    pushHistorySnapshot(); // Save state before grouping
    // Count selected shapes
    var selected_count: usize = 0;
    for (0..shape_count) |i| {
        if (multi_selection[i]) {
            selected_count += 1;
        }
    }

    if (selected_count < 2) return -1; // Need at least 2 shapes to group
    if (shape_count >= MAX_SHAPES) return -1;

    // Create a new group
    const group_id = @as(i32, @intCast(shape_count));
    shapes[shape_count] = .{
        .shape_type = 2, // group
        .x = 0,
        .y = 0,
        .width = 0,
        .height = 0,
        .fill_enabled = false,
        .fill_color = .{ 0, 0, 0 },
        .stroke_enabled = false,
        .stroke_color = .{ 0, 0, 0 },
        .stroke_width = 0.0,
        .parent_id = -1,
        .visible = true,
        .corner_radius = 0.0,
        .path_index = 0,
    };
    shape_count += 1;

    // Set all selected shapes as children of this group
    for (0..shape_count - 1) |i| {
        if (multi_selection[i]) {
            shapes[i].parent_id = group_id;
        }
    }

    // Clear selection and select the group
    for (0..MAX_SHAPES) |i| {
        multi_selection[i] = false;
    }
    multi_selection[@as(usize, @intCast(group_id))] = true;
    selected_shape = group_id;
    markDirty();

    return group_id;
}

// Ungroup a group - removes the group and reparents its children to group's parent
export fn ungroupShape(group_index: usize) void {
    if (group_index >= shape_count) return;
    if (shapes[group_index].shape_type != 2) return; // Not a group

    pushHistorySnapshot(); // Save state before ungrouping
    const group_id = @as(i32, @intCast(group_index));
    const group_parent = shapes[group_index].parent_id; // Inherit group's parent

    // Reparent all direct children to the group's parent and select them
    for (0..MAX_SHAPES) |i| {
        multi_selection[i] = false;
    }
    for (0..shape_count) |i| {
        if (shapes[i].parent_id == group_id) {
            shapes[i].parent_id = group_parent; // Children inherit group's parent
            multi_selection[i] = true;
        }
    }

    // Update parent references for shapes that point to indices after the group
    for (0..shape_count) |i| {
        if (shapes[i].parent_id > group_id) {
            shapes[i].parent_id -= 1;
        }
    }

    // Remove the group by shifting shapes down
    var i = group_index;
    while (i < shape_count - 1) : (i += 1) {
        shapes[i] = shapes[i + 1];
        multi_selection[i] = multi_selection[i + 1];
    }
    shape_count -= 1;
    multi_selection[shape_count] = false;

    // Update selected_shape
    selected_shape = -1;
    for (0..shape_count) |j| {
        if (multi_selection[j]) {
            selected_shape = @as(i32, @intCast(j));
            break;
        }
    }

    markDirty();
}
