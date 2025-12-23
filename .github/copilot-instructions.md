# Wiggle - AI Coding Guidelines

## Architecture Overview

Figma-style 2D vector editor with **Zig WASM** core + **WebGPU** rendering in a Web Worker.

```
┌─────────────────┐    postMessage    ┌──────────────────────────────┐
│  React UI       │ ←───────────────→ │  Web Worker (worker.js)      │
│  src/app/       │                   │  - WebGPU pipelines          │
│  - Tool panel   │                   │  - WGSL compute shaders      │
│  - Layers panel │                   │  - WASM ↔ GPU buffer sync    │
│  - Properties   │                   └──────────────────────────────┘
└─────────────────┘                              ↓
                                       ┌──────────────────────────────┐
                                       │  Zig WASM (src/main.zig)     │
                                       │  - Scene graph (shapes[])    │
                                       │  - Paths with bezier curves  │
                                       │  - Hit testing, selection    │
                                       │  - Undo/redo history         │
                                       └──────────────────────────────┘
```

## Key Files

| File              | Purpose                                           |
| ----------------- | ------------------------------------------------- |
| `src/main.zig`    | Core scene state, input handling, geometry export |
| `www/worker.js`   | WebGPU init, WGSL shaders (embedded), render loop |
| `src/app/App.tsx` | React UI, tool/layer panels, message handling     |
| `ARCHITECTURE.md` | Detailed technical knowledge base                 |

## Build Commands

```bash
bun run build:wasm:debug  # Fast Zig build (iterating on Zig code)
bun run dev               # Vite watch mode (iterating on UI)
bun run build             # Full production build
```

## Data Flow: Zig → GPU

1. **Zig exports** shape/path descriptors to linear memory
2. **Worker.js** reads via `getShapeDescriptorPtr()`, `getPathPointsPtr()`
3. **Uploads** to GPU storage buffers
4. **Compute shaders** generate vertices (curve flattening, stroke expansion)
5. **Render pass** draws from vertex buffer

## Critical Patterns

### Shape Types (in Zig)

```zig
// shape_type values: 0=ellipse, 1=rect, 2=group, 3=path
// Tool IDs: 0=circle, 1=rect, 2=select, 3=pen
// Groups support arbitrary nesting (groups can contain groups)
// parent_id: -1 = root level, >= 0 = index of parent group
```

### Path Data Structure

```zig
// control_points[point_index][handle]: [i][0]=in-handle, [i][1]=out-handle
// segment_types: 0=line, 1=cubic_bezier
```

### Adding WASM Exports

In `main.zig`, use `export` keyword. Call from worker.js via `wasm.instance.exports.functionName()`.

### Modifying Shaders

Shaders are **embedded strings** in `worker.js`:

- `shaderCode` - main vertex/fragment
- `computeShaderCode` - shape geometry generation
- `pathComputeShaderCode` - path/bezier rendering

### UI ↔ Worker Communication

```javascript
// Worker sends to UI:
postMessage({ type: "shapes", shapes, selectedId, hasSelection, currentTool });

// UI sends to Worker:
worker.postMessage({ type: "mouseMove", x, y });
worker.postMessage({ type: "setTool", tool: 3 }); // pen tool
```

## Debugging

- Press **W** to toggle wireframe mode (visualize tessellation)
- Check browser console for `[Zig]` prefixed logs
- Tile rendering stats shown in FPS counter

## Common Tasks

**Add new shape property**: Update `Shape` struct in `main.zig` → Update `ShapeDescriptor` → Update compute shader uniforms → Sync to React via shapes message

**Fix rendering artifacts**: Check `worker.js` compute shaders. Bezier joins use pre-computed miter offsets at path vertices.

**Modify selection behavior**: Look at `handleMouseDown/Up/Move` in `main.zig`, `selected_id` tracking.

## References

See [ARCHITECTURE.md](../ARCHITECTURE.md) for detailed implementation notes on:

- Adaptive bezier subdivision algorithm
- Miter join calculations
- Tile-based dirty rendering
- GPU compute pipeline stages
- Group hierarchy and recursive operations
- Stroke styling (caps, joins, dashes)

## Important: Knowledge Base Updates

**After completing any task**, update the knowledge base:

1. **ARCHITECTURE.md** - Add/update technical details:

   - New features in "Already in Place" checklist
   - Implementation notes for complex systems
   - Configuration constants if changed
   - Data structure changes

2. **copilot-instructions.md** - Update if:
   - New critical patterns or conventions
   - Build process changes
   - New debugging techniques

This ensures future AI sessions have accurate context.
