# Wiggle

A Figma-style 2D vector editor built with **Zig WASM** core and **WebGPU** rendering, running entirely in a Web Worker for smooth 60fps performance.

![Wiggle Editor](https://img.shields.io/badge/WebGPU-Enabled-green) ![Zig](https://img.shields.io/badge/Zig-0.15+-orange) ![License](https://img.shields.io/badge/License-MIT-blue)

## Features

### Drawing Tools

- **Rectangle Tool** - Create rectangles with optional corner radius
- **Ellipse Tool** - Create circles and ellipses
- **Pen Tool** - Draw arbitrary paths with bezier curve support (click+drag for smooth curves)
- **Select Tool** - Select, move, resize, and rotate shapes

### Styling

- **Fill & Stroke** - Per-shape fill and stroke colors
- **Stroke Options** - Configurable width, caps (butt/round/square), joins (miter/round/bevel)
- **Dashed Strokes** - Customizable dash length and gap
- **Corner Radius** - Rounded corners for rectangles

### Organization

- **Nested Groups** - Arbitrary hierarchy depth with Cmd/Ctrl+G to group, Cmd/Ctrl+U to ungroup
- **Layer Panel** - Visual hierarchy with visibility toggles
- **Undo/Redo** - Full history support (Cmd/Ctrl+Z / Cmd/Ctrl+Y)

### Rendering

- **GPU Compute Shaders** - Geometry generation on GPU (curve flattening, stroke expansion)
- **Adaptive Bezier Subdivision** - 4-128 segments based on curvature and zoom
- **Tile-Based Rendering** - Only re-render dirty regions for efficiency
- **4x MSAA Antialiasing** - Smooth edges with fragment shader AA
- **Object Transforms** - Full rotation and scale support with hierarchy propagation

## Prerequisites

- [Bun](https://bun.sh/) or Node.js
- [Zig](https://ziglang.org/download/) 0.15.0 or later
- WebGPU-capable browser (Chrome 113+, Edge 113+, Firefox 139+)

## Quick Start

```bash
# Install dependencies
bun install

# Development build (fast iteration)
bun run build:wasm:debug && bun run dev

# Production build
bun run prod
```

## Project Structure

```
wiggle/
├── src/
│   ├── main.zig         # Core: scene graph, input, geometry export
│   └── app/
│       ├── App.tsx      # React UI: tool panel, layers, properties
│       └── styles.css   # UI styling
├── www/
│   ├── worker.js        # WebGPU init, WGSL shaders, render loop
│   ├── index.html       # Entry point
│   └── index.js         # Canvas setup, worker communication
├── build.zig            # Zig build configuration
├── ARCHITECTURE.md      # Detailed technical documentation
└── package.json
```

## Architecture

```
┌─────────────────┐    postMessage    ┌──────────────────────────────┐
│  React UI       │ ←───────────────→ │  Web Worker                  │
│  - Tool panel   │                   │  - WebGPU pipelines          │
│  - Layers panel │                   │  - WGSL compute shaders      │
│  - Properties   │                   │  - WASM ↔ GPU buffer sync    │
└─────────────────┘                   └──────────────────────────────┘
                                                 ↓
                                      ┌──────────────────────────────┐
                                      │  Zig WASM                    │
                                      │  - Scene graph (shapes[])    │
                                      │  - Paths with bezier curves  │
                                      │  - Hit testing, selection    │
                                      │  - Undo/redo history         │
                                      └──────────────────────────────┘
```

### Data Flow

1. **Zig exports** shape/path descriptors to linear memory
2. **Worker.js** reads via exported pointers (`getShapeDescriptorPtr()`, etc.)
3. **Uploads** to GPU storage buffers
4. **Compute shaders** generate vertices (curve flattening, stroke expansion)
5. **Render pass** draws from vertex buffer with tile-based dirty tracking

## Keyboard Shortcuts

| Key                  | Action                      |
| -------------------- | --------------------------- |
| **W**                | Toggle wireframe debug mode |
| **Delete/Backspace** | Delete selected shape       |
| **Escape**           | Cancel pen tool / deselect  |
| **Enter**            | Finish pen path             |
| **Cmd/Ctrl+Z**       | Undo                        |
| **Cmd/Ctrl+Y**       | Redo                        |
| **Cmd/Ctrl+G**       | Group selected              |
| **Cmd/Ctrl+U**       | Ungroup selected            |

## Build Commands

| Command                      | Description                  |
| ---------------------------- | ---------------------------- |
| `bun run build:wasm:debug`   | Fast Zig build for iteration |
| `bun run build:wasm:release` | Optimized Zig build          |
| `bun run dev`                | Vite watch mode for UI       |
| `bun run build`              | Full production build        |
| `bun run prod`               | Release WASM + minified UI   |

## Technical Details

See [ARCHITECTURE.md](ARCHITECTURE.md) for in-depth documentation on:

- Adaptive bezier subdivision algorithm
- GPU compute pipeline stages
- Tile-based dirty rendering
- Miter join calculations
- Group hierarchy and transforms

## License

MIT
