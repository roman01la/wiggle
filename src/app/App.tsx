import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================================
// Types
// ============================================================================

type Tool = "select" | "circle" | "rect" | "pen";
type StrokeCap = "butt" | "round" | "square";
type StrokeJoin = "miter" | "round" | "bevel";

interface StyleState {
  fillEnabled: boolean;
  fillColor: string;
  strokeEnabled: boolean;
  strokeColor: string;
  strokeWidth: number; // in pixels
  cornerRadius: number; // in pixels (for rectangles)
  // Path stroke styling
  strokeCap: StrokeCap;
  strokeJoin: StrokeJoin;
  dashLength: number; // 0 = solid line
  dashGap: number;
}

interface ShapeInfo {
  id: number;
  type: number; // 0 = circle, 1 = rect, 2 = group
  parentId: number;
  visible: boolean;
  selected: boolean;
  x: number;
  y: number;
  width: number;
  height: number;
  fillEnabled: boolean;
  fillColor: [number, number, number];
  strokeEnabled: boolean;
  strokeColor: [number, number, number];
  strokeWidth: number; // in NDC
}

// ============================================================================
// Utility Functions
// ============================================================================

function hexToRgb(hex: string) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16) / 255,
        g: parseInt(result[2], 16) / 255,
        b: parseInt(result[3], 16) / 255,
      }
    : { r: 0, g: 0, b: 0 };
}

function rgbToHex(r: number, g: number, b: number): string {
  const toHex = (c: number) => {
    const hex = Math.round(c * 255).toString(16);
    return hex.length === 1 ? "0" + hex : hex;
  };
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function getShapeName(type: number, id: number): string {
  switch (type) {
    case 0:
      return `Circle ${id + 1}`;
    case 1:
      return `Rectangle ${id + 1}`;
    case 2:
      return `Group ${id + 1}`;
    case 3:
      return `Path ${id + 1}`;
    default:
      return `Shape ${id + 1}`;
  }
}

function getShapeIcon(type: number): React.ReactNode {
  switch (type) {
    case 0:
      return (
        <svg viewBox="0 0 24 24" width="16" height="16">
          <circle
            cx="12"
            cy="12"
            r="8"
            stroke="currentColor"
            fill="none"
            strokeWidth="2"
          />
        </svg>
      );
    case 1:
      return (
        <svg viewBox="0 0 24 24" width="16" height="16">
          <rect
            x="4"
            y="4"
            width="16"
            height="16"
            rx="2"
            stroke="currentColor"
            fill="none"
            strokeWidth="2"
          />
        </svg>
      );
    case 2:
      return (
        <svg viewBox="0 0 24 24" width="16" height="16">
          <path
            d="M3 7h18M3 12h18M3 17h18"
            stroke="currentColor"
            fill="none"
            strokeWidth="2"
          />
        </svg>
      );
    default:
      return null;
  }
}

// ============================================================================
// Components
// ============================================================================

function ToolButton({
  active,
  onClick,
  title,
  children,
}: {
  active: boolean;
  onClick: () => void;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <button
      className={`tool-btn ${active ? "active" : ""}`}
      onClick={onClick}
      title={title}
    >
      {children}
    </button>
  );
}

function ColorPicker({
  label,
  enabled,
  color,
  onColorChange,
  onToggle,
}: {
  label: string;
  enabled: boolean;
  color: string;
  onColorChange: (color: string) => void;
  onToggle: () => void;
}) {
  return (
    <div className="style-group">
      <span className="style-label">{label}</span>
      <div className="color-picker-row">
        <div className={`color-swatch ${!enabled ? "transparent" : ""}`}>
          <div
            className="swatch-color"
            style={{ background: enabled ? color : "transparent" }}
          />
          <input
            type="color"
            value={color}
            onChange={(e) => onColorChange(e.target.value)}
          />
        </div>
        <span className="color-hex">{enabled ? color : "None"}</span>
        <button className="toggle-btn" onClick={onToggle}>
          {enabled ? "On" : "None"}
        </button>
      </div>
    </div>
  );
}

function Toolbar({
  currentTool,
  onToolChange,
}: {
  currentTool: Tool;
  onToolChange: (tool: Tool) => void;
}) {
  return (
    <div id="toolbar">
      <ToolButton
        active={currentTool === "select"}
        onClick={() => onToolChange("select")}
        title="Select (V)"
      >
        <svg viewBox="0 0 24 24">
          <path d="M4 4l8 16 2-6 6-2z" />
        </svg>
      </ToolButton>
      <ToolButton
        active={currentTool === "circle"}
        onClick={() => onToolChange("circle")}
        title="Circle (O)"
      >
        <svg viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="9" />
        </svg>
      </ToolButton>
      <ToolButton
        active={currentTool === "rect"}
        onClick={() => onToolChange("rect")}
        title="Rectangle (R)"
      >
        <svg viewBox="0 0 24 24">
          <rect x="4" y="4" width="16" height="16" rx="2" />
        </svg>
      </ToolButton>
      <ToolButton
        active={currentTool === "pen"}
        onClick={() => onToolChange("pen")}
        title="Pen (P)"
      >
        <svg viewBox="0 0 24 24">
          <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25z" />
          <path d="M20.71 7.04a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z" />
        </svg>
      </ToolButton>
    </div>
  );
}

// Conversion between internal NDC coordinates and display "pixels"
// 1 NDC unit = 1920 display pixels (consistent across screen sizes)
const UNITS_PER_NDC = 1920;

// Convert NDC (center origin) to pixels (top-left origin)
// NDC: X goes from -aspect to +aspect, Y goes from -1 to +1 (up is positive)
// Pixels: X goes from 0 to width, Y goes from 0 to height (down is positive)
function ndcToPixelsX(ndc: number): number {
  const aspect = window.innerWidth / window.innerHeight;
  // NDC X range is [-aspect, aspect], map to [0, 2*aspect*UNITS_PER_NDC]
  return Math.round((ndc + aspect) * UNITS_PER_NDC);
}

function ndcToPixelsY(ndc: number): number {
  // NDC Y range is [-1, 1] with +1 at top, map to [0, 2*UNITS_PER_NDC] with 0 at top
  return Math.round((1 - ndc) * UNITS_PER_NDC);
}

function pixelsToNdcX(pixels: number): number {
  const aspect = window.innerWidth / window.innerHeight;
  return pixels / UNITS_PER_NDC - aspect;
}

function pixelsToNdcY(pixels: number): number {
  return 1 - pixels / UNITS_PER_NDC;
}

function ndcToPixelsSize(ndc: number): number {
  return Math.round(ndc * UNITS_PER_NDC);
}

function pixelsToNdcSize(pixels: number): number {
  return pixels / UNITS_PER_NDC;
}

function Sidebar({
  style,
  onStyleChange,
  selectedShape,
  onTransformChange,
  currentTool,
}: {
  style: StyleState;
  onStyleChange: (style: Partial<StyleState>) => void;
  selectedShape: ShapeInfo | null;
  onTransformChange: (
    x: number,
    y: number,
    width: number,
    height: number
  ) => void;
  currentTool: Tool;
}) {
  // Local state for input values (to allow typing without immediate updates)
  const [localX, setLocalX] = useState("");
  const [localY, setLocalY] = useState("");
  const [localW, setLocalW] = useState("");
  const [localH, setLocalH] = useState("");

  // Track which input is focused to avoid overwriting while typing
  const [focusedInput, setFocusedInput] = useState<string | null>(null);

  // Sync local state when selection changes or shape transforms (convert NDC to pixels for display)
  // Only update inputs that aren't currently focused
  useEffect(() => {
    if (selectedShape && selectedShape.type !== 2) {
      if (focusedInput !== "x")
        setLocalX(String(ndcToPixelsX(selectedShape.x)));
      if (focusedInput !== "y")
        setLocalY(String(ndcToPixelsY(selectedShape.y)));
      if (focusedInput !== "w")
        setLocalW(String(ndcToPixelsSize(selectedShape.width)));
      if (focusedInput !== "h")
        setLocalH(String(ndcToPixelsSize(selectedShape.height)));
    }
  }, [
    selectedShape?.id,
    selectedShape?.x,
    selectedShape?.y,
    selectedShape?.width,
    selectedShape?.height,
    focusedInput,
  ]);

  const handleTransformSubmit = () => {
    if (!selectedShape) return;
    // Parse as pixels and convert back to NDC
    const xPx = parseFloat(localX);
    const yPx = parseFloat(localY);
    const wPx = parseFloat(localW);
    const hPx = parseFloat(localH);
    if (
      !isNaN(xPx) &&
      !isNaN(yPx) &&
      !isNaN(wPx) &&
      !isNaN(hPx) &&
      wPx > 0 &&
      hPx > 0
    ) {
      onTransformChange(
        pixelsToNdcX(xPx),
        pixelsToNdcY(yPx),
        pixelsToNdcSize(wPx),
        pixelsToNdcSize(hPx)
      );
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleTransformSubmit();
      (e.target as HTMLInputElement).blur();
    }
  };

  const transformInputDisabled = !(selectedShape && selectedShape.type !== 2);

  return (
    <div id="sidebar">
      <div className="transform-section">
        <div className="transform-row">
          <label>X</label>
          <input
            disabled={transformInputDisabled}
            type="text"
            value={localX}
            onChange={(e) => setLocalX(e.target.value)}
            onFocus={() => setFocusedInput("x")}
            onBlur={() => {
              setFocusedInput(null);
              handleTransformSubmit();
            }}
            onKeyDown={handleKeyDown}
          />
        </div>
        <div className="transform-row">
          <label>Y</label>
          <input
            disabled={transformInputDisabled}
            type="text"
            value={localY}
            onChange={(e) => setLocalY(e.target.value)}
            onFocus={() => setFocusedInput("y")}
            onBlur={() => {
              setFocusedInput(null);
              handleTransformSubmit();
            }}
            onKeyDown={handleKeyDown}
          />
        </div>
        <div className="transform-row">
          <label>W</label>
          <input
            disabled={transformInputDisabled}
            type="text"
            value={localW}
            onChange={(e) => setLocalW(e.target.value)}
            onFocus={() => setFocusedInput("w")}
            onBlur={() => {
              setFocusedInput(null);
              handleTransformSubmit();
            }}
            onKeyDown={handleKeyDown}
          />
        </div>
        <div className="transform-row">
          <label>H</label>
          <input
            disabled={transformInputDisabled}
            type="text"
            value={localH}
            onChange={(e) => setLocalH(e.target.value)}
            onFocus={() => setFocusedInput("h")}
            onBlur={() => {
              setFocusedInput(null);
              handleTransformSubmit();
            }}
            onKeyDown={handleKeyDown}
          />
        </div>
      </div>

      <ColorPicker
        label="Fill"
        enabled={style.fillEnabled}
        color={style.fillColor}
        onColorChange={(color) => {
          onStyleChange({ fillColor: color, fillEnabled: true });
        }}
        onToggle={() => {
          onStyleChange({ fillEnabled: !style.fillEnabled });
        }}
      />
      <ColorPicker
        label="Stroke"
        enabled={style.strokeEnabled}
        color={style.strokeColor}
        onColorChange={(color) => {
          onStyleChange({ strokeColor: color, strokeEnabled: true });
        }}
        onToggle={() => {
          onStyleChange({ strokeEnabled: !style.strokeEnabled });
        }}
      />
      {style.strokeEnabled && (
        <>
          <div className="stroke-width-section">
            <label>Stroke Width</label>
            <input
              type="number"
              min="1"
              max="50"
              value={style.strokeWidth}
              onChange={(e) => {
                const value = parseInt(e.target.value, 10);
                if (!isNaN(value) && value > 0) {
                  onStyleChange({ strokeWidth: value });
                }
              }}
            />
          </div>

          {/* Stroke Cap - only for paths */}
          {(selectedShape?.type === 3 || currentTool === "pen") && (
            <div className="stroke-style-section">
              <label>Cap</label>
              <select
                value={style.strokeCap}
                onChange={(e) =>
                  onStyleChange({ strokeCap: e.target.value as StrokeCap })
                }
              >
                <option value="butt">Butt</option>
                <option value="round">Round</option>
                <option value="square">Square</option>
              </select>
            </div>
          )}

          {/* Stroke Join - only for paths */}
          {(selectedShape?.type === 3 || currentTool === "pen") && (
            <div className="stroke-style-section">
              <label>Join</label>
              <select
                value={style.strokeJoin}
                onChange={(e) =>
                  onStyleChange({ strokeJoin: e.target.value as StrokeJoin })
                }
              >
                <option value="miter">Miter</option>
                <option value="round">Round</option>
                <option value="bevel">Bevel</option>
              </select>
            </div>
          )}

          {/* Dash Pattern - only for paths */}
          {(selectedShape?.type === 3 || currentTool === "pen") && (
            <div className="dash-pattern-section">
              <label>Dash</label>
              <div className="dash-inputs">
                <input
                  type="number"
                  min="0"
                  max="100"
                  placeholder="Length"
                  value={style.dashLength || ""}
                  onChange={(e) => {
                    const value = parseInt(e.target.value, 10);
                    onStyleChange({ dashLength: isNaN(value) ? 0 : value });
                  }}
                />
                <input
                  type="number"
                  min="0"
                  max="100"
                  placeholder="Gap"
                  value={style.dashGap || ""}
                  onChange={(e) => {
                    const value = parseInt(e.target.value, 10);
                    onStyleChange({ dashGap: isNaN(value) ? 0 : value });
                  }}
                />
              </div>
            </div>
          )}
        </>
      )}

      <div className="corner-radius-section">
        <label>Corner Radius</label>
        <input
          disabled={!(selectedShape?.type === 1 || currentTool === "rect")}
          type="number"
          min="0"
          max="100"
          value={style.cornerRadius}
          onChange={(e) => {
            const value = parseInt(e.target.value, 10);
            if (!isNaN(value) && value >= 0) {
              onStyleChange({ cornerRadius: value });
            }
          }}
        />
      </div>
    </div>
  );
}

interface FPSData {
  fps: number;
  tiles?: {
    dirty: number;
    total: number;
    gridX: number;
    gridY: number;
  };
}

function FPSCounter({
  data,
  debugTiles,
  onToggleDebug,
}: {
  data: FPSData;
  debugTiles: boolean;
  onToggleDebug: () => void;
}) {
  const { fps, tiles } = data;
  return (
    <div id="fps-counter">
      FPS: {fps > 0 ? fps : "--"}
      {tiles && (
        <span style={{ marginLeft: 8, opacity: 0.7 }}>
          Tiles: {tiles.dirty}/{tiles.total}
        </span>
      )}
      <button
        onClick={onToggleDebug}
        style={{
          marginLeft: 8,
          padding: "2px 6px",
          fontSize: 10,
          cursor: "pointer",
          backgroundColor: debugTiles ? "#ff6b6b" : "#e0e0e0",
          color: debugTiles ? "white" : "black",
          border: "none",
          borderRadius: 3,
        }}
      >
        {debugTiles ? "Debug ON" : "Debug"}
      </button>
    </div>
  );
}

// ============================================================================
// Layers Sidebar
// ============================================================================

function LayerItem({
  shape,
  shapes,
  selectedId,
  expandedGroups,
  onSelect,
  onToggleVisibility,
  onToggleExpand,
  onDelete,
  onDragStart,
  onDragOver,
  onDrop,
  dragOverId,
  depth = 0,
}: {
  shape: ShapeInfo;
  shapes: ShapeInfo[];
  selectedId: number;
  expandedGroups: Set<number>;
  onSelect: (id: number) => void;
  onToggleVisibility: (id: number) => void;
  onToggleExpand: (id: number) => void;
  onDelete: (id: number) => void;
  onDragStart: (id: number) => void;
  onDragOver: (id: number) => void;
  onDrop: (targetId: number) => void;
  dragOverId: number | null;
  depth?: number;
}) {
  const isGroup = shape.type === 2;
  const isExpanded = expandedGroups.has(shape.id);
  const children = shapes.filter((s) => s.parentId === shape.id);
  const isSelected = selectedId === shape.id || shape.selected;
  const isDragOver = dragOverId === shape.id;

  return (
    <div className="layer-item-container">
      <div
        className={`layer-item ${isSelected ? "selected" : ""} ${
          !shape.visible ? "hidden-layer" : ""
        } ${isDragOver ? "drag-over" : ""}`}
        style={{ paddingLeft: 8 + depth * 16 }}
        onClick={() => onSelect(shape.id)}
        draggable
        onDragStart={(e) => {
          e.dataTransfer.effectAllowed = "move";
          onDragStart(shape.id);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
          onDragOver(shape.id);
        }}
        onDragLeave={() => onDragOver(-1)}
        onDrop={(e) => {
          e.preventDefault();
          onDrop(shape.id);
        }}
      >
        {isGroup && (
          <button
            className="expand-btn"
            onClick={(e) => {
              e.stopPropagation();
              onToggleExpand(shape.id);
            }}
          >
            {isExpanded ? "▼" : "▶"}
          </button>
        )}
        <span className="layer-icon">{getShapeIcon(shape.type)}</span>
        <span className="layer-name">{getShapeName(shape.type, shape.id)}</span>
        <button
          className="visibility-btn"
          onClick={(e) => {
            e.stopPropagation();
            onToggleVisibility(shape.id);
          }}
        >
          {shape.visible ? "👁" : "👁‍🗨"}
        </button>
        <button
          className="delete-btn"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(shape.id);
          }}
          title="Delete"
        >
          🗑
        </button>
      </div>
      {isGroup && isExpanded && children.length > 0 && (
        <div className="layer-children">
          {children.map((child) => (
            <LayerItem
              key={child.id}
              shape={child}
              shapes={shapes}
              selectedId={selectedId}
              expandedGroups={expandedGroups}
              onSelect={onSelect}
              onToggleVisibility={onToggleVisibility}
              onToggleExpand={onToggleExpand}
              onDelete={onDelete}
              onDragStart={onDragStart}
              onDragOver={onDragOver}
              onDrop={onDrop}
              dragOverId={dragOverId}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function LayersSidebar({
  shapes,
  selectedId,
  onSelect,
  onToggleVisibility,
  onDelete,
  onReorder,
}: {
  shapes: ShapeInfo[];
  selectedId: number;
  onSelect: (id: number) => void;
  onToggleVisibility: (id: number) => void;
  onDelete: (id: number) => void;
  onReorder: (fromId: number, toId: number) => void;
}) {
  const [expandedGroups, setExpandedGroups] = useState<Set<number>>(new Set());
  const [dragFromId, setDragFromId] = useState<number | null>(null);
  const [dragOverId, setDragOverId] = useState<number | null>(null);

  const toggleExpand = useCallback((id: number) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const handleDragStart = useCallback((id: number) => {
    setDragFromId(id);
  }, []);

  const handleDragOver = useCallback((id: number) => {
    setDragOverId(id);
  }, []);

  const handleDrop = useCallback(
    (targetId: number) => {
      if (dragFromId !== null && dragFromId !== targetId) {
        onReorder(dragFromId, targetId);
      }
      setDragFromId(null);
      setDragOverId(null);
    },
    [dragFromId, onReorder]
  );

  // Get top-level shapes (no parent)
  const rootShapes = shapes.filter((s) => s.parentId === -1);

  return (
    <div id="layers-sidebar">
      <div className="layers-header">
        <span className="layers-title">Layers</span>
      </div>
      <div className="layers-list">
        {rootShapes.length === 0 ? (
          <div className="layers-empty">No shapes yet</div>
        ) : (
          rootShapes.map((shape) => (
            <LayerItem
              key={shape.id}
              shape={shape}
              shapes={shapes}
              selectedId={selectedId}
              expandedGroups={expandedGroups}
              onSelect={onSelect}
              onToggleVisibility={onToggleVisibility}
              onToggleExpand={toggleExpand}
              onDelete={onDelete}
              onDragStart={handleDragStart}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              dragOverId={dragOverId}
            />
          ))
        )}
      </div>
    </div>
  );
}

function ErrorOverlay({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div id="error" style={{ display: "block" }}>
      <h2>Error</h2>
      <p style={{ marginTop: 10 }}>{message}</p>
    </div>
  );
}

// ============================================================================
// Main App
// ============================================================================

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);

  const [currentTool, setCurrentTool] = useState<Tool>("select");
  const [style, setStyle] = useState<StyleState>({
    fillEnabled: false,
    fillColor: "#ffffff",
    strokeEnabled: true,
    strokeColor: "#000000",
    strokeWidth: 2,
    cornerRadius: 0,
    strokeCap: "butt",
    strokeJoin: "miter",
    dashLength: 0,
    dashGap: 0,
  });
  const [fpsData, setFpsData] = useState<FPSData>({ fps: 0 });
  const [error, setError] = useState<string | null>(null);
  const [shapes, setShapes] = useState<ShapeInfo[]>([]);
  const [selectedShapeId, setSelectedShapeId] = useState(-1);
  const [debugTiles, setDebugTiles] = useState(false);
  const [wireframeMode, setWireframeMode] = useState(false);
  const [hasSelection, setHasSelection] = useState(false);

  // Zoom/pan state (refs to avoid re-renders)
  const zoomRef = useRef(1.0);
  const panXRef = useRef(0);
  const panYRef = useRef(0);
  const metaKeyRef = useRef(false);
  const shiftKeyRef = useRef(false);
  const mousePressedRef = useRef(false);
  const mouseXRef = useRef(0);
  const mouseYRef = useRef(0);
  const zoomStartYRef = useRef(0);
  const zoomStartZoomRef = useRef(1.0);

  // Send style update to worker (for new shapes)
  const sendStyleUpdate = useCallback(() => {
    const worker = workerRef.current;
    if (!worker) return;

    const fill = hexToRgb(style.fillColor);
    const stroke = hexToRgb(style.strokeColor);

    worker.postMessage({
      type: "style",
      fillEnabled: style.fillEnabled,
      fillR: fill.r,
      fillG: fill.g,
      fillB: fill.b,
      strokeEnabled: style.strokeEnabled,
      strokeR: stroke.r,
      strokeG: stroke.g,
      strokeB: stroke.b,
      strokeWidth: style.strokeWidth,
      cornerRadius: style.cornerRadius,
    });
  }, [style]);

  // Send tool update to worker
  const sendToolUpdate = useCallback((tool: Tool) => {
    workerRef.current?.postMessage({ type: "tool", tool });
  }, []);

  // Send zoom update to worker
  const sendZoomUpdate = useCallback(() => {
    workerRef.current?.postMessage({
      type: "zoom",
      zoom: zoomRef.current,
      panX: panXRef.current,
      panY: panYRef.current,
    });
  }, []);

  // Toggle debug tiles
  const toggleDebugTiles = useCallback(() => {
    setDebugTiles((prev) => {
      const newValue = !prev;
      workerRef.current?.postMessage({
        type: "setDebugTiles",
        enabled: newValue,
      });
      return newValue;
    });
  }, []);

  // Toggle wireframe mode for path debugging
  const toggleWireframeMode = useCallback(() => {
    setWireframeMode((prev) => {
      const newValue = !prev;
      workerRef.current?.postMessage({
        type: "setWireframeMode",
        enabled: newValue,
      });
      return newValue;
    });
  }, []);

  // Send mouse update to worker
  const sendMouseUpdate = useCallback(
    (x: number, y: number, pressed: boolean) => {
      const dpr = window.devicePixelRatio || 1;
      workerRef.current?.postMessage({
        type: "mouse",
        x: x * dpr,
        y: y * dpr,
        pressed,
      });
    },
    []
  );

  // Initialize worker and canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Check WebGPU support
    if (!navigator.gpu) {
      setError(
        "WebGPU not supported. Try Chrome 113+, Edge 113+, or Firefox Nightly."
      );
      return;
    }

    // Transfer canvas to worker
    const offscreen = canvas.transferControlToOffscreen();
    const worker = new Worker("worker.js");
    workerRef.current = worker;

    const dpr = window.devicePixelRatio || 1;
    worker.postMessage(
      {
        type: "init",
        canvas: offscreen,
        width: window.innerWidth * dpr,
        height: window.innerHeight * dpr,
        dpr,
      },
      [offscreen]
    );

    // Handle messages from worker
    worker.onmessage = (e) => {
      const { type, data } = e.data;
      if (type === "error") {
        setError(data);
      } else if (type === "fps") {
        setFpsData({ fps: data, tiles: e.data.tiles });
      } else if (type === "shapes") {
        setShapes(e.data.shapes);
        if (e.data.selectedId !== undefined) {
          setSelectedShapeId(e.data.selectedId);
        }
        setHasSelection(e.data.hasSelection || false);

        // Sync tool from Zig (e.g., after creating a shape, it switches to select)
        if (e.data.currentTool !== undefined) {
          const toolMap: Record<number, Tool> = {
            0: "circle",
            1: "rect",
            2: "select",
            3: "pen",
          };
          const newTool = toolMap[e.data.currentTool];
          if (newTool) {
            setCurrentTool(newTool);
          }
        }

        // Sync style from first selected shape
        const selectedShape = e.data.shapes.find((s: ShapeInfo) => s.selected);
        if (selectedShape && selectedShape.type !== 2) {
          // Convert NDC stroke width to pixels (0.01 NDC = 2px), round to integer
          const strokeWidthPx = Math.round(selectedShape.strokeWidth * 200);
          setStyle((prev) => ({
            ...prev,
            fillEnabled: selectedShape.fillEnabled,
            fillColor: rgbToHex(
              selectedShape.fillColor[0],
              selectedShape.fillColor[1],
              selectedShape.fillColor[2]
            ),
            strokeEnabled: selectedShape.strokeEnabled,
            strokeColor: rgbToHex(
              selectedShape.strokeColor[0],
              selectedShape.strokeColor[1],
              selectedShape.strokeColor[2]
            ),
            strokeWidth: strokeWidthPx,
          }));
        }
      }
    };

    // Request initial shapes (will be pushed on subsequent changes)
    worker.postMessage({ type: "getShapes" });

    // Resize handler
    const handleResize = () => {
      const dpr = window.devicePixelRatio || 1;
      worker.postMessage({
        type: "resize",
        width: window.innerWidth * dpr,
        height: window.innerHeight * dpr,
      });
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      worker.terminate();
    };
  }, []);

  // Update style when it changes
  useEffect(() => {
    sendStyleUpdate();
  }, [style, sendStyleUpdate]);

  // Handle tool change
  const handleToolChange = useCallback(
    (tool: Tool) => {
      setCurrentTool(tool);
      sendToolUpdate(tool);
    },
    [sendToolUpdate]
  );

  // Handle style change - updates both default style and selected shapes
  const handleStyleChange = useCallback(
    (partial: Partial<StyleState>) => {
      setStyle((prev) => {
        const newStyle = { ...prev, ...partial };

        // If there's a selection, also update the selected shapes
        if (hasSelection) {
          const fill = hexToRgb(newStyle.fillColor);
          const stroke = hexToRgb(newStyle.strokeColor);
          // Convert stroke width and corner radius to NDC (2px = 0.01 NDC)
          const strokeWidthNDC = (newStyle.strokeWidth || 2) / 200;
          const cornerRadiusNDC = (newStyle.cornerRadius || 0) / 200;
          // Convert dash pattern to NDC
          const dashLengthNDC = (newStyle.dashLength || 0) / 200;
          const dashGapNDC = (newStyle.dashGap || 0) / 200;
          // Convert cap/join to numeric values
          const capValue =
            newStyle.strokeCap === "butt"
              ? 0
              : newStyle.strokeCap === "round"
              ? 1
              : 2;
          const joinValue =
            newStyle.strokeJoin === "miter"
              ? 0
              : newStyle.strokeJoin === "round"
              ? 1
              : 2;
          workerRef.current?.postMessage({
            type: "updateSelectedStyle",
            fillEnabled: newStyle.fillEnabled,
            fillR: fill.r,
            fillG: fill.g,
            fillB: fill.b,
            strokeEnabled: newStyle.strokeEnabled,
            strokeR: stroke.r,
            strokeG: stroke.g,
            strokeB: stroke.b,
            strokeWidth: strokeWidthNDC,
            cornerRadius: cornerRadiusNDC,
            strokeCap: capValue,
            strokeJoin: joinValue,
            dashLength: dashLengthNDC,
            dashGap: dashGapNDC,
          });
        }

        return newStyle;
      });
    },
    [hasSelection]
  );

  // Layer management callbacks
  const handleSelectShape = useCallback((id: number) => {
    setSelectedShapeId(id);
    workerRef.current?.postMessage({ type: "selectShape", index: id });
  }, []);

  const handleToggleVisibility = useCallback(
    (id: number) => {
      const shape = shapes.find((s) => s.id === id);
      if (shape) {
        workerRef.current?.postMessage({
          type: "setShapeVisible",
          index: id,
          visible: !shape.visible,
        });
      }
    },
    [shapes]
  );

  const handleCreateGroup = useCallback(() => {
    // Check if exactly one group is selected - if so, ungroup it
    const selectedShapes = shapes.filter((s) => s.selected);
    if (selectedShapes.length === 1 && selectedShapes[0].type === 2) {
      // Ungroup
      workerRef.current?.postMessage({
        type: "ungroupShape",
        index: selectedShapes[0].id,
      });
    } else {
      // Group all currently selected shapes
      workerRef.current?.postMessage({ type: "groupSelected" });
    }
  }, [shapes]);

  const handleDeleteSelected = useCallback(() => {
    const selectedShapes = shapes.filter((s) => s.selected);
    // Delete in reverse order to avoid index shifting issues
    const sortedByIdDesc = [...selectedShapes].sort((a, b) => b.id - a.id);
    for (const shape of sortedByIdDesc) {
      workerRef.current?.postMessage({
        type: "deleteShape",
        index: shape.id,
      });
    }
  }, [shapes]);

  const handleDeleteShape = useCallback((id: number) => {
    workerRef.current?.postMessage({
      type: "deleteShape",
      index: id,
    });
  }, []);

  const handleReorderShape = useCallback((fromId: number, toId: number) => {
    workerRef.current?.postMessage({
      type: "reorderShape",
      fromIndex: fromId,
      toIndex: toId,
    });
  }, []);

  const handleTransformChange = useCallback(
    (x: number, y: number, width: number, height: number) => {
      const selectedShapes = shapes.filter((s) => s.selected);
      if (selectedShapes.length === 1) {
        workerRef.current?.postMessage({
          type: "setShapeTransform",
          index: selectedShapes[0].id,
          x,
          y,
          width,
          height,
        });
      }
    },
    [shapes]
  );

  // Pointer event handlers (work across entire screen via pointer capture)
  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      // Capture pointer to track mouse even when it leaves the canvas
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      mousePressedRef.current = true;
      // Always send mouse update so Zig knows about the click
      sendMouseUpdate(e.clientX, e.clientY, true);
      if (metaKeyRef.current) {
        // Also track for zoom gesture
        zoomStartYRef.current = e.clientY;
        zoomStartZoomRef.current = zoomRef.current;
      }
    },
    [sendMouseUpdate]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      mouseXRef.current = e.clientX;
      mouseYRef.current = e.clientY;

      if (mousePressedRef.current && metaKeyRef.current) {
        // Zooming with drag
        const deltaY = e.clientY - zoomStartYRef.current;
        const zoomFactor = Math.pow(2, -deltaY / 100);
        zoomRef.current = Math.max(
          0.1,
          Math.min(10, zoomStartZoomRef.current * zoomFactor)
        );
        sendZoomUpdate();
      } else {
        sendMouseUpdate(e.clientX, e.clientY, mousePressedRef.current);
      }
    },
    [sendZoomUpdate, sendMouseUpdate]
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent) => {
      // Release pointer capture
      (e.target as HTMLElement).releasePointerCapture(e.pointerId);
      mousePressedRef.current = false;
      sendMouseUpdate(e.clientX, e.clientY, false);
    },
    [sendMouseUpdate]
  );

  const handleMouseLeave = useCallback(() => {
    // Don't release on leave - pointer capture handles this
  }, []);

  // Touch handlers (native, non-passive to allow preventDefault)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleTouchStart = (e: TouchEvent) => {
      e.preventDefault();
      const touch = e.touches[0];
      mousePressedRef.current = true;
      sendMouseUpdate(touch.clientX, touch.clientY, true);
    };

    const handleTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      const touch = e.touches[0];
      sendMouseUpdate(touch.clientX, touch.clientY, mousePressedRef.current);
    };

    const handleTouchEnd = (e: TouchEvent) => {
      e.preventDefault();
      mousePressedRef.current = false;
      sendMouseUpdate(mouseXRef.current, mouseYRef.current, false);
    };

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (metaKeyRef.current) {
        const zoomFactor = Math.pow(1.002, -e.deltaY);
        zoomRef.current = Math.max(
          0.1,
          Math.min(10, zoomRef.current * zoomFactor)
        );
      } else {
        const dpr = window.devicePixelRatio || 1;
        const aspect = window.innerWidth / window.innerHeight;
        const ndcDeltaX =
          (-e.deltaX * 2.0 * aspect) / (window.innerWidth * dpr);
        const ndcDeltaY = (e.deltaY * 2.0) / (window.innerHeight * dpr);
        panXRef.current += ndcDeltaX;
        panYRef.current += ndcDeltaY;
      }
      sendZoomUpdate();
    };

    canvas.addEventListener("touchstart", handleTouchStart, { passive: false });
    canvas.addEventListener("touchmove", handleTouchMove, { passive: false });
    canvas.addEventListener("touchend", handleTouchEnd, { passive: false });
    canvas.addEventListener("wheel", handleWheel, { passive: false });

    return () => {
      canvas.removeEventListener("touchstart", handleTouchStart);
      canvas.removeEventListener("touchmove", handleTouchMove);
      canvas.removeEventListener("touchend", handleTouchEnd);
      canvas.removeEventListener("wheel", handleWheel);
    };
  }, [sendMouseUpdate, sendZoomUpdate]);

  // Keyboard handlers for meta key and tool shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Meta" || e.key === "Control") {
        metaKeyRef.current = true;
        workerRef.current?.postMessage({ type: "metaKey", pressed: true });
      }
      if (e.key === "Shift") {
        shiftKeyRef.current = true;
        workerRef.current?.postMessage({ type: "shiftKey", pressed: true });
      }

      // Tool shortcuts (only when not typing in an input)
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Cmd+G to group selected shapes
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "g") {
        e.preventDefault();
        handleCreateGroup();
        return;
      }

      // Cmd+Z to undo, Cmd+Shift+Z to redo
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) {
          workerRef.current?.postMessage({ type: "redo" });
        } else {
          workerRef.current?.postMessage({ type: "undo" });
        }
        return;
      }

      // Delete/Backspace to delete selected shapes
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        handleDeleteSelected();
        return;
      }

      switch (e.key.toLowerCase()) {
        case "v":
          if (!e.metaKey && !e.ctrlKey) handleToolChange("select");
          break;
        case "o":
          if (!e.metaKey && !e.ctrlKey) handleToolChange("circle");
          break;
        case "r":
          if (!e.metaKey && !e.ctrlKey) handleToolChange("rect");
          break;
        case "p":
          if (!e.metaKey && !e.ctrlKey) handleToolChange("pen");
          break;
        case "w":
          // Toggle wireframe debug mode
          if (!e.metaKey && !e.ctrlKey) toggleWireframeMode();
          break;
        case "enter":
          // Finish pen path
          workerRef.current?.postMessage({ type: "finishPenPath" });
          break;
        case "escape":
          // Cancel pen path
          workerRef.current?.postMessage({ type: "cancelPenPath" });
          break;
      }

      workerRef.current?.postMessage({
        type: "keydown",
        key: e.key,
        code: e.code,
      });
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Meta" || e.key === "Control") {
        metaKeyRef.current = false;
        workerRef.current?.postMessage({ type: "metaKey", pressed: false });
      }
      if (e.key === "Shift") {
        shiftKeyRef.current = false;
        workerRef.current?.postMessage({ type: "shiftKey", pressed: false });
      }
      workerRef.current?.postMessage({
        type: "keyup",
        key: e.key,
        code: e.code,
      });
    };

    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("keyup", handleKeyUp);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("keyup", handleKeyUp);
    };
  }, [
    handleToolChange,
    handleCreateGroup,
    handleDeleteSelected,
    toggleWireframeMode,
  ]);

  return (
    <>
      <canvas
        ref={canvasRef}
        id="canvas"
        onPointerMove={handlePointerMove}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onMouseLeave={handleMouseLeave}
        style={{ touchAction: "none" }}
      />
      <FPSCounter
        data={fpsData}
        debugTiles={debugTiles}
        onToggleDebug={toggleDebugTiles}
      />
      <Toolbar currentTool={currentTool} onToolChange={handleToolChange} />
      <Sidebar
        style={style}
        onStyleChange={handleStyleChange}
        selectedShape={shapes.find((s) => s.selected) || null}
        onTransformChange={handleTransformChange}
        currentTool={currentTool}
      />
      <LayersSidebar
        shapes={shapes}
        selectedId={selectedShapeId}
        onSelect={handleSelectShape}
        onToggleVisibility={handleToggleVisibility}
        onDelete={handleDeleteShape}
        onReorder={handleReorderShape}
      />
      <ErrorOverlay message={error} />
    </>
  );
}
