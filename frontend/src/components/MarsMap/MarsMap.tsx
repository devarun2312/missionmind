import './MarsMap.css';
import type { WaypointInput, MissionPlan } from '../../types/mission';

interface Props {
  /** All candidate waypoints to render (includes base). */
  waypoints: WaypointInput[];
  /** Current mission plan — route is drawn only if present. */
  currentPlan?: MissionPlan | null;
  /** Rover position in world metres. */
  roverX: number;
  roverY: number;
  /** IDs of waypoints currently flagged as hazardous. */
  hazardIds?: Set<string>;
  /** ID of freshly discovered waypoint (shows pulse animation). */
  discoveryId?: string | null;
}

// ── Coordinate system ──────────────────────────────────────────────────────
//
//  World space:  metres from base, +X = right, +Y = up
//  SVG space:    origin top-left, +X = right, +Y = down
//
//  The viewBox is 500 × 420.  Base is anchored at SVG (250, 340).
//  1 world metre = SCALE SVG units.
//
const SVG_W  = 500;
const SVG_H  = 420;
const BASE_X = 250;  // SVG x of world (0,0)
const BASE_Y = 340;  // SVG y of world (0,0)
const SCALE  = 0.9;  // SVG px per world metre

function worldToSvg(wx: number, wy: number): [number, number] {
  return [BASE_X + wx * SCALE, BASE_Y - wy * SCALE];
}

// ── Waypoint colours ───────────────────────────────────────────────────────
function wpFill(isBase: boolean, isHazard: boolean, isDiscovery: boolean): string {
  if (isBase)      return '#39d353';
  if (isHazard)    return '#f85149';
  if (isDiscovery) return '#c084fc';
  return '#58a6ff';
}

// ── Terrain background grid ────────────────────────────────────────────────
function TerrainBackground() {
  const gridLines: React.JSX.Element[] = [];
  // Horizontal grid lines every 50 metres in world space
  for (let worldY = -100; worldY <= 300; worldY += 50) {
    const [, sy] = worldToSvg(0, worldY);
    gridLines.push(
      <line
        key={`hy${worldY}`}
        x1={0} y1={sy}
        x2={SVG_W} y2={sy}
        stroke="#1a2535" strokeWidth="0.5"
      />
    );
  }
  // Vertical grid lines every 50 metres
  for (let worldX = -200; worldX <= 300; worldX += 50) {
    const [sx] = worldToSvg(worldX, 0);
    gridLines.push(
      <line
        key={`vx${worldX}`}
        x1={sx} y1={0}
        x2={sx} y2={SVG_H}
        stroke="#1a2535" strokeWidth="0.5"
      />
    );
  }
  return <>{gridLines}</>;
}

// ── Mars terrain fill patches ──────────────────────────────────────────────
function TerrainPatches() {
  // Stylised rocky outcrops and dust patches using SVG ellipses
  const patches: Array<{ cx: number; cy: number; rx: number; ry: number; opacity: number }> = [
    { cx: 320, cy: 200, rx: 55, ry: 25, opacity: 0.35 },
    { cx: 140, cy: 260, rx: 40, ry: 18, opacity: 0.25 },
    { cx: 390, cy: 310, rx: 30, ry: 14, opacity: 0.30 },
    { cx: 200, cy: 140, rx: 28, ry: 12, opacity: 0.20 },
    { cx: 70,  cy: 180, rx: 22, ry: 10, opacity: 0.18 },
    { cx: 440, cy: 150, rx: 35, ry: 16, opacity: 0.22 },
    { cx: 260, cy: 90,  rx: 45, ry: 20, opacity: 0.28 },
  ];
  return (
    <>
      {patches.map((p, i) => (
        <ellipse
          key={i}
          cx={p.cx} cy={p.cy}
          rx={p.rx} ry={p.ry}
          fill="#c84b17"
          opacity={p.opacity}
        />
      ))}
    </>
  );
}

// ── Route polyline ─────────────────────────────────────────────────────────
function RouteLine({ plan }: { plan: MissionPlan }) {
  if (plan.waypoints.length < 2) return null;
  const points = plan.waypoints
    .map(wp => worldToSvg(wp.x, wp.y).join(','))
    .join(' ');
  return (
    <>
      {/* Shadow / glow under route */}
      <polyline
        points={points}
        fill="none"
        stroke="#39d353"
        strokeWidth="4"
        strokeOpacity="0.1"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {/* Main route line */}
      <polyline
        points={points}
        fill="none"
        stroke="#39d353"
        strokeWidth="1.5"
        strokeDasharray="6 3"
        strokeLinejoin="round"
        strokeLinecap="round"
        strokeOpacity="0.85"
      />
    </>
  );
}

// ── Rover icon ─────────────────────────────────────────────────────────────
function RoverMarker({ x, y }: { x: number; y: number }) {
  const [sx, sy] = worldToSvg(x, y);
  return (
    <g transform={`translate(${sx},${sy})`}>
      {/* Rover pulse ring */}
      <circle r="10" fill="none" stroke="#e07a3a" strokeWidth="0.8" strokeOpacity="0.4" />
      {/* Rover body */}
      <rect x="-6" y="-4" width="12" height="7" rx="1.5" fill="#e07a3a" />
      {/* Solar panels */}
      <rect x="-10" y="-3" width="4" height="2" rx="0.5" fill="#58a6ff" opacity="0.9" />
      <rect x="6"  y="-3" width="4" height="2" rx="0.5" fill="#58a6ff" opacity="0.9" />
      {/* Antenna */}
      <line x1="0" y1="-4" x2="0" y2="-8" stroke="#c9d1d9" strokeWidth="0.8" />
      <circle cy="-8.5" r="1" fill="#39d353" />
      {/* Wheels */}
      <circle cx="-5" cy="3.5" r="2" fill="#1a2535" stroke="#4a5568" strokeWidth="0.5" />
      <circle cx="5"  cy="3.5" r="2" fill="#1a2535" stroke="#4a5568" strokeWidth="0.5" />
    </g>
  );
}

// ── Base station ───────────────────────────────────────────────────────────
function BaseMarker({ sx, sy }: { sx: number; sy: number }) {
  return (
    <g>
      {/* Base ring */}
      <circle cx={sx} cy={sy} r="14" fill="none" stroke="#39d353" strokeWidth="1" strokeOpacity="0.3" />
      <circle cx={sx} cy={sy} r="10" fill="none" stroke="#39d353" strokeWidth="0.8" strokeOpacity="0.5" />
      {/* Base octagon */}
      <polygon
        points="0,-7 5,-5 7,0 5,5 0,7 -5,5 -7,0 -5,-5"
        transform={`translate(${sx},${sy})`}
        fill="#1a3d22"
        stroke="#39d353"
        strokeWidth="1.2"
      />
      {/* B label */}
      <text
        x={sx}
        y={sy + 4}
        textAnchor="middle"
        fontSize="6"
        fill="#39d353"
        fontWeight="700"
        fontFamily="monospace"
      >
        B
      </text>
    </g>
  );
}

// ── Science waypoint ───────────────────────────────────────────────────────
function WaypointMarker({
  wp,
  isHazard,
  isDiscovery,
}: {
  wp: WaypointInput;
  isHazard: boolean;
  isDiscovery: boolean;
}) {
  const [sx, sy] = worldToSvg(wp.x, wp.y);
  const fill = wpFill(false, isHazard, isDiscovery);
  const r = 6;

  return (
    <g>
      {/* Discovery pulse */}
      {isDiscovery && (
        <circle cx={sx} cy={sy} r={r} fill="none" stroke={fill} strokeWidth="1.5" className="pulse-ring" />
      )}
      {/* Hazard blink ring */}
      {isHazard && (
        <circle cx={sx} cy={sy} r={r + 5} fill="none" stroke={fill} strokeWidth="1.2" className="hazard-ring" />
      )}
      {/* Main dot */}
      <circle cx={sx} cy={sy} r={r} fill={fill} fillOpacity="0.85" />
      <circle cx={sx} cy={sy} r={r} fill="none" stroke={fill} strokeWidth="0.8" />
      {/* Inner pip */}
      <circle cx={sx} cy={sy} r="2" fill="white" fillOpacity="0.6" />
      {/* Label */}
      <text
        x={sx}
        y={sy - r - 4}
        textAnchor="middle"
        fontSize="9"
        fill={fill}
        fontWeight="500"
        fontFamily="-apple-system, Segoe UI, sans-serif"
        opacity="0.9"
      >
        {wp.label || wp.id}
      </text>
    </g>
  );
}

// ── Main component ─────────────────────────────────────────────────────────
export default function MarsMap({
  waypoints,
  currentPlan,
  roverX,
  roverY,
  hazardIds = new Set(),
  discoveryId,
}: Props) {
  const baseWp = waypoints.find(w => w.is_base);
  const scienceWps = waypoints.filter(w => !w.is_base);
  const [baseSx, baseSy] = worldToSvg(baseWp?.x ?? 0, baseWp?.y ?? 0);

  return (
    <div className="mars-map">
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Mars mission terrain map"
      >
        {/* ── Deep space background ── */}
        <rect width={SVG_W} height={SVG_H} fill="#0a1018" />

        {/* ── Terrain base colour (Mars surface) ── */}
        <rect width={SVG_W} height={SVG_H} fill="#1a0e08" opacity="0.7" />

        {/* ── Terrain texture patches ── */}
        <TerrainPatches />

        {/* ── Grid ── */}
        <TerrainBackground />

        {/* ── Coordinate axes (faint) ── */}
        <line
          x1={BASE_X} y1={0} x2={BASE_X} y2={SVG_H}
          stroke="#2a3f55" strokeWidth="0.8" strokeOpacity="0.6"
        />
        <line
          x1={0} y1={BASE_Y} x2={SVG_W} y2={BASE_Y}
          stroke="#2a3f55" strokeWidth="0.8" strokeOpacity="0.6"
        />

        {/* ── Route (only when plan exists) ── */}
        {currentPlan && <RouteLine plan={currentPlan} />}

        {/* ── Base station ── */}
        {baseWp && <BaseMarker sx={baseSx} sy={baseSy} />}

        {/* ── Science waypoints ── */}
        {scienceWps.map(wp => (
          <WaypointMarker
            key={wp.id}
            wp={wp}
            isHazard={hazardIds.has(wp.id)}
            isDiscovery={wp.id === discoveryId}
          />
        ))}

        {/* ── Rover ── */}
        <RoverMarker x={roverX} y={roverY} />

        {/* ── Scale indicator ── */}
        <g transform={`translate(${SVG_W - 90}, ${SVG_H - 20})`}>
          <line x1="0" y1="0" x2={50 * SCALE} y2="0" stroke="#4a5568" strokeWidth="1" />
          <line x1="0" y1="-4" x2="0" y2="4" stroke="#4a5568" strokeWidth="1" />
          <line x1={50 * SCALE} y1="-4" x2={50 * SCALE} y2="4" stroke="#4a5568" strokeWidth="1" />
          <text x={25 * SCALE} y="-6" textAnchor="middle" fontSize="8" fill="#4a5568">50 m</text>
        </g>

        {/* ── North arrow ── */}
        <g transform={`translate(${SVG_W - 22}, 22)`}>
          <line x1="0" y1="8" x2="0" y2="-8" stroke="#4a5568" strokeWidth="1.2" />
          <polygon points="0,-12 -4,-4 4,-4" fill="#4a5568" />
          <text x="0" y="18" textAnchor="middle" fontSize="7" fill="#4a5568">N</text>
        </g>
      </svg>

      {/* Legend */}
      <div className="mars-map__legend">
        <div className="mars-map__legend-item">
          <div className="mars-map__legend-dot" style={{ background: '#39d353' }} />
          Base Station
        </div>
        <div className="mars-map__legend-item">
          <div className="mars-map__legend-dot" style={{ background: '#58a6ff' }} />
          Science Target
        </div>
        {[...hazardIds].length > 0 && (
          <div className="mars-map__legend-item">
            <div className="mars-map__legend-dot" style={{ background: '#f85149' }} />
            Terrain Hazard
          </div>
        )}
        {discoveryId && (
          <div className="mars-map__legend-item">
            <div className="mars-map__legend-dot" style={{ background: '#c084fc' }} />
            New Discovery
          </div>
        )}
      </div>
    </div>
  );
}
