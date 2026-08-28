/**
 * TypeScript types derived from the MissionMind Python backend.
 *
 * These interfaces map 1-to-1 to the Pydantic schemas in:
 *   missionmind/api/schemas/planning.py
 *   missionmind/api/schemas/replanning.py
 *   missionmind/models/mission.py
 *   missionmind/models/events.py
 *
 * BATTERY CONVENTION (critical):
 *   battery_pct is always a FRACTION — 0.68 means 68 %.
 *   Never send percentage integers to the API.
 *   Use toPercent(f) for display only.
 */

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

export type MissionStatus =
  | 'PENDING'
  | 'ACTIVE'
  | 'REPLANNING'
  | 'ABORTED'
  | 'COMPLETE';

export type EventType =
  | 'BATTERY_FAILURE'
  | 'COMM_LOSS'
  | 'TERRAIN_HAZARD'
  | 'NEW_DISCOVERY'
  | 'RETURN_TO_BASE';

// ---------------------------------------------------------------------------
// Domain models — received from the API (MissionPlan response)
// ---------------------------------------------------------------------------

/** A single waypoint as returned inside a MissionPlan response. */
export interface Waypoint {
  id: string;
  x: number;
  y: number;
  scientific_value: number;    // 0.0–1.0, assigned by AI agents
  terrain_risk: number;        // 0.0–1.0
  estimated_travel_time_minutes: number;
  estimated_energy_wh: number;
  is_base: boolean;
  label: string;
}

/** A complete mission plan as returned by POST /api/mission/plan or /replan. */
export interface MissionPlan {
  plan_id: string;             // UUID
  waypoints: Waypoint[];
  total_energy_wh: number;
  total_time_minutes: number;
  status: MissionStatus;
  created_at: string;          // ISO-8601 UTC
  reasoning: string;           // LLM-generated explanation
  confidence: number;          // 0.0–1.0
}

// ---------------------------------------------------------------------------
// Request input types — sent TO the API
// ---------------------------------------------------------------------------

/** Rover state sent to /plan and /replan. battery_pct is a FRACTION (0.0–1.0). */
export interface RoverState {
  battery_pct: number;           // FRACTION: 0.68 = 68%
  battery_capacity_wh: number;   // total Wh, > 0
  position_x: number;            // metres from base
  position_y: number;            // metres from base
  rover_speed_mps: number;       // > 0
  power_consumption_w: number;   // > 0
}

/** A candidate waypoint sent to the API. scientific_value is optional. */
export interface WaypointInput {
  id: string;
  x: number;
  y: number;
  terrain_risk: number;                   // 0.0–1.0
  is_base: boolean;
  label: string;
  estimated_travel_time_minutes: number;  // >= 0
  estimated_energy_wh: number;            // >= 0
  scientific_value?: number;              // 0.0–1.0, optional hint
}

/** Weather forecast sent to the API. */
export interface WeatherForecast {
  dust_storm_probability: number;  // 0.0–1.0
  temperature_min_c: number;
  temperature_max_c: number;
  wind_speed_mps: number;          // >= 0
  forecast_hours: number;          // integer >= 1
}

/** A single communication window with mission control. */
export interface CommWindow {
  start_utc: string;         // ISO-8601 UTC
  duration_minutes: number;  // integer >= 0
}

/** Environment state sent to /plan and /replan. */
export interface EnvState {
  candidate_waypoints: WaypointInput[];        // must contain at least one is_base=true
  weather_forecast: WeatherForecast;
  comm_windows: CommWindow[];
  terrain_map: Record<string, unknown>;        // may be {}
  mission_objectives: string[];               // at least 1
}

// ---------------------------------------------------------------------------
// Event types — sent in /replan requests
// ---------------------------------------------------------------------------

/**
 * Event-type-specific payload data.
 *
 * BATTERY_FAILURE  → { battery_pct: 0.07 }             (fraction)
 * COMM_LOSS        → { safe_comm_radius_m: 150.0 }
 * TERRAIN_HAZARD   → { waypoint_id: "wp-crater-ridge" }
 * NEW_DISCOVERY    → { x, y, id?, label?, scientific_value?, terrain_risk?,
 *                       estimated_travel_time_minutes?, estimated_energy_wh? }
 * RETURN_TO_BASE   → {}
 */
export interface MissionEventPayload {
  // BATTERY_FAILURE
  battery_pct?: number;
  // COMM_LOSS
  safe_comm_radius_m?: number;
  // TERRAIN_HAZARD
  waypoint_id?: string;
  neighbour_ids?: string[];
  // NEW_DISCOVERY
  x?: number;
  y?: number;
  id?: string;
  label?: string;
  scientific_value?: number;
  terrain_risk?: number;
  estimated_travel_time_minutes?: number;
  estimated_energy_wh?: number;
}

/** An event sent inside a /replan request body. */
export interface MissionEvent {
  event_type: EventType;
  severity: number;               // 0.0–1.0
  payload: MissionEventPayload;
}

// ---------------------------------------------------------------------------
// Full request bodies
// ---------------------------------------------------------------------------

export interface PlanRequest {
  rover_state: RoverState;
  env_state: EnvState;
}

export interface ReplanRequest {
  current_plan: MissionPlan;
  event: MissionEvent;
  rover_state: RoverState;
  env_state: EnvState;
}

// ---------------------------------------------------------------------------
// API responses / errors
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  backend: string;
  version: string;
}

/** Detail block returned by the API on error responses (4xx / 5xx). */
export interface ApiError {
  error: string;
  message: string;
  violations?: string[];
  attempts?: number;
}

export type ApiStatus = 'unknown' | 'online' | 'offline';

// ---------------------------------------------------------------------------
// UI-only types (not sent to the API)
// ---------------------------------------------------------------------------

/** A single entry in the browser-local event log. */
export interface EventLogEntry {
  id: string;             // unique key for React rendering
  timestamp: string;      // display string, e.g. "14:02:11"
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
}

// ---------------------------------------------------------------------------
// Display helper
// ---------------------------------------------------------------------------

/** Convert a battery fraction to a display percentage string. */
export function toPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}
