/**
 * Demo initial state for MissionMind UI.
 *
 * These are the static simulation inputs the React app owns and sends to
 * the FastAPI backend. All values are backend-compatible — field names and
 * types match the Python Pydantic schemas exactly.
 *
 * BATTERY CONVENTION: battery_pct is a FRACTION. 0.68 = 68%.
 *
 * Candidate waypoints are chosen so that:
 *   - Terrain risks are well below the 0.70 hard limit
 *   - Total energy is within the usable budget:
 *     usable_wh = 0.68 × 500 − 0.20 × 500 = 340 − 100 = 240 Wh
 *   - At least one is_base=true waypoint exists
 */

import type { RoverState, EnvState, WaypointInput } from '../types/mission';

// ---------------------------------------------------------------------------
// Candidate waypoints
// ---------------------------------------------------------------------------

export const DEMO_WAYPOINTS: WaypointInput[] = [
  {
    id: 'wp-base',
    x: 0,
    y: 0,
    terrain_risk: 0.0,
    is_base: true,
    label: 'Mars Base',
    estimated_travel_time_minutes: 0,
    estimated_energy_wh: 0,
  },
  {
    id: 'wp-ice-deposit',
    x: 120,
    y: -80,
    terrain_risk: 0.10,
    is_base: false,
    label: 'Ice Deposit',
    estimated_travel_time_minutes: 28,
    estimated_energy_wh: 14.0,
  },
  {
    id: 'wp-ancient-rock',
    x: 210,
    y: 40,
    terrain_risk: 0.20,
    is_base: false,
    label: 'Ancient Rock Formation',
    estimated_travel_time_minutes: 45,
    estimated_energy_wh: 22.5,
  },
  {
    id: 'wp-crater-ridge',
    x: 160,
    y: 150,
    terrain_risk: 0.35,
    is_base: false,
    label: 'Crater Ridge',
    estimated_travel_time_minutes: 38,
    estimated_energy_wh: 19.0,
  },
];

/**
 * The "hidden" discovery waypoint injected by the NEW_DISCOVERY event.
 * Not included in the initial candidate list — added dynamically.
 */
export const DISCOVERY_WAYPOINT: WaypointInput = {
  id: 'wp-subsurface-lake',
  x: 90,
  y: 200,
  terrain_risk: 0.12,
  is_base: false,
  label: 'Subsurface Lake',
  estimated_travel_time_minutes: 22,
  estimated_energy_wh: 11.0,
  scientific_value: 0.95,
};

// ---------------------------------------------------------------------------
// Initial rover state
// ---------------------------------------------------------------------------

/** Starting rover state. battery_pct=0.68 (68%) gives a realistic planning scenario. */
export const INITIAL_ROVER_STATE: RoverState = {
  battery_pct: 0.68,          // FRACTION — 68%
  battery_capacity_wh: 500.0,
  position_x: 0.0,
  position_y: 0.0,
  rover_speed_mps: 0.5,
  power_consumption_w: 50.0,
};

// ---------------------------------------------------------------------------
// Initial environment state
// ---------------------------------------------------------------------------

export const INITIAL_ENV_STATE: EnvState = {
  candidate_waypoints: DEMO_WAYPOINTS,
  weather_forecast: {
    dust_storm_probability: 0.08,
    temperature_min_c: -55,
    temperature_max_c: 12,
    wind_speed_mps: 4.0,
    forecast_hours: 8,
  },
  comm_windows: [
    { start_utc: '2025-06-15T10:00:00Z', duration_minutes: 45 },
  ],
  terrain_map: {},
  mission_objectives: [
    'Search for biosignatures and water-ice deposits',
    'Analyse sedimentary rock formations for geological history',
  ],
};
