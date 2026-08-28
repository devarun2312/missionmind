/**
 * MissionMind API client.
 *
 * All communication with the FastAPI backend goes through this module.
 * Uses native browser fetch — no Axios.
 *
 * Base URL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
 *
 * IBM watsonx credentials must NEVER appear here or anywhere in the frontend.
 */

import type {
  HealthResponse,
  MissionPlan,
  PlanRequest,
  ReplanRequest,
  ApiError,
} from '../types/mission';

// ---------------------------------------------------------------------------
// Base URL
// ---------------------------------------------------------------------------

const API_BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8000';

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Parse an API error response into a human-readable message. */
async function parseErrorMessage(resp: Response): Promise<string> {
  try {
    const body = await resp.json() as { detail?: ApiError | string | unknown };
    const detail = body.detail;
    if (detail && typeof detail === 'object' && 'message' in detail) {
      const typed = detail as ApiError;
      let msg = typed.message || typed.error || `HTTP ${resp.status}`;
      if (typed.violations?.length) {
        msg += ` — Violations: ${typed.violations.join('; ')}`;
      }
      if (typed.attempts !== undefined) {
        msg += ` (${typed.attempts} attempts)`;
      }
      return msg;
    }
    if (typeof detail === 'string') return detail;
    return `HTTP ${resp.status}: ${resp.statusText}`;
  } catch {
    return `HTTP ${resp.status}: ${resp.statusText}`;
  }
}

/** Typed fetch wrapper — throws a user-readable Error on failure. */
async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      ...init,
    });
  } catch (err) {
    // Network-level failure (backend offline, CORS preflight failed, etc.)
    throw new Error(`Cannot reach MissionMind API at ${API_BASE}. Is uvicorn running?`);
  }

  if (!resp.ok) {
    const message = await parseErrorMessage(resp);
    throw new Error(message);
  }

  return resp.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/**
 * GET /api/health
 *
 * Returns a HealthResponse or throws if the backend is unreachable.
 * Used for the connection indicator in the header.
 */
export async function checkHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/api/health');
}

/**
 * POST /api/mission/plan
 *
 * Run the full MissionMind AI planning pipeline.
 * Returns a MissionPlan with status=ACTIVE on success.
 *
 * NOTE: battery_pct in rover_state must be a FRACTION (0.68 = 68%).
 */
export async function planMission(request: PlanRequest): Promise<MissionPlan> {
  return apiFetch<MissionPlan>('/api/mission/plan', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * POST /api/mission/replan
 *
 * Replan the mission in response to a mid-mission event.
 * Returns a revised MissionPlan with status=ACTIVE on success.
 *
 * NOTE: battery_pct in rover_state must be a FRACTION (0.68 = 68%).
 */
export async function replanMission(request: ReplanRequest): Promise<MissionPlan> {
  return apiFetch<MissionPlan>('/api/mission/replan', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}
