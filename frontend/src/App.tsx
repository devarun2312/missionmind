import { useState, useCallback, useEffect } from 'react';
import './styles/global.css';
import './App.css';

// Components
import Header from './components/Header/Header';
import MarsMap from './components/MarsMap/MarsMap';
import RoverStatus from './components/RoverStatus/RoverStatus';
import MissionDecision from './components/MissionDecision/MissionDecision';
import RouteList from './components/RouteList/RouteList';
import EventControls from './components/EventControls/EventControls';
import EventLog from './components/EventLog/EventLog';
import ErrorBanner from './components/shared/ErrorBanner';

// API
import { checkHealth, planMission, replanMission } from './api/missionApi';

// Types & data
import type {
  MissionPlan,
  MissionEvent,
  ApiStatus,
  EventLogEntry,
} from './types/mission';
import {
  INITIAL_ROVER_STATE,
  INITIAL_ENV_STATE,
} from './data/demoState';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let _logIdCounter = 0;
function makeLogEntry(
  message: string,
  type: EventLogEntry['type'] = 'info',
): EventLogEntry {
  const now = new Date();
  const timestamp = now.toTimeString().slice(0, 8);
  return { id: String(++_logIdCounter), timestamp, message, type };
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
let _healthChecked = false;
export default function App() {
  // ── State ──────────────────────────────────────────────────────────────
  // NOTE: setRoverState, setCurrentPlan, setIsPlanning, setError are used
  // in Phase 5 / 6 handlers. Prefixed with _ to satisfy noUnusedLocals
  // until those phases are wired.
  const [roverState, _setRoverState] = useState(INITIAL_ROVER_STATE);
  const [envState, setEnvState] = useState(INITIAL_ENV_STATE);
  const [currentPlan, _setCurrentPlan] = useState<MissionPlan | null>(null);
  const [apiStatus, setApiStatus] = useState<ApiStatus>('unknown');
  const [apiVersion, setApiVersion] = useState<string | undefined>(undefined);
  const [isPlanning, _setIsPlanning] = useState(false);
  const [error, _setError] = useState<string | null>(null);
  const [eventLog, setEventLog] = useState<EventLogEntry[]>([
    makeLogEntry('Mission control initialized', 'info'),
  ]);
  const [hazardIds, setHazardIds] = useState<Set<string>>(new Set());
  const [discoveryId, setDiscoveryId] = useState<string | null>(null);

  // ── Event log helper ────────────────────────────────────────────────────
  const addLog = useCallback((message: string, type: EventLogEntry['type'] = 'info') => {
    setEventLog(prev => [...prev, makeLogEntry(message, type)]);
  }, []);

  // ── Health check on mount ───────────────────────────────────────────────
  useEffect(() => {
    if (_healthChecked) return;
    _healthChecked = true;
    addLog('Checking MissionMind API connection…', 'info');
    checkHealth()
      .then(health => {
        setApiStatus('online');
        setApiVersion(health.version);
        addLog(`Backend connected — MissionMind v${health.version}`, 'success');
      })
      .catch(() => {
        setApiStatus('offline');
        addLog('API offline — start uvicorn to enable mission planning', 'warning');
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Stubs — wired in Phase 5 / 6 ────────────────────────────────────────
  const handleStartMission = useCallback(async () => {
    if (isPlanning) return;

    _setError(null);
    _setIsPlanning(true);

    addLog('Mission start requested', 'info');
    addLog(
      'MissionMind evaluating science, resources, and safety...',
      'info',
    );

    try {
      const plan = await planMission({
        rover_state: roverState,
        env_state: envState,
      });

      _setCurrentPlan(plan);

      addLog(
        `Mission plan accepted — confidence ${Math.round(plan.confidence * 100)}%`,
        'success',
      );
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'MissionMind encountered an unexpected planning error.';

      _setError(message);
      addLog(`Mission planning failed — ${message}`, 'error');
    } finally {
      _setIsPlanning(false);
    }
  }, [addLog, envState, isPlanning, roverState]);

  const handleEvent = useCallback(async (eventType: string) => {
    if (isPlanning || !currentPlan) return;

    let event: MissionEvent;

    switch (eventType) {
      case 'BATTERY_FAILURE':
        event = {
          event_type: 'BATTERY_FAILURE',
          severity: 1.0,
          payload: {
            battery_pct: 0.07,
          },
        };
        break;

      case 'TERRAIN_HAZARD':
        event = {
          event_type: 'TERRAIN_HAZARD',
          severity: 0.8,
          payload: {
            waypoint_id: 'wp-crater-ridge',
          },
        };
        break;

      case 'NEW_DISCOVERY':
        event = {
          event_type: 'NEW_DISCOVERY',
          severity: 0.6,
          payload: {
            id: 'wp-subsurface-lake',
            x: 90,
            y: 200,
            label: 'Subsurface Lake',
            scientific_value: 0.95,
            terrain_risk: 0.12,
            estimated_travel_time_minutes: 22,
            estimated_energy_wh: 11.0,
          },
        };
        break;

      case 'RETURN_TO_BASE':
        event = {
          event_type: 'RETURN_TO_BASE',
          severity: 1.0,
          payload: {},
        };
        break;

      case 'COMM_LOSS':
        event = {
          event_type: 'COMM_LOSS',
          severity: 0.7,
          payload: {
            safe_comm_radius_m: 150.0,
          },
        };
        break;

      default:
        addLog(`Unknown mission event: ${eventType}`, 'error');
        return;
    }

    _setError(null);
    _setIsPlanning(true);

    addLog(`Mission event detected — ${event.event_type}`, 'warning');
    addLog('MissionMind replanning mission...', 'info');

    try {
      const nextRoverState =
        event.event_type === 'BATTERY_FAILURE'
          ? {
            ...roverState,
            battery_pct: event.payload.battery_pct ?? roverState.battery_pct,
          }
          : roverState;

      const plan = await replanMission({
        current_plan: currentPlan,
        event,
        rover_state: nextRoverState,
        env_state: envState,
      });

      _setCurrentPlan(plan);

      if (event.event_type === 'BATTERY_FAILURE') {
        _setRoverState(nextRoverState);
      }

      if (event.event_type === 'TERRAIN_HAZARD') {
        setHazardIds(prev => {
          const next = new Set(prev);
          next.add('wp-crater-ridge');
          return next;
        });
      }

      if (event.event_type === 'NEW_DISCOVERY') {
        setDiscoveryId('wp-subsurface-lake');

        setEnvState(prev => ({
          ...prev,
          candidate_waypoints: prev.candidate_waypoints.some(
            waypoint => waypoint.id === 'wp-subsurface-lake',
          )
            ? prev.candidate_waypoints
            : [
              ...prev.candidate_waypoints,
              {
                id: 'wp-subsurface-lake',
                x: 90,
                y: 200,
                label: 'Subsurface Lake',
                scientific_value: 0.95,
                terrain_risk: 0.12,
                estimated_travel_time_minutes: 22,
                estimated_energy_wh: 11.0,
                is_base: false,
              },
            ],
        }));
      }

      addLog(
        `Mission replanned — ${plan.status}, confidence ${Math.round(plan.confidence * 100)}%`,
        'success',
      );
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'MissionMind encountered an unexpected replanning error.';

      _setError(message);
      addLog(`Mission replanning failed — ${message}`, 'error');
    } finally {
      _setIsPlanning(false);
    }
  }, [
    addLog,
    currentPlan,
    envState,
    isPlanning,
    roverState,
  ]);

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="app">
      {/* Header */}
      <Header apiStatus={apiStatus} version={apiVersion} />

      {/* Global error strip */}
      {error && (
        <div className="app__error-strip">
          <ErrorBanner message={error} onDismiss={() => _setError(null)} />
        </div>
      )}

      {/* Main workspace */}
      <div className="app__workspace">
        {/* Left: Mars map */}
        <div className="app__left">
          <div className="app__map-wrap">
            <MarsMap
              waypoints={envState.candidate_waypoints}
              currentPlan={currentPlan}
              roverX={roverState.position_x}
              roverY={roverState.position_y}
              hazardIds={hazardIds}
              discoveryId={discoveryId}
            />
          </div>
        </div>

        {/* Right: status panels */}
        <div className="app__right">
          <div className="panel">
            <div className="panel__header">
              <span className="panel__icon">🛸</span>
              <span className="panel__title">Rover Status</span>
            </div>
            <div className="panel__body">
              <RoverStatus
                roverState={roverState}
                currentPlan={currentPlan}
                isBusy={isPlanning}
              />
            </div>
          </div>

          <div className="panel">
            <div className="panel__header">
              <span className="panel__icon">🤖</span>
              <span className="panel__title">AI Mission Decision</span>
            </div>
            <div className="panel__body">
              <MissionDecision
                currentPlan={currentPlan}
                isBusy={isPlanning}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Bottom strip */}
      <div className="app__bottom">
        {/* Current Route */}
        <div className="panel" style={{ overflow: 'auto' }}>
          <div className="panel__header">
            <span className="panel__icon">📍</span>
            <span className="panel__title">Current Route</span>
          </div>
          <div className="panel__body" style={{ padding: 0 }}>
            <RouteList currentPlan={currentPlan} />
          </div>
        </div>

        {/* Mission Events */}
        <div className="panel">
          <div className="panel__header">
            <span className="panel__icon">⚡</span>
            <span className="panel__title">Mission Events</span>
          </div>
          <div className="panel__body">
            <EventControls
              currentPlan={currentPlan}
              isBusy={isPlanning}
              onStartMission={handleStartMission}
              onEvent={handleEvent}
            />
          </div>
        </div>

        {/* Event Log */}
        <div className="panel" style={{ overflow: 'hidden' }}>
          <div className="panel__header">
            <span className="panel__icon">📋</span>
            <span className="panel__title">Event Log</span>
          </div>
          <div className="panel__body" style={{ padding: 0 }}>
            <EventLog entries={eventLog} />
          </div>
        </div>
      </div>
    </div>
  );
}
