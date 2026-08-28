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
import { checkHealth } from './api/missionApi';

// Types & data
import type {
  MissionPlan,
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

export default function App() {
  // ── State ──────────────────────────────────────────────────────────────
  // NOTE: setRoverState, setCurrentPlan, setIsPlanning, setError are used
  // in Phase 5 / 6 handlers. Prefixed with _ to satisfy noUnusedLocals
  // until those phases are wired.
  const [roverState, _setRoverState] = useState(INITIAL_ROVER_STATE);
  const [envState] = useState(INITIAL_ENV_STATE);
  const [currentPlan, _setCurrentPlan] = useState<MissionPlan | null>(null);
  const [apiStatus, setApiStatus] = useState<ApiStatus>('unknown');
  const [apiVersion, setApiVersion] = useState<string | undefined>(undefined);
  const [isPlanning, _setIsPlanning] = useState(false);
  const [error, _setError] = useState<string | null>(null);
  const [eventLog, setEventLog] = useState<EventLogEntry[]>([
    makeLogEntry('Mission control initialized', 'info'),
  ]);
  const [hazardIds] = useState<Set<string>>(new Set());
  const [discoveryId] = useState<string | null>(null);

  // ── Event log helper ────────────────────────────────────────────────────
  const addLog = useCallback((message: string, type: EventLogEntry['type'] = 'info') => {
    setEventLog(prev => [...prev, makeLogEntry(message, type)]);
  }, []);

  // ── Health check on mount ───────────────────────────────────────────────
  useEffect(() => {
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
  const handleStartMission = useCallback(() => {
    addLog('START MISSION — planning will be wired in Phase 5', 'warning');
  }, [addLog]);

  const handleEvent = useCallback((eventType: string) => {
    addLog(`Event triggered: ${eventType} — replanning will be wired in Phase 6`, 'warning');
  }, [addLog]);

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
