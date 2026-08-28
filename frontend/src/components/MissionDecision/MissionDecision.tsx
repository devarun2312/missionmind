import './MissionDecision.css';
import type { MissionPlan } from '../../types/mission';

interface Props {
  currentPlan: MissionPlan | null;
  isBusy: boolean;
}

export default function MissionDecision({ currentPlan, isBusy }: Props) {
  if (isBusy) {
    return (
      <div className="mission-decision">
        <div className="empty-state">
          <div className="empty-state__icon">⏳</div>
          <div className="empty-state__title">MissionMind is planning…</div>
          <div>Evaluating science, resources, and safety constraints.</div>
        </div>
      </div>
    );
  }

  if (!currentPlan) {
    return (
      <div className="mission-decision">
        <div className="empty-state">
          <div className="empty-state__icon">🛰</div>
          <div className="empty-state__title">Awaiting Mission Plan</div>
          <div>
            MissionMind will balance scientific value, energy budget, and
            safety constraints to generate an optimal route.
          </div>
        </div>
      </div>
    );
  }

  const confidencePct = Math.round(currentPlan.confidence * 100);
  const scienceWaypoints = currentPlan.waypoints.filter(w => !w.is_base);

  return (
    <div className="mission-decision">
      {/* Confidence bar */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span className="text-muted" style={{ fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            AI Confidence
          </span>
          <span
            className="text-mono"
            style={{ fontSize: 13, fontWeight: 600, color: confidencePct >= 75 ? 'var(--accent-green)' : 'var(--mars-amber)' }}
          >
            {confidencePct}%
          </span>
        </div>
        <div className="confidence-bar">
          <div className="confidence-bar__fill" style={{ width: `${confidencePct}%` }} />
        </div>
      </div>

      {/* Plan stats */}
      <div className="mission-decision__stats">
        <div className="mission-decision__stat">
          <span className="mission-decision__stat-label">Waypoints</span>
          <span className="mission-decision__stat-value">{scienceWaypoints.length}</span>
        </div>
        <div className="mission-decision__stat">
          <span className="mission-decision__stat-label">Energy</span>
          <span className="mission-decision__stat-value">
            {currentPlan.total_energy_wh.toFixed(0)} Wh
          </span>
        </div>
        <div className="mission-decision__stat">
          <span className="mission-decision__stat-label">Duration</span>
          <span className="mission-decision__stat-value">
            {currentPlan.total_time_minutes.toFixed(0)} min
          </span>
        </div>
        <div className="mission-decision__stat">
          <span className="mission-decision__stat-label">Status</span>
          <span
            className="mission-decision__stat-value"
            style={{ fontSize: 12, color: currentPlan.status === 'ACTIVE' ? 'var(--accent-green)' : 'var(--mars-amber)' }}
          >
            {currentPlan.status}
          </span>
        </div>
      </div>

      {/* Reasoning */}
      {currentPlan.reasoning && (
        <div className="mission-decision__reasoning">
          "{currentPlan.reasoning}"
        </div>
      )}
    </div>
  );
}
