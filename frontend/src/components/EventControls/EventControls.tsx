import './EventControls.css';
import type { MissionPlan } from '../../types/mission';

interface Props {
  currentPlan: MissionPlan | null;
  isBusy: boolean;
  /** Called when user clicks START MISSION */
  onStartMission: () => void;
  /** Called when user triggers an event button */
  onEvent: (eventType: string) => void;
}

export default function EventControls({
  currentPlan,
  isBusy,
  onStartMission,
  onEvent,
}: Props) {
  const hasPlan = currentPlan !== null;
  const disabled = isBusy;

  return (
    <div className="event-controls">
      {/* START MISSION — always shown, disabled while busy or replanning */}
      <button
        className="btn btn--primary event-controls__start"
        onClick={onStartMission}
        disabled={disabled}
        title={hasPlan ? 'Generate a new mission plan' : 'Start the mission planner'}
      >
        {isBusy ? '⏳ Planning…' : hasPlan ? '↺  Replan Mission' : '▶  Start Mission'}
      </button>

      {/* Event buttons — only enabled after a plan exists */}
      <div className="event-controls__grid">
        <button
          className="btn btn--danger"
          onClick={() => onEvent('BATTERY_FAILURE')}
          disabled={disabled || !hasPlan}
          title="Simulate battery failure"
        >
          🔋 Battery Failure
        </button>

        <button
          className="btn btn--warning"
          onClick={() => onEvent('TERRAIN_HAZARD')}
          disabled={disabled || !hasPlan}
          title="Simulate terrain hazard at Crater Ridge"
        >
          ⚠ Terrain Hazard
        </button>

        <button
          className="btn btn--violet"
          onClick={() => onEvent('NEW_DISCOVERY')}
          disabled={disabled || !hasPlan}
          title="Simulate new scientific discovery"
        >
          ✦ New Discovery
        </button>

        <button
          className="btn btn--danger"
          onClick={() => onEvent('RETURN_TO_BASE')}
          disabled={disabled || !hasPlan}
          title="Issue return-to-base command"
        >
          🏠 Return to Base
        </button>

        <button
          className="btn btn--info"
          onClick={() => onEvent('COMM_LOSS')}
          disabled={disabled || !hasPlan}
          title="Simulate communication loss"
        >
          📡 Comm Loss
        </button>
      </div>

      {!hasPlan && !isBusy && (
        <div className="event-controls__hint">
          Event controls will activate after a mission plan is generated.
        </div>
      )}
    </div>
  );
}
