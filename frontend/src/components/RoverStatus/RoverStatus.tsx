import './RoverStatus.css';
import type { RoverState, MissionPlan, MissionStatus } from '../../types/mission';
import { toPercent } from '../../types/mission';

interface Props {
  roverState: RoverState;
  currentPlan: MissionPlan | null;
  isBusy: boolean;
}

function batteryClass(frac: number): string {
  if (frac >= 0.5) return 'battery-bar__fill--high';
  if (frac >= 0.25) return 'battery-bar__fill--medium';
  return 'battery-bar__fill--low';
}

function batteryTextClass(frac: number): string {
  if (frac >= 0.5) return 'text-green';
  if (frac >= 0.25) return 'text-amber';
  return 'text-danger';
}

const STATUS_LABELS: Partial<Record<MissionStatus, string>> = {
  ACTIVE:      'Mission Active',
  PENDING:     'Mission Pending',
  REPLANNING:  'Replanning…',
  ABORTED:     'Mission Aborted',
  COMPLETE:    'Mission Complete',
};

function phaseLabel(plan: MissionPlan | null, busy: boolean): { label: string; cls: string } {
  if (busy)  return { label: 'Processing…',    cls: 'text-muted' };
  if (!plan) return { label: 'System Ready',   cls: 'text-muted' };
  const lbl = STATUS_LABELS[plan.status] ?? plan.status;
  const cls = plan.status === 'ACTIVE' ? 'text-green'
            : plan.status === 'REPLANNING' ? 'text-amber'
            : plan.status === 'ABORTED' ? 'text-danger'
            : 'text-muted';
  return { label: lbl, cls };
}

export default function RoverStatus({ roverState, currentPlan, isBusy }: Props) {
  const { battery_pct, battery_capacity_wh, position_x, position_y } = roverState;
  const phase = phaseLabel(currentPlan, isBusy);

  return (
    <div className="rover-status">
      {/* Battery */}
      <div className="stat-row">
        <span className="stat-row__label">Battery</span>
        <span className={`stat-row__value ${batteryTextClass(battery_pct)}`}>
          {toPercent(battery_pct)}
        </span>
      </div>
      <div className="battery-bar" style={{ margin: '4px 0 8px' }}>
        <div
          className={`battery-bar__fill ${batteryClass(battery_pct)}`}
          style={{ width: toPercent(battery_pct) }}
        />
      </div>

      {/* Capacity */}
      <div className="stat-row">
        <span className="stat-row__label">Capacity</span>
        <span className="stat-row__value">{battery_capacity_wh} Wh</span>
      </div>

      {/* Position */}
      <div className="stat-row">
        <span className="stat-row__label">Position</span>
        <span className="stat-row__value">
          ({position_x.toFixed(0)}, {position_y.toFixed(0)}) m
        </span>
      </div>

      {/* Communications */}
      <div className="stat-row">
        <span className="stat-row__label">Comms</span>
        <span className="stat-row__value text-green">Nominal</span>
      </div>

      {/* Speed */}
      <div className="stat-row">
        <span className="stat-row__label">Speed</span>
        <span className="stat-row__value">{roverState.rover_speed_mps} m/s</span>
      </div>

      {/* Plan energy if available */}
      {currentPlan && (
        <div className="stat-row">
          <span className="stat-row__label">Plan Energy</span>
          <span className="stat-row__value">
            {currentPlan.total_energy_wh.toFixed(1)} Wh
          </span>
        </div>
      )}

      {/* Plan time if available */}
      {currentPlan && (
        <div className="stat-row">
          <span className="stat-row__label">Plan Duration</span>
          <span className="stat-row__value">
            {currentPlan.total_time_minutes.toFixed(0)} min
          </span>
        </div>
      )}

      {/* Phase label */}
      <div className={`rover-status__phase ${phase.cls}`}>
        {phase.label}
      </div>
    </div>
  );
}
