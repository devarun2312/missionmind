import './RouteList.css';
import type { MissionPlan, Waypoint } from '../../types/mission';

interface Props {
  currentPlan: MissionPlan | null;
}

function dotClass(wp: Waypoint): string {
  if (wp.is_base) return 'route-list__dot--base';
  return 'route-list__dot--science';
}

export default function RouteList({ currentPlan }: Props) {
  if (!currentPlan || currentPlan.waypoints.length === 0) {
    return (
      <div className="empty-state" style={{ padding: '16px 12px' }}>
        <div className="empty-state__icon">📍</div>
        <div className="empty-state__title">No Active Route</div>
        <div>Start the mission to generate a waypoint route.</div>
      </div>
    );
  }

  return (
    <div className="route-list">
      {currentPlan.waypoints.map((wp, i) => (
        <div key={wp.id} className="route-list__item">
          <div className={`route-list__order ${wp.is_base ? 'route-list__order--base' : ''}`}>
            {i + 1}
          </div>
          <div className={`route-list__dot ${dotClass(wp)}`} />
          <div className="route-list__info">
            <span className={`route-list__label ${wp.is_base ? 'route-list__label--base' : ''}`}>
              {wp.label || wp.id}
            </span>
            <span className="route-list__meta">
              ({wp.x.toFixed(0)}, {wp.y.toFixed(0)}) m
              {!wp.is_base && ` · ${wp.estimated_travel_time_minutes.toFixed(0)} min · ${wp.estimated_energy_wh.toFixed(1)} Wh`}
            </span>
          </div>
          {!wp.is_base && wp.scientific_value > 0 && (
            <span className="route-list__sv">
              ★ {(wp.scientific_value * 100).toFixed(0)}%
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
