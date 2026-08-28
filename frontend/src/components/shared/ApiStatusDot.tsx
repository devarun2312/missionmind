import './ApiStatusDot.css';
import type { ApiStatus } from '../../types/mission';

interface Props {
  status: ApiStatus;
}

const LABELS: Record<ApiStatus, string> = {
  online:  'API Connected',
  offline: 'API Offline',
  unknown: 'Connecting…',
};

export default function ApiStatusDot({ status }: Props) {
  return (
    <div className="api-status">
      <div className={`api-status__dot api-status__dot--${status}`} />
      <span>{LABELS[status]}</span>
    </div>
  );
}
