import './Header.css';
import ApiStatusDot from '../shared/ApiStatusDot';
import type { ApiStatus } from '../../types/mission';

interface Props {
  apiStatus: ApiStatus;
  version?: string;
}

export default function Header({ apiStatus, version }: Props) {
  const missionId = 'MSN-2025-ARES-01';

  return (
    <header className="header">
      {/* Brand left */}
      <div className="header__brand">
        {/* Inline SVG logo-mark: stylised rover silhouette */}
        <svg
          className="header__logo-mark"
          viewBox="0 0 28 28"
          fill="none"
          aria-hidden="true"
        >
          {/* planet circle */}
          <circle cx="14" cy="14" r="11" stroke="#c84b17" strokeWidth="1.5" />
          {/* rover body */}
          <rect x="9" y="12" width="10" height="5" rx="1" fill="#e07a3a" opacity="0.9" />
          {/* solar panel */}
          <rect x="6" y="10" width="4" height="2" rx="0.5" fill="#58a6ff" opacity="0.8" />
          <rect x="18" y="10" width="4" height="2" rx="0.5" fill="#58a6ff" opacity="0.8" />
          {/* antenna */}
          <line x1="14" y1="12" x2="14" y2="9" stroke="#c9d1d9" strokeWidth="1" />
          <circle cx="14" cy="8.5" r="1" fill="#39d353" />
          {/* wheels */}
          <circle cx="11" cy="17.5" r="1.5" fill="#4a5568" />
          <circle cx="17" cy="17.5" r="1.5" fill="#4a5568" />
        </svg>

        <div>
          <div className="header__name">MissionMind</div>
          <div className="header__tagline">Autonomous Mars Mission Commander</div>
        </div>
      </div>

      {/* Centre mission ID */}
      <div className="header__mission-id">{missionId}</div>

      {/* Right controls */}
      <div className="header__right">
        {version && (
          <span className="header__version">v{version}</span>
        )}
        <ApiStatusDot status={apiStatus} />
      </div>
    </header>
  );
}
