import { useEffect, useRef } from 'react';
import './EventLog.css';
import type { EventLogEntry } from '../../types/mission';

interface Props {
  entries: EventLogEntry[];
}

export default function EventLog({ entries }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest entry
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries]);

  if (entries.length === 0) {
    return (
      <div className="event-log">
        <div className="empty-state" style={{ padding: '12px' }}>
          <div className="empty-state__title">No events yet</div>
        </div>
      </div>
    );
  }

  return (
    <div className="event-log">
      {entries.map(entry => (
        <div key={entry.id} className={`event-log__entry event-log__entry--${entry.type}`}>
          <span className="event-log__dot" />
          <span className="event-log__time">{entry.timestamp}</span>
          <span className="event-log__msg">{entry.message}</span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
