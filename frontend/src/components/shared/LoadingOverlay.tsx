import './LoadingOverlay.css';

interface Props {
  title: string;
  sub?: string;
}

export default function LoadingOverlay({ title, sub }: Props) {
  return (
    <div className="loading-overlay">
      <div className="loading-overlay__spinner" />
      <div className="loading-overlay__title">{title}</div>
      {sub && <div className="loading-overlay__sub">{sub}</div>}
    </div>
  );
}
