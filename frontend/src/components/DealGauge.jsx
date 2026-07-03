function fmtMoney(n) {
  if (n === null || n === undefined) return "—";
  return `$${Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export default function DealGauge({ floor, target, currentOffer }) {
  const range = Math.max(target - floor, 1e-6);
  const raw = (currentOffer - floor) / range;
  const clamped = Math.min(Math.max(raw, 0), 1);
  const pct = clamped * 100;

  const aboveTarget = currentOffer > target;
  const belowFloor = currentOffer < floor;

  let markerClass = "gauge-marker";
  if (aboveTarget) markerClass += " above-target";
  if (belowFloor) markerClass += " below-floor";

  return (
    <div className="gauge-wrap">
      <div className="gauge-labels">
        <span className="gauge-label-floor">FLOOR · {fmtMoney(floor)}</span>
        <span className="gauge-label-target">TARGET · {fmtMoney(target)}</span>
      </div>
      <div className="gauge-track">
        <div className="gauge-fill" style={{ width: `${pct}%` }} />
        <div className={markerClass} style={{ left: `${pct}%` }} />
      </div>
      <div className="gauge-current">{fmtMoney(currentOffer)}</div>
      <div className="gauge-current-caption">
        {aboveTarget
          ? "above target"
          : belowFloor
          ? "below floor"
          : `${Math.round(pct)}% of the way to target`}
      </div>
    </div>
  );
}
