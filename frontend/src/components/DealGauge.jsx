const LOCALE_BY_SYMBOL = { "$": "en-US", "₹": "en-IN" };

function fmtMoney(n, symbol = "$") {
  if (n === null || n === undefined) return "—";
  const locale = LOCALE_BY_SYMBOL[symbol] || "en-US";
  return `${symbol}${Number(n).toLocaleString(locale, { maximumFractionDigits: 0 })}`;
}

export default function DealGauge({ floor, target, currentOffer, currency = "$" }) {
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
        <span className="gauge-label-floor">FLOOR · {fmtMoney(floor, currency)}</span>
        <span className="gauge-label-target">TARGET · {fmtMoney(target, currency)}</span>
      </div>
      <div className="gauge-track">
        <div className="gauge-fill" style={{ width: `${pct}%` }} />
        <div className={markerClass} style={{ left: `${pct}%` }} />
      </div>
      <div className="gauge-current">{fmtMoney(currentOffer, currency)}</div>
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
