import { memo } from "react";

function StatBadge({ label = "Revenue", value = 0, unit = "", delta = 0 }) {
  const up = delta >= 0;
  return (
    <div className="stat-badge">
      <span className="stat-badge__label">{label}</span>
      <span className="stat-badge__value">
        {unit === "$" ? "$" : ""}
        {value.toLocaleString()}
        {unit !== "$" ? unit : ""}
      </span>
      <span className={up ? "stat-badge__delta up" : "stat-badge__delta down"}>
        {up ? "▲" : "▼"} {Math.abs(delta)}%
      </span>
    </div>
  );
}

export default memo(StatBadge);
