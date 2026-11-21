import { useState } from "react";
import { Icon } from "../../../shell/icons";
import { addDays, sameDay, startOfMonth, startOfWeek } from "../../../time";

const monthLabel = new Intl.DateTimeFormat("en-US", { month: "long", year: "numeric" });
const WEEKDAYS = ["M", "T", "W", "T", "F", "S", "S"] as const;

type Props = { readonly weekStart: Date; readonly onPick: (d: Date) => void };

export function MiniMonth({ weekStart, onPick }: Props) {
  const [offset, setOffset] = useState(0);
  const month = new Date(weekStart.getFullYear(), weekStart.getMonth() + offset, 1);
  const first = startOfWeek(startOfMonth(month));
  const cells = Array.from({ length: 42 }, (_, i) => addDays(first, i));
  const today = new Date();
  return (
    <div className="mini-month">
      <div className="mini-head">
        <strong>{monthLabel.format(month)}</strong>
        <button type="button" className="icon-button" aria-label="Previous month" data-tip="Previous month" onClick={() => setOffset(offset - 1)}>
          <Icon name="chevronLeft" size={16} />
        </button>
        <button type="button" className="icon-button" aria-label="Next month" data-tip="Next month" onClick={() => setOffset(offset + 1)}>
          <Icon name="chevronRight" size={16} />
        </button>
      </div>
      <div className="mini-grid">
        {WEEKDAYS.map((d, i) => (
          <span key={i} className="mini-dow">
            {d}
          </span>
        ))}
        {cells.map((d) => {
          const inWeek = d >= weekStart && d < addDays(weekStart, 7);
          const classes = ["mini-day", d.getMonth() === month.getMonth() ? "" : "other", inWeek ? "in-week" : "", sameDay(d, today) ? "today" : ""];
          return (
            <button key={d.toISOString()} type="button" className={classes.filter(Boolean).join(" ")} onClick={() => onPick(d)}>
              {d.getDate()}
            </button>
          );
        })}
      </div>
    </div>
  );
}
