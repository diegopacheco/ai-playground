import { useEffect, useRef, useState } from "react";
import { roleColor } from "../../../components/Person";
import { formatMoney } from "../../../money";
import { addDays, formatDay, formatTime, minutesSinceMidnight, sameDay, weekDays } from "../../../time";
import type { User } from "../../people/types";
import { layoutDay } from "../layout";
import type { Meeting } from "../types";

const START_HOUR = 7;
const END_HOUR = 21;
const HOUR_PX = 56;
const SNAP_MINUTES = 30;

type Props = {
  readonly weekStart: Date;
  readonly meetings: readonly Meeting[];
  readonly users: ReadonlyMap<string, User>;
  readonly onSlot: (start: Date) => void;
  readonly onOpen: (m: Meeting) => void;
};

function top(d: Date): number {
  return ((minutesSinceMidnight(d) - START_HOUR * 60) / 60) * HOUR_PX;
}

function useNow(): Date {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 60000);
    return () => window.clearInterval(id);
  }, []);
  return now;
}

export function WeekView({ weekStart, meetings, users, onSlot, onOpen }: Props) {
  const now = useNow();
  const days = weekDays(weekStart);
  const todayIndex = days.findIndex((d) => sameDay(d, now));
  const [picked, setPicked] = useState<number | undefined>(undefined);
  const focus = picked ?? Math.max(todayIndex, 0);
  const scroller = useRef<HTMLDivElement>(null);
  const hours = Array.from({ length: END_HOUR - START_HOUR }, (_, i) => START_HOUR + i);

  useEffect(() => {
    if (scroller.current) scroller.current.scrollTop = HOUR_PX * 1.5;
  }, []);

  const onColumnClick = (day: Date, e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget) return;
    const y = e.clientY - e.currentTarget.getBoundingClientRect().top;
    const minutes = START_HOUR * 60 + Math.floor(((y / HOUR_PX) * 60) / SNAP_MINUTES) * SNAP_MINUTES;
    onSlot(new Date(day.getFullYear(), day.getMonth(), day.getDate(), Math.floor(minutes / 60), minutes % 60));
  };

  return (
    <div className="week">
      <div className="week-head">
        <div className="gutter" />
        {days.map((d, i) => (
          <button
            key={d.toISOString()}
            type="button"
            className={`day-head ${i === todayIndex ? "is-today" : ""} ${i === focus ? "is-focus" : ""}`}
            onClick={() => setPicked(i)}
          >
            {formatDay(d)}
          </button>
        ))}
      </div>
      <div className="week-scroll" ref={scroller}>
        <div className="week-grid" style={{ height: hours.length * HOUR_PX }}>
          <div className="gutter">
            {hours.map((h) => (
              <span key={h} style={{ top: (h - START_HOUR) * HOUR_PX }}>
                {formatTime(new Date(2000, 0, 1, h))}
              </span>
            ))}
          </div>
          {days.map((day, i) => {
            const next = addDays(day, 1);
            const placed = layoutDay(meetings.filter((m) => m.start >= day && m.start < next));
            return (
              <div
                key={day.toISOString()}
                className={`day-col ${i === focus ? "is-focus" : ""}`}
                style={{ backgroundSize: `100% ${HOUR_PX / 2}px` }}
                onClick={(e) => onColumnClick(day, e)}
                data-day={i}
              >
                {placed.map(({ item: m, lane, lanes }) => {
                  const organizer = users.get(m.organizerId);
                  const height = Math.max(top(m.end) - top(m.start), 22);
                  return (
                    <button
                      key={m.id}
                      type="button"
                      className={`event ${height < 48 ? "event--short" : ""}`}
                      style={{
                        top: top(m.start),
                        height,
                        left: `calc(${(lane / lanes) * 100}% + 2px)`,
                        width: `calc(${100 / lanes}% - 4px)`,
                        borderLeftColor: organizer ? roleColor(organizer.role) : undefined,
                      }}
                      onClick={() => onOpen(m)}
                    >
                      <strong>{m.title}</strong>
                      <span>
                        {formatTime(m.start)} - {formatTime(m.end)}
                      </span>
                      <span className="event-cost">{formatMoney(m.totalCost)}</span>
                    </button>
                  );
                })}
                {i === todayIndex && minutesSinceMidnight(now) >= START_HOUR * 60 && minutesSinceMidnight(now) < END_HOUR * 60 && (
                  <div className="now-line" style={{ top: top(now) }} />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
