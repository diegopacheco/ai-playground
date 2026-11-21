import { Icon } from "../shell/icons";
import { useWeek } from "../shell/week";
import { formatWeekRange } from "../time";

export function WeekNav() {
  const week = useWeek();
  return (
    <div className="week-nav">
      <button type="button" className="button" data-tip="This week" data-keys="this-week" onClick={week.thisWeek}>
        <Icon name="today" />
        <span>Today</span>
      </button>
      <button type="button" className="icon-button" data-tip="Previous week" data-keys="prev-week" aria-label="Previous week" onClick={week.prev}>
        <Icon name="chevronLeft" />
      </button>
      <button type="button" className="icon-button" data-tip="Next week" data-keys="next-week" aria-label="Next week" onClick={week.next}>
        <Icon name="chevronRight" />
      </button>
      <span className="week-range">{formatWeekRange(week.weekStart)}</span>
    </div>
  );
}
