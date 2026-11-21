import { createFileRoute } from "@tanstack/react-router";
import { WeekNav } from "../components/WeekNav";
import { useMeetingDialogs } from "../features/meetings/components/dialogs";
import { MeetingsTable } from "../features/meetings/components/MeetingsTable";
import { useWeekMeetings } from "../features/meetings/hooks";
import { useUserMap } from "../features/people/hooks";
import { formatMoney } from "../money";
import { Icon } from "../shell/icons";
import { Panel } from "../shell/panels";
import { useWeek } from "../shell/week";

export const Route = createFileRoute("/meetings")({ component: MeetingsPage });

function MeetingsPage() {
  const week = useWeek();
  const meetings = useWeekMeetings(week.weekStart);
  const users = useUserMap();
  const dialogs = useMeetingDialogs();
  const list = meetings.data ?? [];
  const total = list.reduce((sum, m) => sum + m.totalCost, 0);
  return (
    <div className="page">
      <Panel
        id="meetings"
        title={`Meetings - ${list.length} this week, ${formatMoney(total)} spent`}
        icon={<Icon name="list" />}
        className="grow"
        actions={<WeekNav />}
      >
        <MeetingsTable meetings={list} users={users} onOpen={dialogs.openDetails} />
      </Panel>
    </div>
  );
}
