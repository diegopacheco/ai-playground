import { createFileRoute } from "@tanstack/react-router";
import { useMeetingDialogs } from "../features/meetings/components/dialogs";
import { MiniMonth } from "../features/meetings/components/MiniMonth";
import { WeekView } from "../features/meetings/components/WeekView";
import { useWeekMeetings } from "../features/meetings/hooks";
import { useUserMap } from "../features/people/hooks";
import { MyWallet } from "../features/wallets/components/MyWallet";
import { Rates } from "../features/wallets/components/Rates";
import { useWallets } from "../features/wallets/hooks";
import { WeekNav } from "../components/WeekNav";
import { useActingUser } from "../shell/acting";
import { Icon } from "../shell/icons";
import { Panel } from "../shell/panels";
import { useWeek } from "../shell/week";

export const Route = createFileRoute("/")({ component: CalendarPage });

function CalendarPage() {
  const week = useWeek();
  const meetings = useWeekMeetings(week.weekStart);
  const wallets = useWallets(week.weekStart);
  const users = useUserMap();
  const dialogs = useMeetingDialogs();
  const { userId } = useActingUser();
  return (
    <div className="page page--calendar">
      <aside className="side">
        <Panel id="mini-month" title="Calendar" icon={<Icon name="calendar" />}>
          <MiniMonth weekStart={week.weekStart} onPick={week.goTo} />
        </Panel>
        <Panel id="my-wallet" title="My wallet this week" icon={<Icon name="wallet" />}>
          <MyWallet wallet={wallets.data?.find((w) => w.user.id === userId)} />
        </Panel>
        <Panel id="rates" title="Meeting rates" icon={<Icon name="coin" />}>
          <Rates />
        </Panel>
      </aside>
      <Panel
        id="week"
        title="Week"
        icon={<Icon name="today" />}
        className="grow"
        actions={
          <>
            <button type="button" className="button button--primary" data-tip="New meeting" data-keys="new-meeting" onClick={() => dialogs.openNew()}>
              <Icon name="plus" />
              <span>New meeting</span>
            </button>
            <WeekNav />
          </>
        }
      >
        {meetings.error && <p className="quote-warning">{meetings.error.message}</p>}
        <WeekView weekStart={week.weekStart} meetings={meetings.data ?? []} users={users} onSlot={dialogs.openNew} onOpen={dialogs.openDetails} />
      </Panel>
    </div>
  );
}
