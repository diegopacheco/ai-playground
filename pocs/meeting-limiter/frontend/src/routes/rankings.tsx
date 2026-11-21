import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { WeekNav } from "../components/WeekNav";
import { useMeetingDialogs } from "../features/meetings/components/dialogs";
import { CostliestMeetings, SpenderRanking } from "../features/rankings/components/RankingTables";
import { useRankings } from "../features/rankings/hooks";
import { periodFor } from "../features/rankings/range";
import { RANGES, rangeLabel, type Range } from "../features/rankings/types";
import { Icon } from "../shell/icons";
import { Panel } from "../shell/panels";
import { useWeek } from "../shell/week";

type RankingSearch = { readonly range?: Range };

function parseSearch(search: Record<string, unknown>): RankingSearch {
  const range = RANGES.find((r) => r === search.range);
  return range ? { range } : {};
}

export const Route = createFileRoute("/rankings")({ validateSearch: parseSearch, component: RankingsPage });

function RankingsPage() {
  const { range = "week" } = Route.useSearch();
  const navigate = useNavigate({ from: "/rankings" });
  const week = useWeek();
  const rankings = useRankings(periodFor(range, week.weekStart));
  const dialogs = useMeetingDialogs();
  const data = rankings.data;
  const toolbar = (
    <>
      <div className="segmented" role="tablist" aria-label="Period">
        {RANGES.map((r) => (
          <button key={r} type="button" role="tab" aria-selected={r === range} className={r === range ? "on" : ""} onClick={() => navigate({ search: (prev) => ({ ...prev, range: r }) })}>
            {rangeLabel(r)}
          </button>
        ))}
      </div>
      {range !== "all" && <WeekNav />}
    </>
  );
  return (
    <div className="page page--stack">
      <Panel id="rank-meetings" title="Most expensive meetings" icon={<Icon name="coin" />} actions={toolbar}>
        <CostliestMeetings meetings={data?.meetings ?? []} onOpen={dialogs.openDetails} />
      </Panel>
      <div className="row">
        <Panel id="rank-money" title="Who spends the most money" icon={<Icon name="wallet" />} className="grow">
          <SpenderRanking spenders={data?.spenders ?? []} metric="money" />
        </Panel>
        <Panel id="rank-time" title="Who spends the most time" icon={<Icon name="clock" />} className="grow">
          <SpenderRanking spenders={data?.timeSinks ?? []} metric="time" />
        </Panel>
      </div>
    </div>
  );
}
