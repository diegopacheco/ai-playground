import { columnHelper, DataTable } from "../../../components/DataTable";
import { Meter } from "../../../components/Meter";
import { PersonCell, RoleBadge } from "../../../components/Person";
import { formatMinutes, formatMoney } from "../../../money";
import { formatWhen } from "../../../time";
import type { Meeting } from "../../meetings/types";
import { durationMinutes } from "../../meetings/types";
import type { Spender } from "../types";

function Rank({ n }: { readonly n: number }) {
  return <span className={`rank rank--${n <= 3 ? n : "rest"}`}>{n}</span>;
}

const meetingCol = columnHelper<Meeting>();

function meetingColumns(max: number) {
  return meetingCol.columns([
    meetingCol.display({ id: "rank", header: "#", cell: ({ row }) => <Rank n={row.index + 1} /> }),
    meetingCol.display({
      id: "meeting",
      header: "Meeting",
      cell: ({ row }) => (
        <span className="stack">
          <strong>{row.original.title}</strong>
          <small>{formatWhen(row.original.start, row.original.end)}</small>
        </span>
      ),
    }),
    meetingCol.display({ id: "people", header: "People", cell: ({ row }) => row.original.charges.length }),
    meetingCol.display({ id: "length", header: "Length", cell: ({ row }) => formatMinutes(durationMinutes(row.original)) }),
    meetingCol.display({
      id: "cost",
      header: "Cost",
      cell: ({ row }) => (
        <span className="meter-cell">
          <Meter value={row.original.totalCost} max={max} tone="neutral" label="cost" />
          <span className="money">{formatMoney(row.original.totalCost)}</span>
        </span>
      ),
    }),
  ]);
}

export function CostliestMeetings({ meetings, onOpen }: { readonly meetings: readonly Meeting[]; readonly onOpen: (m: Meeting) => void }) {
  const max = meetings[0]?.totalCost ?? 0;
  return <DataTable columns={meetingColumns(max)} data={meetings} rowKey={(m) => m.id} onRowClick={onOpen} empty="No meetings in this period." />;
}

const spenderCol = columnHelper<Spender>();

type Metric = "money" | "time";

function spenderColumns(metric: Metric, max: number) {
  const value = (s: Spender) => (metric === "money" ? s.spent : s.minutes);
  return spenderCol.columns([
    spenderCol.display({ id: "rank", header: "#", cell: ({ row }) => <Rank n={row.index + 1} /> }),
    spenderCol.display({ id: "person", header: "Person", cell: ({ row }) => <PersonCell user={row.original.user} /> }),
    spenderCol.display({ id: "role", header: "Role", cell: ({ row }) => <RoleBadge role={row.original.user.role} /> }),
    spenderCol.display({ id: "meetings", header: "Meetings", cell: ({ row }) => row.original.meetings }),
    spenderCol.display({
      id: "value",
      header: metric === "money" ? "Money spent" : "Time in meetings",
      cell: ({ row }) => (
        <span className="meter-cell">
          <Meter value={value(row.original)} max={max} tone="neutral" label={metric} />
          <span className="money">{metric === "money" ? formatMoney(row.original.spent) : formatMinutes(row.original.minutes)}</span>
        </span>
      ),
    }),
  ]);
}

export function SpenderRanking({ spenders, metric }: { readonly spenders: readonly Spender[]; readonly metric: Metric }) {
  const first = spenders[0];
  const max = first ? (metric === "money" ? first.spent : first.minutes) : 0;
  return <DataTable columns={spenderColumns(metric, max)} data={spenders} rowKey={(s) => s.user.id} empty="Nobody spent anything in this period." />;
}
