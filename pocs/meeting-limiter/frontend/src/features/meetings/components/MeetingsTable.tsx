import { columnHelper, DataTable } from "../../../components/DataTable";
import { formatMinutes, formatMoney } from "../../../money";
import { formatWhen } from "../../../time";
import type { User } from "../../people/types";
import { durationMinutes, type Meeting } from "../types";

const col = columnHelper<Meeting>();

function buildColumns(users: ReadonlyMap<string, User>) {
  return col.columns([
    col.display({ id: "title", header: "Meeting", cell: ({ row }) => <strong>{row.original.title}</strong> }),
    col.display({ id: "when", header: "When", cell: ({ row }) => formatWhen(row.original.start, row.original.end) }),
    col.display({ id: "length", header: "Length", cell: ({ row }) => formatMinutes(durationMinutes(row.original)) }),
    col.display({ id: "organizer", header: "Organizer", cell: ({ row }) => users.get(row.original.organizerId)?.name ?? row.original.organizerId }),
    col.display({ id: "people", header: "People", cell: ({ row }) => row.original.charges.length }),
    col.display({ id: "cost", header: "Cost", cell: ({ row }) => <span className="money">{formatMoney(row.original.totalCost)}</span> }),
  ]);
}

type Props = {
  readonly meetings: readonly Meeting[];
  readonly users: ReadonlyMap<string, User>;
  readonly onOpen: (m: Meeting) => void;
};

export function MeetingsTable({ meetings, users, onOpen }: Props) {
  return <DataTable columns={buildColumns(users)} data={meetings} rowKey={(m) => m.id} onRowClick={onOpen} empty="No meetings this week. Your wallet thanks you." />;
}
