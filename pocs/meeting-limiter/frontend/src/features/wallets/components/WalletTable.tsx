import { columnHelper, DataTable } from "../../../components/DataTable";
import { Meter } from "../../../components/Meter";
import { PersonCell, RoleBadge } from "../../../components/Person";
import { formatMinutes, formatMoney } from "../../../money";
import { walletHealth, type Wallet } from "../types";

const col = columnHelper<Wallet>();

const columns = col.columns([
  col.display({ id: "person", header: "Person", cell: ({ row }) => <PersonCell user={row.original.user} /> }),
  col.display({ id: "role", header: "Role", cell: ({ row }) => <RoleBadge role={row.original.user.role} /> }),
  col.display({
    id: "hours",
    header: "Meeting time",
    cell: ({ row }) => (
      <span className="meter-cell">
        <Meter value={row.original.minutesUsed} max={row.original.minutesLimit} tone={walletHealth(row.original)} label="hours used" />
        {formatMinutes(row.original.minutesUsed)} / {formatMinutes(row.original.minutesLimit)}
      </span>
    ),
  }),
  col.display({ id: "budget", header: "Budget", cell: ({ row }) => formatMoney(row.original.budget) }),
  col.display({ id: "spent", header: "Spent", cell: ({ row }) => formatMoney(row.original.spent) }),
  col.display({
    id: "balance",
    header: "Balance",
    cell: ({ row }) => <span className={`money money--${walletHealth(row.original)}`}>{formatMoney(row.original.balance)}</span>,
  }),
]);

export function WalletTable({ wallets }: { readonly wallets: readonly Wallet[] }) {
  return <DataTable columns={columns} data={wallets} rowKey={(w) => w.user.id} empty="No people found." />;
}
