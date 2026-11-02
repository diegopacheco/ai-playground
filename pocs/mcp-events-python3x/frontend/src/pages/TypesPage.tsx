import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { useEventTypes } from "../mcp/hooks";
import type { EventDescriptor, ObjectSchema } from "../mcp/types";

const column = createColumnHelper<EventDescriptor>();

function SchemaList({ schema }: { schema: ObjectSchema }) {
  return (
    <ul className="schema">
      {Object.entries(schema.properties).map(([key, prop]) => (
        <li key={key}>
          <span className="mono">{key}</span> <span className="dim">{prop.enum ? prop.enum.join(" | ") : prop.type}</span>
        </li>
      ))}
    </ul>
  );
}

const columns = [
  column.accessor("name", { header: "name", cell: (c) => <span className="mono strong">{c.getValue()}</span> }),
  column.accessor("description", { header: "description" }),
  column.accessor("delivery", {
    header: "delivery",
    cell: (c) => (
      <div className="chips">
        {(["poll", "push", "webhook"] as const).map((mode) => (
          <span key={mode} className={`chip ${c.getValue().includes(mode) ? "mode-on" : "mode-off"}`}>
            {mode}
          </span>
        ))}
      </div>
    ),
  }),
  column.accessor("inputSchema", { header: "inputSchema (subscription arguments)", cell: (c) => <SchemaList schema={c.getValue()} /> }),
  column.accessor("payloadSchema", { header: "payloadSchema (event data)", cell: (c) => <SchemaList schema={c.getValue()} /> }),
];

export function TypesPage() {
  const types = useEventTypes();
  const table = useReactTable({ data: types.data ?? [], columns, getCoreRowModel: getCoreRowModel() });
  return (
    <div className="page">
      <p className="lead">
        Discovery: <span className="mono">events/list</span> returns every event type with its delivery modes, the{" "}
        <span className="mono">inputSchema</span> for subscription arguments and the <span className="mono">payloadSchema</span> of the
        event data. Webhook is off here, so <span className="mono">events/subscribe</span> answers <span className="mono">-32026 Unsupported</span>.
      </p>
      <section className="card">
        <table>
          <thead>
            {table.getHeaderGroups().map((group) => (
              <tr key={group.id}>
                {group.headers.map((header) => (
                  <th key={header.id}>{flexRender(header.column.columnDef.header, header.getContext())}</th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
