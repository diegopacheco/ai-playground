import { useForm } from "@tanstack/react-form";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { useMemo, useState } from "react";
import { apiBase, type Agent } from "../agent";
import { Card, ErrorLine, WithAgent } from "../components";
import type { Hold, Room } from "../poppy/types";

type Filters = { maxPrice: string; pets: boolean };

const column = createColumnHelper<Room>();

function Rooms({ agent }: { agent: Agent }) {
  const base = apiBase(agent.document);
  const queryClient = useQueryClient();
  const [filters, setFilters] = useState<Filters>({ maxPrice: "", pets: false });
  const query = new URLSearchParams({ max_price: filters.maxPrice, pets: String(filters.pets) });
  const rooms = useQuery({ queryKey: ["rooms", filters], queryFn: () => agent.client.call<{ rooms: Room[] }>(`${base}/api/rooms?${query}`) });
  const holds = useQuery({ queryKey: ["holds"], queryFn: () => agent.client.call<{ user_id: string; holds: Hold[] }>(`${base}/api/holds`) });
  const hold = useMutation({
    mutationFn: (roomId: string) => agent.client.call<Hold>(`${base}/api/holds`, { method: "POST", body: { room_id: roomId, nights: 2 } }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["holds"] }),
  });
  const form = useForm({ defaultValues: filters, onSubmit: ({ value }) => setFilters(value) });
  const columns = useMemo(
    () => [
      column.accessor("name", { header: "Room" }),
      column.accessor("beds", { header: "Beds" }),
      column.accessor("price_per_night", { header: "Per night", cell: (info) => `$${info.getValue()}` }),
      column.accessor("pets_allowed", { header: "Pets", cell: (info) => (info.getValue() ? "yes" : "no") }),
      column.display({ id: "hold", header: "", cell: (info) => <button onClick={() => hold.mutate(info.row.original.id)}>Hold 2 nights</button> }),
    ],
    [hold],
  );
  const table = useReactTable({ data: rooms.data?.rooms ?? [], columns, getCoreRowModel: getCoreRowModel() });
  return (
    <div className="grid">
      <Card title="Search rooms (GET /api/rooms, signed Session)">
        <form
          className="row"
          onSubmit={(event) => {
            event.preventDefault();
            form.handleSubmit();
          }}
        >
          <form.Field name="maxPrice">
            {(field) => <input placeholder="max price" inputMode="numeric" value={field.state.value} onChange={(event) => field.handleChange(event.target.value)} />}
          </form.Field>
          <form.Field name="pets">
            {(field) => (
              <label>
                <input type="checkbox" checked={field.state.value} onChange={(event) => field.handleChange(event.target.checked)} /> pets
              </label>
            )}
          </form.Field>
          <button type="submit">Search</button>
        </form>
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
        <ErrorLine error={rooms.error ?? hold.error} />
      </Card>
      <Card title="My holds (keyed by pairwise User ID)">
        <p className="dim">
          User ID <code>{holds.data?.user_id}</code>. Switch user on the Discover tab and this list starts empty.
        </p>
        <ul>
          {holds.data?.holds.map((item) => (
            <li key={item.id}>
              {item.room_name}, {item.nights} nights, ${item.total}
            </li>
          ))}
        </ul>
        {holds.data?.holds.length === 0 ? <p className="dim">No holds yet.</p> : null}
      </Card>
    </div>
  );
}

export function RoomsPage() {
  return <WithAgent>{(agent) => <Rooms agent={agent} />}</WithAgent>;
}
