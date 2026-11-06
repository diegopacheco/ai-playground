import { useForm } from "@tanstack/react-form";
import { useMutation } from "@tanstack/react-query";
import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { COMPANY_URL } from "../agent";
import { Card, ErrorLine } from "../components";
import type { Check, InspectorReport } from "../poppy/types";

const column = createColumnHelper<Check>();
const columns = [
  column.accessor("level", { header: "Result", cell: (info) => <span className={`pill ${info.getValue()}`}>{info.getValue()}</span> }),
  column.accessor("name", { header: "Check" }),
  column.accessor("detail", { header: "Detail" }),
];

async function validate(value: { document: string; host: string }): Promise<InspectorReport> {
  const response = await fetch(`${COMPANY_URL}/inspector/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document: JSON.parse(value.document), host: value.host }),
  });
  if (!response.ok) throw new Error((await response.json()).error_description);
  return response.json();
}

export function InspectorPage() {
  const run = useMutation({ mutationFn: validate });
  const form = useForm({ defaultValues: { document: "", host: "" }, onSubmit: ({ value }) => run.mutate(value) });
  const table = useReactTable({ data: run.data?.checks ?? [], columns, getCoreRowModel: getCoreRowModel() });
  const loadWillow = async () => {
    const document = await (await fetch(`${COMPANY_URL}/.well-known/poppy.json`)).json();
    form.setFieldValue("document", JSON.stringify(document, null, 2));
    form.setFieldValue("host", new URL(COMPANY_URL).host);
  };
  return (
    <div className="grid">
      <Card title="poppy.json inspector (draft 0.4 rules)">
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            form.handleSubmit();
          }}
        >
          <form.Field name="host">{(field) => <input placeholder="host that serves the document, like shop.example" value={field.state.value} onChange={(event) => field.handleChange(event.target.value)} />}</form.Field>
          <form.Field name="document">{(field) => <textarea rows={18} className="mono" placeholder="paste a poppy.json" value={field.state.value} onChange={(event) => field.handleChange(event.target.value)} />}</form.Field>
          <div className="row">
            <button type="button" onClick={loadWillow}>Load Willow Inn</button>
            <button type="submit">Validate</button>
          </div>
        </form>
        <ErrorLine error={run.error} />
      </Card>
      <Card title={run.data ? `${run.data.valid ? "Valid" : "Invalid"} for ${run.data.host}` : "Report"}>
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
      </Card>
    </div>
  );
}
