import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { api } from "../api";
import { columnHelper, DataTable } from "../components/DataTable";
import { CreateSandboxForm } from "../components/CreateSandboxForm";
import type { Sandbox } from "../types";

const col = columnHelper<Sandbox>();

function DeleteButton({ name }: { name: string }) {
  const client = useQueryClient();
  const remove = useMutation({
    mutationFn: () => api.deleteSandbox(name),
    onSuccess: () => client.invalidateQueries({ queryKey: ["sandboxes"] }),
  });
  return (
    <button type="button" className="danger" onClick={() => remove.mutate()} disabled={remove.isPending}>
      {remove.isPending ? "Deleting..." : "Delete"}
    </button>
  );
}

const columns = col.columns([
  col.accessor("name", {
    header: "Sandbox",
    cell: (c) => <Link to="/sandboxes/$name" params={{ name: c.getValue() }}>{c.getValue()}</Link>,
  }),
  col.accessor("phase", { header: "Phase", cell: (c) => <span className={`badge phase-${c.getValue().toLowerCase()}`}>{c.getValue()}</span> }),
  col.accessor("policyVersion", { header: "Policy version" }),
  col.accessor("createdAt", { header: "Created (UTC)" }),
  col.display({ id: "actions", header: "", cell: (c) => <DeleteButton name={c.row.original.name} /> }),
]);

export function SandboxesPage() {
  const sandboxes = useQuery({ queryKey: ["sandboxes"], queryFn: api.sandboxes, refetchInterval: 3000 });
  return (
    <>
      <section className="card">
        <h2>Sandboxes</h2>
        <p className="hint">Each sandbox is an isolated container managed by the OpenShell gateway, with Landlock filesystem rules, a seccomp process filter and a default deny network proxy.</p>
        <CreateSandboxForm />
      </section>
      <section className="card">
        {sandboxes.error && <em className="error">{sandboxes.error.message}</em>}
        <DataTable data={sandboxes.data ?? []} columns={columns} empty={sandboxes.isLoading ? "Loading..." : "No sandboxes yet."} />
      </section>
    </>
  );
}
