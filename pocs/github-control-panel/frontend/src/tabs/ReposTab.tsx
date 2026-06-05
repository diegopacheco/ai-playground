import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Card } from "../components/Card";
import { Loading, ErrorView, Empty } from "../components/Status";

export function ReposTab() {
  const queryClient = useQueryClient();
  const [input, setInput] = useState("");
  const repos = useQuery({ queryKey: ["repos"], queryFn: api.listRepos });

  const add = useMutation({
    mutationFn: (raw: string[]) => api.addRepos(raw),
    onSuccess: () => {
      setInput("");
      void queryClient.invalidateQueries({ queryKey: ["repos"] });
    },
  });

  const remove = useMutation({
    mutationFn: (id: number) => api.removeRepo(id),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["repos"] }),
  });

  const submit = () => {
    const list = input
      .split(/[\n,]/)
      .map((value) => value.trim())
      .filter(Boolean);
    if (list.length > 0) {
      add.mutate(list);
    }
  };

  return (
    <div className="stack">
      <Card title="Add repositories">
        <p className="hint">
          Paste public repos as <code>owner/name</code> or full GitHub URLs — one per line or comma-separated.
        </p>
        <textarea
          className="repo-input"
          rows={4}
          placeholder={"facebook/react\nspring-projects/spring-boot\nhttps://github.com/vercel/next.js"}
          value={input}
          onChange={(event) => setInput(event.target.value)}
        />
        <div className="row">
          <button className="primary" onClick={submit} disabled={add.isPending}>
            {add.isPending ? "Adding…" : "Add repos"}
          </button>
          {add.isError && <span className="status error inline">could not add repos</span>}
        </div>
      </Card>

      <Card title={`Tracked repositories (${repos.data?.length ?? 0})`}>
        {repos.isLoading && <Loading what="repositories" />}
        {repos.isError && <ErrorView error={repos.error} />}
        {repos.data && repos.data.length === 0 && <Empty message="No repositories yet. Add some above, then press Sync." />}
        {repos.data && repos.data.length > 0 && (
          <ul className="repo-list">
            {repos.data.map((repo) => (
              <li key={repo.id}>
                <div className="repo-main">
                  <a href={`https://github.com/${repo.fullName}`} target="_blank" rel="noreferrer">
                    {repo.fullName}
                  </a>
                  <span className="repo-sync">
                    {repo.lastSyncedAt ? `synced ${new Date(repo.lastSyncedAt).toLocaleString()}` : "never synced"}
                  </span>
                </div>
                <button className="ghost danger" onClick={() => remove.mutate(repo.id)} disabled={remove.isPending}>
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
