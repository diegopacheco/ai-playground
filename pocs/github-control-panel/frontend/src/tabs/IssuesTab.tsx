import { useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import { api } from "../lib/api";
import { LabelChip, StateBadge } from "../components/Badges";
import { Loading, ErrorView, Empty } from "../components/Status";

export function IssuesTab() {
  const issues = useQuery({ queryKey: ["issues"], queryFn: api.issues });
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [search, setSearch] = useState("");
  const [stateFilter, setStateFilter] = useState("ALL");
  const parentRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    const all = issues.data ?? [];
    const term = search.trim().toLowerCase();
    return all.filter((issue) => {
      if (stateFilter !== "ALL" && issue.state !== stateFilter) {
        return false;
      }
      if (!term) {
        return true;
      }
      return (
        issue.title.toLowerCase().includes(term) ||
        issue.repo.toLowerCase().includes(term) ||
        (issue.author ?? "").toLowerCase().includes(term)
      );
    });
  }, [issues.data, search, stateFilter]);

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 88,
    overscan: 8,
  });

  const detail = useQuery({
    queryKey: ["issue", selectedId],
    queryFn: () => api.issue(selectedId as number),
    enabled: selectedId !== null,
  });

  if (issues.isLoading) {
    return <Loading what="issues" />;
  }
  if (issues.isError) {
    return <ErrorView error={issues.error} />;
  }
  if (!issues.data || issues.data.length === 0) {
    return <Empty message="No issues yet. Add repos, then press Sync." />;
  }

  return (
    <div className="issues-layout">
      <div className="issues-list">
        <div className="issues-filters">
          <input
            className="search"
            placeholder="Filter by title, repo or author…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <select value={stateFilter} onChange={(event) => setStateFilter(event.target.value)}>
            <option value="ALL">All</option>
            <option value="OPEN">Open</option>
            <option value="CLOSED">Closed</option>
          </select>
        </div>
        <div className="virtual-scroll" ref={parentRef}>
          <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
            {virtualizer.getVirtualItems().map((virtualRow) => {
              const issue = filtered[virtualRow.index];
              return (
                <button
                  key={issue.id}
                  className={`issue-row ${selectedId === issue.id ? "selected" : ""}`}
                  style={{ transform: `translateY(${virtualRow.start}px)`, height: virtualRow.size }}
                  onClick={() => setSelectedId(issue.id)}
                >
                  <div className="issue-row-top">
                    <span className="issue-repo">{issue.repo}</span>
                    <StateBadge state={issue.state} />
                  </div>
                  <div className="issue-row-title">
                    #{issue.number} {issue.title}
                  </div>
                  <div className="issue-row-labels">
                    {issue.labels.slice(0, 4).map((label) => (
                      <LabelChip key={label.name} label={label} />
                    ))}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="issue-detail">
        {selectedId === null && <Empty message="Select an issue on the left to read it." />}
        {selectedId !== null && detail.isLoading && <Loading what="issue" />}
        {selectedId !== null && detail.isError && <ErrorView error={detail.error} />}
        {detail.data && (
          <article>
            <header className="detail-head">
              <div className="detail-meta">
                <span className="issue-repo">{detail.data.repo}</span>
                <StateBadge state={detail.data.state} />
              </div>
              <h1>
                #{detail.data.number} {detail.data.title}
              </h1>
              <div className="detail-sub">
                opened by <strong>{detail.data.author ?? "ghost"}</strong>
                {detail.data.createdAt ? ` on ${new Date(detail.data.createdAt).toLocaleDateString()}` : ""} ·{" "}
                {detail.data.commentsCount} comments
                <a href={detail.data.url} target="_blank" rel="noreferrer" className="detail-link">
                  open on GitHub ↗
                </a>
              </div>
              <div className="issue-row-labels">
                {detail.data.labels.map((label) => (
                  <LabelChip key={label.name} label={label} />
                ))}
              </div>
              {detail.data.assignees.length > 0 && (
                <div className="detail-assignees">assignees: {detail.data.assignees.join(", ")}</div>
              )}
            </header>
            <div className="issue-body">{detail.data.body?.trim() ? detail.data.body : "No description provided."}</div>
          </article>
        )}
      </div>
    </div>
  );
}
