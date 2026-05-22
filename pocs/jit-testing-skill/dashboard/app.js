async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  return r.json();
}

function el(tag, attrs, ...children) {
  const e = document.createElement(tag);
  if (attrs) {
    for (const k in attrs) {
      if (k === "class") e.className = attrs[k];
      else if (k === "html") e.innerHTML = attrs[k];
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
  }
  for (const c of children.flat()) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function scoreClass(s) {
  if (s > 0.3) return "score pos";
  if (s < -0.3) return "score neg";
  return "score";
}

function fmtTs(ts) {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleString();
}

async function renderCounters() {
  const s = await fetchJSON("/api/summary");
  const root = document.getElementById("counters");
  root.innerHTML = "";
  const tiles = [
    { label: "Runs", value: s.runs, cls: "" },
    { label: "Total catches", value: s.total_catches, cls: "" },
    { label: "Surfaced", value: s.surfaced, cls: "warn" },
    { label: "Confirmed bugs", value: s.confirmed, cls: "bad" },
    { label: "Dismissed", value: s.dismissed, cls: "good" },
  ];
  for (const t of tiles) {
    root.appendChild(el("div", { class: "counter " + t.cls },
      el("div", { class: "label" }, t.label),
      el("div", { class: "value" }, String(t.value)),
    ));
  }
  renderSparkline(s.trend || []);
  renderTargets(s.by_target || {});
}

function renderSparkline(trend) {
  const root = document.getElementById("sparkline");
  root.innerHTML = "";
  const max = Math.max(1, ...trend);
  for (const v of trend) {
    const h = Math.max(2, Math.round((v / max) * 64));
    root.appendChild(el("div", { class: "bar", style: `height:${h}px`, title: String(v) }));
  }
  if (trend.length === 0) {
    root.appendChild(el("div", { class: "muted" }, "no runs yet"));
  }
}

function renderTargets(byTarget) {
  const root = document.getElementById("targets-cards");
  root.innerHTML = "";
  const entries = Object.entries(byTarget);
  if (entries.length === 0) {
    root.appendChild(el("div", { class: "muted" }, "no data yet"));
    return;
  }
  for (const [target, count] of entries) {
    root.appendChild(el("div", { class: "card" },
      el("div", { class: "target-name" }, target),
      el("div", { class: "stat" }, el("span", null, "total catches"), el("span", { class: "v" }, String(count))),
    ));
  }
}

async function renderRuns() {
  const runs = await fetchJSON("/api/runs");
  const tbody = document.querySelector("#runs-table tbody");
  tbody.innerHTML = "";
  if (runs.length === 0) {
    tbody.appendChild(el("tr", null, el("td", { colspan: "8", class: "muted" }, "No runs yet. Use /jit-testing first.")));
    return;
  }
  for (const r of runs) {
    const row = el("tr", { onclick: () => showRun(r.run_id) },
      el("td", null, r.run_id),
      el("td", null, r.target || "-"),
      el("td", null, r.workflow || "-"),
      el("td", null, r.diff || "-"),
      el("td", null, String(r.total)),
      el("td", null, String(r.surfaced)),
      el("td", null, String(r.confirmed)),
      el("td", null, String(r.dismissed)),
    );
    tbody.appendChild(row);
  }
}

async function showRun(rid) {
  const data = await fetchJSON("/api/run/" + encodeURIComponent(rid));
  const sec = document.getElementById("detail");
  const body = document.getElementById("detail-body");
  sec.hidden = false;
  body.innerHTML = "";
  body.appendChild(el("p", { class: "muted" }, `run ${rid} — target ${data.target} — ${(data.catches||[]).length} catches`));
  for (const c of (data.catches || [])) {
    const sc = (c.rubfake && c.rubfake.score) || 0;
    const card = el("div", { class: "catch" },
      el("div", { class: "sense" }, c.sense_check || c.name || "(no sense check)"),
      el("div", null, el("span", { class: scoreClass(sc) }, `score ${sc.toFixed(2)}`)),
      el("div", { class: "chips" },
        ...(c.rubfake && c.rubfake.fp_patterns || []).map(p => el("span", { class: "chip fp" }, p)),
        ...(c.rubfake && c.rubfake.tp_patterns || []).map(p => el("span", { class: "chip tp" }, p)),
      ),
      c.behavior_input ? el("div", { class: "muted" }, "input: " + c.behavior_input) : null,
      (c.parent_output !== undefined && c.diff_output !== undefined)
        ? el("div", { class: "muted" }, `parent → ${c.parent_output}    diff → ${c.diff_output}`)
        : null,
      c.test_code ? el("pre", null, c.test_code) : null,
      el("div", { class: "verdict-row" },
        el("button", { class: "primary", onclick: () => setVerdict(rid, c.id, "confirmed") }, "Confirm bug"),
        el("button", { onclick: () => setVerdict(rid, c.id, "expected") }, "Expected change"),
        el("button", { onclick: () => setVerdict(rid, c.id, "deferred") }, "Defer"),
        c.verdict ? el("span", { class: "muted", style: "margin-left:8px" }, "verdict: " + c.verdict) : null,
      ),
    );
    body.appendChild(card);
  }
  sec.scrollIntoView({ behavior: "smooth" });
}

async function setVerdict(rid, cid, verdict) {
  await fetch(`/api/verdict/${encodeURIComponent(rid)}/${encodeURIComponent(cid)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ verdict }),
  });
  await renderCounters();
  await renderRuns();
  await showRun(rid);
}

async function main() {
  await renderCounters();
  await renderRuns();
}

main();
