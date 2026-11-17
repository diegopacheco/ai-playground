import { useState } from "react";
import { catalog, findPlaybook } from "../domain/catalog";
import { route, triggers } from "../domain/router";
import { complete, gate, reopen, report, skip, startRun, todoLine, togglePrinciple, type Run } from "../domain/run";
import { StepRow } from "./StepRow";

const PRESETS: readonly string[] = [
  "The contact form is broken, it sends emails twice on double click",
  "Search page got slow, latency doubled after the last deploy",
  "Explain how does the session cache work",
  "Add a dark mode toggle to the settings component",
  "Migrate every service across the codebase to the new auth API",
];

export function LabPage() {
  const [task, setTask] = useState<string>(PRESETS[0] ?? "");
  const [run, setRun] = useState<Run | null>(null);
  const routed = route(task);
  const fired = triggers(task);
  const playbook = routed.kind === "playbook" ? findPlaybook(routed.id) : undefined;
  const result = run ? gate(run) : null;

  const choose = (text: string) => {
    setTask(text);
    setRun(null);
  };

  return (
    <div className="lab">
      <section className="card">
        <h2>1. /poteto-mode &lt;outcome&gt;</h2>
        <textarea value={task} onChange={(e) => choose(e.target.value)} rows={2} />
        <div className="chips">
          {PRESETS.map((p) => (
            <button key={p} className="chip" onClick={() => choose(p)}>
              {p}
            </button>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>2. Route</h2>
        {routed.kind === "playbook" && playbook ? (
          <p>
            Playbook <strong>{playbook.title}</strong> <span className="muted">matched: {routed.matched.join(", ")}</span>
          </p>
        ) : (
          <p>
            <strong>figure-it-out</strong> <span className="muted">{routed.matched.length ? `too broad: ${routed.matched.join(", ")}` : "no playbook fits"}, design a custom playbook</span>
          </p>
        )}
        <div className="chips">
          {fired.map((t) => (
            <span key={t.skill} className="tag" title={t.reason}>
              /{t.skill} <span className="muted">{t.reason}</span>
            </span>
          ))}
        </div>
        {playbook && !run && (
          <button className="primary" onClick={() => setRun(startRun(playbook))}>
            Copy steps into todo list
          </button>
        )}
      </section>

      {run && result && (
        <>
          <section className="card">
            <h2>3. Todo list, word for word</h2>
            {run.steps.map((s, i) => (
              <StepRow
                key={s.text}
                step={s}
                onDone={(evidence) => setRun(complete(run, i, evidence))}
                onSkip={(reason) => setRun(skip(run, i, reason))}
                onReopen={() => setRun(reopen(run, i))}
              />
            ))}
          </section>

          <section className="card">
            <h2>4. Name the principles that changed a decision</h2>
            <div className="chips">
              {catalog.principles.map((p) => (
                <button key={p.name} title={p.when} className={run.principles.includes(p.name) ? "chip on" : "chip"} onClick={() => setRun(togglePrinciple(run, p.name))}>
                  {p.name}
                </button>
              ))}
            </div>
          </section>

          <section className={result.ok ? "card ok" : "card blocked"}>
            <h2>5. Evidence gate</h2>
            {result.ok ? (
              <pre>{report(run)}</pre>
            ) : (
              <>
                <p>
                  <strong>Not done.</strong> Done needs evidence from the real artifact.
                </p>
                <ul>
                  {result.missing.map((m) => (
                    <li key={m}>{m}</li>
                  ))}
                </ul>
                <pre className="muted">{run.steps.map(todoLine).join("\n")}</pre>
              </>
            )}
          </section>
        </>
      )}
    </div>
  );
}
