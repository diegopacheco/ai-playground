import { catalog } from "../domain/catalog";

export function PlaybooksPage() {
  return (
    <div className="grid">
      {catalog.playbooks.map((p) => (
        <section className="card" key={p.id}>
          <h3>{p.title}</h3>
          <code className="muted">playbooks/{p.id}.md</code>
          <ol>
            {p.steps.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ol>
        </section>
      ))}
    </div>
  );
}
