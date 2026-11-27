import type { TableView } from "../types";

type TablesViewProps = { readonly tables: readonly TableView[] };

export function TablesView({ tables }: TablesViewProps) {
  if (tables.length === 0) return <p className="muted">No tables found in this document.</p>;
  return (
    <div className="tables">
      {tables.map((t) => (
        <figure key={t.index} className="card">
          <figcaption>
            Table {t.index + 1}
            {t.caption ? ` - ${t.caption}` : ""}
          </figcaption>
          <div className="scroll">
            <table className="grid">
              <thead>
                <tr>
                  {t.headers.map((h, i) => (
                    <th key={i}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {t.rows.map((row, r) => (
                  <tr key={r}>
                    {row.map((cell, c) => (
                      <td key={c}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </figure>
      ))}
    </div>
  );
}
