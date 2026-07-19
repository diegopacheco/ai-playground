import "./DataGrid.css";

export type GridRow = Record<string, unknown>;

export interface DataGridProps {
  columns: string[];
  rows: GridRow[];
  emptyLabel?: string;
  onRowActivate?: (index: number) => void;
}

function cell(value: unknown) {
  if (value === null || value === undefined) {
    return <span className="ds-grid-null">null</span>;
  }
  return String(value);
}

export function DataGrid({ columns, rows, emptyLabel = "no rows", onRowActivate }: DataGridProps) {
  if (columns.length === 0) {
    return <p className="ds-grid-empty">{emptyLabel}</p>;
  }
  return (
    <div className="ds-grid-scroll">
      <table className={onRowActivate ? "ds-grid ds-grid-clickable" : "ds-grid"}>
        <thead>
          <tr>
            <th className="ds-grid-index">#</th>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td className="ds-grid-empty" colSpan={columns.length + 1}>{emptyLabel}</td>
            </tr>
          ) : (
            rows.map((row, index) => (
              <tr
                key={index}
                onDoubleClick={onRowActivate ? () => onRowActivate(index) : undefined}
                title={onRowActivate ? "double-click to see the full row" : undefined}
              >
                <td className="ds-grid-index">{index + 1}</td>
                {columns.map((column) => (
                  <td key={column} title={row[column] === null ? "null" : String(row[column] ?? "")}>
                    {cell(row[column])}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
