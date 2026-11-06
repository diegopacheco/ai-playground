import { createColumnHelper, tableFeatures, useTable, type ColumnHelper, type RowData } from "@tanstack/react-table";

export const features = tableFeatures({});
export type Features = typeof features;

export const columnHelper = <T extends RowData>() => createColumnHelper<Features, T>();

type Props<T extends RowData> = {
  data: T[];
  columns: ReturnType<ColumnHelper<Features, T>["columns"]>;
  empty: string;
  rowClass?: (row: T) => string;
};

export function DataTable<T extends RowData>({ data, columns, empty, rowClass }: Props<T>) {
  const table = useTable({ features, columns, data });
  if (data.length === 0) {
    return <p className="empty">{empty}</p>;
  }
  return (
    <table className="grid">
      <thead>
        {table.getHeaderGroups().map((group) => (
          <tr key={group.id}>
            {group.headers.map((header) => (
              <th key={header.id}>
                <table.FlexRender header={header} />
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id} className={rowClass?.(row.original) ?? ""}>
            {row.getAllCells().map((cell) => (
              <td key={cell.id}>
                <table.FlexRender cell={cell} />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
