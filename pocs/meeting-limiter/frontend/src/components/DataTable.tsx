import { createColumnHelper, tableFeatures, useTable, type ColumnDef, type RowData } from "@tanstack/react-table";

export const tableFeatureSet = tableFeatures({});

type Features = typeof tableFeatureSet;

export type Columns<T extends RowData> = ReadonlyArray<ColumnDef<Features, T, unknown>>;

export function columnHelper<T extends RowData>() {
  return createColumnHelper<Features, T>();
}

type Props<T extends RowData> = {
  readonly columns: Columns<T>;
  readonly data: readonly T[];
  readonly empty: string;
  readonly rowKey: (row: T) => string;
  readonly onRowClick?: (row: T) => void;
};

export function DataTable<T extends RowData>({ columns, data, empty, rowKey, onRowClick }: Props<T>) {
  const table = useTable({ features: tableFeatureSet, columns, data: [...data], getRowId: rowKey });
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          {table.getHeaderGroups().map((group) => (
            <tr key={group.id}>
              {group.headers.map((header) => (
                <th key={header.id}>{header.isPlaceholder ? null : <table.FlexRender header={header} />}</th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {data.length === 0 && (
            <tr>
              <td className="empty" colSpan={columns.length}>
                {empty}
              </td>
            </tr>
          )}
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className={onRowClick ? "clickable" : undefined} onClick={onRowClick ? () => onRowClick(row.original) : undefined}>
              {row.getAllCells().map((cell) => (
                <td key={cell.id}>
                  <table.FlexRender cell={cell} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
