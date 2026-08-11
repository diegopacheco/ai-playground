import { useMemo } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { Product } from "@/types";
import {
  TableWrapper,
  StyledTable,
  StockBadge,
  ActionButton,
  ActionCell,
  PriceCell,
  EmptyState,
} from "./table.styles";

export interface ProductTableProps {
  products: Product[];
  onEdit: (product: Product) => void;
  onDelete: (product: Product) => void;
}

export function ProductTable({
  products,
  onEdit,
  onDelete,
}: ProductTableProps) {
  const columns = useMemo<ColumnDef<Product>[]>(
    () => [
      {
        accessorKey: "name",
        header: "Name",
        cell: (info) => (
          <strong style={{ fontWeight: 500 }}>{info.getValue() as string}</strong>
        ),
      },
      {
        accessorKey: "description",
        header: "Description",
        cell: (info) => info.getValue() || "—",
      },
      {
        accessorKey: "price",
        header: "Price",
        cell: (info) => `$${Number(info.getValue()).toFixed(2)}`,
      },
      {
        accessorKey: "category",
        header: "Category",
      },
      {
        accessorKey: "in_stock",
        header: "Stock",
        cell: (info) => (
          <StockBadge $inStock={info.getValue() as boolean}>
            {info.getValue() ? "In Stock" : "Out of Stock"}
          </StockBadge>
        ),
      },
      {
        id: "actions",
        header: "",
        enableSorting: false,
        cell: ({ row }) => (
          <ActionCell>
            <ActionButton
              onClick={() => onEdit(row.original)}
              title="Edit"
            >
              ✎
            </ActionButton>
            <ActionButton
              onClick={() => onDelete(row.original)}
              className="danger"
              title="Delete"
            >
              🗑
            </ActionButton>
          </ActionCell>
        ),
      },
    ],
    [onEdit, onDelete]
  );

  const table = useReactTable({
    data: products,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  if (products.length === 0) {
    return (
      <TableWrapper>
        <EmptyState>
          <div>📦</div>
          <p>No products found. Add your first product to get started.</p>
        </EmptyState>
      </TableWrapper>
    );
  }

  return (
    <TableWrapper>
      <StyledTable>
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th key={header.id}>
                  {header.isPlaceholder
                    ? null
                    : flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows?.length ? (
            table.getRowModel().rows.map((row) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => {
                  const isActions = cell.column.id === "actions";
                  const isPrice = cell.column.id === "price";
                  return isPrice ? (
                    <PriceCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </PriceCell>
                  ) : isActions ? (
                    <ActionCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </ActionCell>
                  ) : (
                    <td key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  );
                })}
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={columns.length} style={{ textAlign: "center" }}>
                No products found.
              </td>
            </tr>
          )}
        </tbody>
      </StyledTable>
    </TableWrapper>
  );
}
