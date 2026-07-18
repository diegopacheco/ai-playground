import { render, screen } from "@testing-library/react";
import { DataGrid } from "./DataGrid";

describe("DataGrid", () => {
  const columns = ["id", "email", "deleted_at"];
  const rows = [
    { id: "1", email: "a@example.com", deleted_at: null },
    { id: "2", email: "b@example.com", deleted_at: "2026-01-01" }
  ];

  it("renders every column header returned by the engine", () => {
    render(<DataGrid columns={columns} rows={rows} />);
    columns.forEach((column) => expect(screen.getByText(column)).toBeInTheDocument());
  });

  it("renders each row", () => {
    render(<DataGrid columns={columns} rows={rows} />);
    expect(screen.getByText("a@example.com")).toBeInTheDocument();
    expect(screen.getByText("b@example.com")).toBeInTheDocument();
  });

  it("distinguishes a database null from the empty string, because they mean different things", () => {
    render(<DataGrid columns={columns} rows={[{ id: "1", email: "", deleted_at: null }]} />);
    expect(screen.getByText("null")).toBeInTheDocument();
    expect(screen.getByText("null")).toHaveClass("ds-grid-null");
  });

  it("numbers rows so an operator can refer to a position in the page", () => {
    const { container } = render(<DataGrid columns={columns} rows={rows} />);
    const indexCells = Array.from(container.querySelectorAll("tbody .ds-grid-index")).map((c) => c.textContent);
    expect(indexCells).toEqual(["1", "2"]);
  });

  it("says the result was empty rather than rendering a bare table", () => {
    render(<DataGrid columns={columns} rows={[]} emptyLabel="0 rows returned" />);
    expect(screen.getByText("0 rows returned")).toBeInTheDocument();
  });

  it("says nothing has run yet when there are no columns at all", () => {
    render(<DataGrid columns={[]} rows={[]} emptyLabel="run a query" />);
    expect(screen.getByText("run a query")).toBeInTheDocument();
  });
});
