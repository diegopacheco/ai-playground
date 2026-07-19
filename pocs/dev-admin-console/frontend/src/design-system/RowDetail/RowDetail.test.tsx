import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RowDetail } from "./RowDetail";
import { DataGrid } from "../DataGrid/DataGrid";

const columns = ["id", "email", "payload", "deleted_at"];
const rows = [
  { id: "1", email: "a@example.com", payload: '{"orderId":1,"status":"placed"}', deleted_at: null },
  { id: "2", email: "b@example.com", payload: '{"orderId":2}', deleted_at: "2026-01-01" }
];

describe("DataGrid row activation", () => {
  it("opens the row on double-click, not single-click, so selecting text does not pop a modal", async () => {
    const onRowActivate = jest.fn();
    render(<DataGrid columns={columns} rows={rows} onRowActivate={onRowActivate} />);
    const cell = screen.getByText("a@example.com");
    await userEvent.click(cell);
    expect(onRowActivate).not.toHaveBeenCalled();
    await userEvent.dblClick(cell);
    expect(onRowActivate).toHaveBeenCalledWith(0);
  });

  it("does not make rows look clickable when no handler is wired", () => {
    const { container } = render(<DataGrid columns={columns} rows={rows} />);
    expect(container.querySelector(".ds-grid-clickable")).toBeNull();
  });
});

describe("RowDetail", () => {
  const setup = (index = 0) => {
    const onClose = jest.fn();
    const onNavigate = jest.fn();
    render(
      <RowDetail columns={columns} rows={rows} index={index} onClose={onClose} onNavigate={onNavigate} />
    );
    return { onClose, onNavigate };
  };

  it("shows every column of the row, including ones truncated in the grid", () => {
    setup();
    columns.forEach((column) => expect(screen.getByText(column)).toBeInTheDocument());
    expect(screen.getByText("a@example.com")).toBeInTheDocument();
  });

  it("pretty-prints JSON values, which is the main reason to open a row at all", () => {
    setup();
    expect(screen.getByText(/"orderId": 1/)).toBeInTheDocument();
    expect(screen.getByText(/"status": "placed"/)).toBeInTheDocument();
  });

  it("distinguishes a null from an empty value", () => {
    setup();
    expect(screen.getByText("null")).toHaveClass("row-detail-null");
  });

  it("reports its position in the page so an operator knows where they are", () => {
    setup(1);
    expect(screen.getByText(/row 2/)).toBeInTheDocument();
    expect(screen.getByText(/of 2 on this page/)).toBeInTheDocument();
  });

  it("moves to the next row with the arrow key, so a page can be read without closing", async () => {
    const { onNavigate } = setup(0);
    await userEvent.keyboard("{ArrowDown}");
    expect(onNavigate).toHaveBeenCalledWith(1);
  });

  it("does not move past the last row of the page", async () => {
    const { onNavigate } = setup(1);
    await userEvent.keyboard("{ArrowDown}");
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it("disables the previous control on the first row", () => {
    setup(0);
    expect(screen.getByRole("button", { name: "Previous row" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Next row" })).toBeEnabled();
  });

  it("closes on Escape", async () => {
    const { onClose } = setup();
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });

  it("closes when the backdrop is clicked but not when the panel itself is", async () => {
    const onClose = jest.fn();
    render(<RowDetail columns={columns} rows={rows} index={0} onClose={onClose} onNavigate={jest.fn()} />);
    await userEvent.click(document.querySelector(".row-detail") as HTMLElement);
    expect(onClose).not.toHaveBeenCalled();
    await userEvent.click(document.querySelector(".row-detail-backdrop") as HTMLElement);
    expect(onClose).toHaveBeenCalled();
  });

  it("renders nothing when the index points past the page, rather than crashing", () => {
    render(<RowDetail columns={columns} rows={rows} index={9} onClose={jest.fn()} onNavigate={jest.fn()} />);
    expect(document.querySelector(".row-detail")).toBeNull();
  });

  it("renders into document.body so panel chrome with backdrop-filter cannot trap it in a stacking context", () => {
    render(<RowDetail columns={columns} rows={rows} index={0} onClose={jest.fn()} onNavigate={jest.fn()} />);
    const backdrop = document.querySelector(".row-detail-backdrop");
    expect(backdrop?.parentElement).toBe(document.body);
  });
});
