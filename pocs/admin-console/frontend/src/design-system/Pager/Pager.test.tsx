import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Pager } from "./Pager";

const handlers = () => ({
  onFirst: jest.fn(),
  onPrevious: jest.fn(),
  onNext: jest.fn()
});

describe("Pager", () => {
  it("disables next on the last page so an operator cannot request rows that do not exist", () => {
    render(<Pager pageNumber={3} rowCount={12} hasMore={false} {...handlers()} />);
    expect(screen.getByRole("button", { name: "Next page" })).toBeDisabled();
  });

  it("enables next while the engine reports more rows", () => {
    render(<Pager pageNumber={1} rowCount={100} hasMore {...handlers()} />);
    expect(screen.getByRole("button", { name: "Next page" })).toBeEnabled();
  });

  it("disables first and prev on page one", () => {
    render(<Pager pageNumber={1} rowCount={100} hasMore {...handlers()} />);
    expect(screen.getByRole("button", { name: "Previous page" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "First page" })).toBeDisabled();
  });

  it("advances only when next is pressed", async () => {
    const calls = handlers();
    render(<Pager pageNumber={1} rowCount={100} hasMore {...calls} />);
    await userEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(calls.onNext).toHaveBeenCalledTimes(1);
    expect(calls.onPrevious).not.toHaveBeenCalled();
  });

  it("shows page, row count and timing so a slow query is obvious", () => {
    render(<Pager pageNumber={2} rowCount={100} elapsedMs={42} hasMore {...handlers()} />);
    expect(screen.getByText(/page 2 · 100 rows · 42ms/)).toBeInTheDocument();
  });

  it("offers a count only when the total is unknown, since counting is a separate expensive query", async () => {
    const onCount = jest.fn();
    const { rerender } = render(
      <Pager pageNumber={1} rowCount={100} hasMore totalRows={null} onCount={onCount} {...handlers()} />
    );
    await userEvent.click(screen.getByRole("button", { name: "count rows" }));
    expect(onCount).toHaveBeenCalled();
    rerender(<Pager pageNumber={1} rowCount={100} hasMore totalRows={2000} onCount={onCount} {...handlers()} />);
    expect(screen.queryByRole("button", { name: "count rows" })).not.toBeInTheDocument();
    expect(screen.getByText(/2000 total/)).toBeInTheDocument();
  });
});
