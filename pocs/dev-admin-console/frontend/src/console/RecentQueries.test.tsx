import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RecentQueries } from "./RecentQueries";
import { api } from "@lib/api";

jest.mock("@lib/api");
const mocked = api as jest.Mocked<typeof api>;

beforeEach(() => {
  jest.clearAllMocks();
  mocked.history.mockResolvedValue([
    "SELECT * FROM customers ORDER BY id",
    "SELECT count(*) FROM orders"
  ]);
});

async function openList(onPick = jest.fn()) {
  render(<RecentQueries connectionId={7} onPick={onPick} />);
  await userEvent.click(screen.getByRole("button", { name: /recent/ }));
  await screen.findByRole("listbox");
  return onPick;
}

describe("RecentQueries", () => {
  it("loads the caller's own history for this connection", async () => {
    await openList();
    await waitFor(() => expect(mocked.history).toHaveBeenCalledWith(7));
  });

  it("loads the chosen statement into the editor", async () => {
    const onPick = await openList();
    await userEvent.click(await screen.findByTitle("SELECT count(*) FROM orders"));
    expect(onPick).toHaveBeenCalledWith("SELECT count(*) FROM orders");
  });

  it("closes after picking so the dropdown does not cover the editor", async () => {
    await openList();
    await userEvent.click(await screen.findByTitle("SELECT count(*) FROM orders"));
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("loads without executing, so a history entry cannot re-run a slow scan by accident", async () => {
    await openList();
    await userEvent.click(await screen.findByTitle("SELECT count(*) FROM orders"));
    expect(mocked.query).not.toHaveBeenCalled();
  });

  it("renders into document.body so the toolbar's backdrop-filter cannot render it behind the editor", async () => {
    await openList();
    expect(document.querySelector(".recent-queries-list")?.parentElement).toBe(document.body);
  });

  it("closes when clicking outside", async () => {
    await openList();
    await userEvent.click(document.body);
    await waitFor(() => expect(screen.queryByRole("listbox")).not.toBeInTheDocument());
  });

  it("closes on Escape", async () => {
    await openList();
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("listbox")).not.toBeInTheDocument());
  });

  it("says so when there is no history yet", async () => {
    mocked.history.mockResolvedValue([]);
    await openList();
    expect(await screen.findByText("nothing yet")).toBeInTheDocument();
  });
});
