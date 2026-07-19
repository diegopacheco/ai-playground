import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SavedQueries } from "./SavedQueries";
import { api } from "@lib/api";
import type { SavedQuery } from "@lib/types";

jest.mock("@lib/api");
const mocked = api as jest.Mocked<typeof api>;

const saved = (overrides: Partial<SavedQuery> = {}): SavedQuery => ({
  id: 1,
  projectId: 3,
  connectionId: 7,
  name: "stuck orders",
  statement: "SELECT * FROM orders WHERE status = 'placed'",
  kind: "postgres",
  description: null,
  createdBy: "diego",
  updatedAt: "2026-07-18T00:00:00Z",
  ...overrides
});

beforeEach(() => {
  jest.clearAllMocks();
  mocked.savedQueries.mockResolvedValue([saved()]);
  mocked.saveQuery.mockResolvedValue(saved());
  mocked.deleteSavedQuery.mockResolvedValue({});
});

async function open(onUse = jest.fn(), statement = "SELECT 1") {
  render(
    <SavedQueries projectId={3} connectionId={7} connectionKind="postgres" statement={statement} onUse={onUse} />
  );
  await userEvent.click(screen.getByRole("button", { name: /saved/ }));
  await screen.findByText("stuck orders");
  return onUse;
}

describe("SavedQueries", () => {
  it("lists queries shared across the project, not just the caller's own", async () => {
    await open();
    await waitFor(() => expect(mocked.savedQueries).toHaveBeenCalledWith(3));
    expect(screen.getByText("stuck orders")).toBeInTheDocument();
  });

  it("loads a saved query into the editor without running it", async () => {
    const onUse = await open();
    await userEvent.click(screen.getByText("stuck orders"));
    expect(onUse).toHaveBeenCalledWith("SELECT * FROM orders WHERE status = 'placed'");
    expect(mocked.query).not.toHaveBeenCalled();
  });

  it("saves the current statement pinned to this connection by default", async () => {
    await open(jest.fn(), "SELECT count(*) FROM customers");
    await userEvent.type(screen.getByLabelText("saved query name"), "customer count");
    await userEvent.click(screen.getByRole("button", { name: "save" }));
    await waitFor(() =>
      expect(mocked.saveQuery).toHaveBeenCalledWith(3, {
        name: "customer count",
        statement: "SELECT count(*) FROM customers",
        connectionId: 7,
        description: null
      })
    );
  });

  it("can save a query unpinned so the same sql runs against staging and prod", async () => {
    await open(jest.fn(), "SELECT 1");
    await userEvent.type(screen.getByLabelText("saved query name"), "portable");
    await userEvent.click(screen.getByLabelText(/pin to this connection/));
    await userEvent.click(screen.getByRole("button", { name: "save" }));
    await waitFor(() =>
      expect(mocked.saveQuery).toHaveBeenCalledWith(3, expect.objectContaining({ connectionId: null }))
    );
  });

  it("will not save without a name", async () => {
    await open();
    expect(screen.getByRole("button", { name: "save" })).toBeDisabled();
    expect(mocked.saveQuery).not.toHaveBeenCalled();
  });

  it("will not save an empty statement", async () => {
    await open(jest.fn(), "   ");
    await userEvent.type(screen.getByLabelText("saved query name"), "empty");
    expect(screen.getByRole("button", { name: "save" })).toBeDisabled();
  });

  it("marks a query belonging to another engine as not usable here", async () => {
    mocked.savedQueries.mockResolvedValue([saved({ connectionId: 99, kind: "redis" })]);
    await open();
    expect(document.querySelector(".saved-item-foreign")).not.toBeNull();
  });

  it("treats an unpinned query as usable on any connection", async () => {
    mocked.savedQueries.mockResolvedValue([saved({ connectionId: null, kind: "any" })]);
    await open();
    expect(document.querySelector(".saved-item-foreign")).toBeNull();
    expect(screen.getByText("any")).toBeInTheDocument();
  });

  it("deletes a saved query", async () => {
    await open();
    await userEvent.click(screen.getByRole("button", { name: "Delete stuck orders" }));
    await waitFor(() => expect(mocked.deleteSavedQuery).toHaveBeenCalledWith(3, 1));
  });

  it("renders into document.body so toolbar chrome cannot render it behind the editor", async () => {
    await open();
    expect(document.querySelector(".saved-panel")?.parentElement).toBe(document.body);
  });

  it("says so when the project has no saved queries yet", async () => {
    mocked.savedQueries.mockResolvedValue([]);
    render(<SavedQueries projectId={3} connectionId={7} connectionKind="postgres" statement="SELECT 1" onUse={jest.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /saved/ }));
    expect(await screen.findByText("nothing saved yet")).toBeInTheDocument();
  });
});
