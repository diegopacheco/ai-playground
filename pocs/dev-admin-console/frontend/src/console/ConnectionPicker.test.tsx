import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConnectionPicker } from "./ConnectionPicker";
import type { Connection, ConnectionKind } from "@lib/types";

const make = (id: number, name: string, kind: ConnectionKind, host = "localhost", database?: string): Connection => ({
  id,
  projectId: 1,
  name,
  kind,
  host,
  port: 5432,
  database: database ?? null,
  keyspace: null,
  datacenter: null,
  username: null,
  hasPassword: false,
  createdBy: "admin"
});

const connections = [
  make(1, "demo-postgres", "postgres", "localhost", "shop"),
  make(2, "demo-mysql", "mysql"),
  make(3, "demo-redis", "redis"),
  make(4, "prod-kafka", "kafka", "kafka.internal"),
  make(5, "demo-elasticsearch", "elasticsearch")
];

async function openPicker(onSelect = jest.fn(), selected: Connection | null = connections[0]) {
  render(<ConnectionPicker connections={connections} selected={selected} onSelect={onSelect} />);
  await userEvent.click(screen.getByRole("button", { name: /demo-postgres|choose a connection/ }));
  return onSelect;
}

describe("ConnectionPicker", () => {
  it("shows the current connection and its engine logo on the trigger", () => {
    render(<ConnectionPicker connections={connections} selected={connections[0]} onSelect={jest.fn()} />);
    expect(screen.getByText("demo-postgres")).toBeInTheDocument();
    expect(screen.getByTestId("engine-logo-postgres")).toBeInTheDocument();
  });

  it("prompts to choose when nothing is selected yet", () => {
    render(<ConnectionPicker connections={connections} selected={null} onSelect={jest.fn()} />);
    expect(screen.getByText("choose a connection")).toBeInTheDocument();
  });

  it("opens a modal listing every connection with its logo", async () => {
    await openPicker();
    expect(screen.getByRole("dialog", { name: "Choose a connection" })).toBeInTheDocument();
    expect(screen.getAllByRole("option")).toHaveLength(5);
    expect(screen.getByTestId("engine-logo-kafka")).toBeInTheDocument();
  });

  it("focuses the search box on open so the user can type immediately", async () => {
    await openPicker();
    expect(screen.getByLabelText("search connections")).toHaveFocus();
  });

  it("filters by connection name as the user types", async () => {
    await openPicker();
    await userEvent.type(screen.getByLabelText("search connections"), "redis");
    expect(screen.getAllByRole("option")).toHaveLength(1);
    expect(screen.getByText("demo-redis")).toBeInTheDocument();
  });

  it("filters by engine kind, so typing an engine name finds its connections", async () => {
    await openPicker();
    await userEvent.type(screen.getByLabelText("search connections"), "kafka");
    expect(screen.getAllByRole("option")).toHaveLength(1);
    expect(screen.getByText("prod-kafka")).toBeInTheDocument();
  });

  it("filters by host, which is how an operator tells prod from local", async () => {
    await openPicker();
    await userEvent.type(screen.getByLabelText("search connections"), "internal");
    expect(screen.getAllByRole("option")).toHaveLength(1);
    expect(screen.getByText("prod-kafka")).toBeInTheDocument();
  });

  it("selects the top match on Enter, which is the whole point of search-and-go", async () => {
    const onSelect = await openPicker();
    await userEvent.type(screen.getByLabelText("search connections"), "mysql");
    await userEvent.keyboard("{Enter}");
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ name: "demo-mysql" }));
  });

  it("moves the highlight forward with the right arrow and opens the highlighted one", async () => {
    const onSelect = await openPicker();
    await userEvent.keyboard("{ArrowRight}{ArrowRight}{Enter}");
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ name: "demo-redis" }));
  });

  it("wraps backwards past the start to the end of the grid", async () => {
    const onSelect = await openPicker();
    await userEvent.keyboard("{ArrowLeft}{Enter}");
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ name: "demo-elasticsearch" }));
  });

  it("moves with Tab too, so the modal is reachable without arrow keys", async () => {
    const onSelect = await openPicker();
    await userEvent.keyboard("{Tab}{Enter}");
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ name: "demo-mysql" }));
  });

  it("selects on click", async () => {
    const onSelect = await openPicker();
    await userEvent.click(screen.getByText("prod-kafka"));
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ name: "prod-kafka" }));
  });

  it("closes after selecting so the console is immediately usable", async () => {
    await openPicker();
    await userEvent.click(screen.getByText("demo-mysql"));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("closes on Escape without selecting anything", async () => {
    const onSelect = await openPicker();
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("says so when nothing matches instead of showing an empty grid", async () => {
    await openPicker();
    await userEvent.type(screen.getByLabelText("search connections"), "oracle");
    expect(screen.getByText(/no connection matches/)).toBeInTheDocument();
  });

  it("does nothing on Enter when no connection matches, rather than opening a stale one", async () => {
    const onSelect = await openPicker();
    await userEvent.type(screen.getByLabelText("search connections"), "oracle");
    await userEvent.keyboard("{Enter}");
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("clears the previous search when reopened, so an old filter never hides connections", async () => {
    await openPicker();
    await userEvent.type(screen.getByLabelText("search connections"), "redis");
    await userEvent.keyboard("{Escape}");
    await userEvent.click(screen.getByRole("button", { name: /demo-postgres/ }));
    expect(screen.getAllByRole("option")).toHaveLength(5);
  });
});

describe("ConnectionPicker autoOpen", () => {
  it("opens immediately when arriving from the command palette, so ⌘K → Consoles lands on the picker", () => {
    render(<ConnectionPicker connections={connections} selected={null} onSelect={jest.fn()} autoOpen />);
    expect(screen.getByRole("dialog", { name: "Choose a connection" })).toBeInTheDocument();
  });

  it("stays closed on a normal visit", () => {
    render(<ConnectionPicker connections={connections} selected={connections[0]} onSelect={jest.fn()} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});

describe("ConnectionPicker layering", () => {
  it("renders into document.body so the toolbar's backdrop-filter cannot render it behind other chrome", () => {
    render(<ConnectionPicker connections={connections} selected={null} onSelect={jest.fn()} autoOpen />);
    expect(document.querySelector(".picker-backdrop")?.parentElement).toBe(document.body);
  });
});
