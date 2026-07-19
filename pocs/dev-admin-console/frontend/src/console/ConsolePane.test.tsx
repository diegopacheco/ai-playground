import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConsolePane } from "./ConsolePane";
import { api } from "@lib/api";
import { ApiError, type Connection } from "@lib/types";

jest.mock("@lib/api");

const mocked = api as jest.Mocked<typeof api>;

const connection: Connection = {
  id: 7,
  projectId: 1,
  name: "demo-postgres",
  kind: "postgres",
  host: "localhost",
  port: 5432,
  database: "shop",
  keyspace: "public",
  datacenter: null,
  username: "console_reader",
  hasPassword: true,
  createdBy: "admin"
};

const schema = [
  { name: "customers", kind: "table", detail: "BASE TABLE", children: [{ name: "email", kind: "column", detail: "text" }] }
];

const page = (overrides: Partial<Awaited<ReturnType<typeof api.query>>> = {}) => ({
  queryId: "q-1",
  columns: ["id"],
  rows: [{ id: "1" }, { id: "2" }],
  elapsedMs: 5,
  pageNumber: 1,
  nextCursor: "2",
  hasMore: true,
  totalRows: null,
  ...overrides
});

beforeEach(() => {
  jest.clearAllMocks();
  mocked.schema.mockResolvedValue(schema);
  mocked.history.mockResolvedValue([]);
  mocked.query.mockResolvedValue(page());
});

describe("ConsolePane", () => {
  it("loads the schema for the connection and lists it in the left panel", async () => {
    render(<ConsolePane connection={connection} />);
    expect(await screen.findByText("customers")).toBeInTheDocument();
    expect(mocked.schema).toHaveBeenCalledWith(7);
  });

  it("runs the statement when Run is pressed", async () => {
    render(<ConsolePane connection={connection} />);
    await screen.findByText("customers");
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(mocked.query).toHaveBeenCalled());
    expect(mocked.query.mock.calls[0][0]).toBe(7);
  });

  it("shows the denial and no grid when the backend rejects a write, so nothing looks like it succeeded", async () => {
    mocked.query.mockRejectedValue(new ApiError("only read statements are allowed, found: DELETE", 400, true));
    render(<ConsolePane connection={connection} />);
    await screen.findByText("customers");
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("only read statements are allowed");
    expect(screen.getByText("read only")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("distinguishes a connection error from a read-only denial, because the fixes differ", async () => {
    mocked.query.mockRejectedValue(new ApiError("connection refused", 500, false));
    render(<ConsolePane connection={connection} />);
    await screen.findByText("customers");
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("connection refused");
    expect(screen.getByText("error")).toBeInTheDocument();
    expect(screen.queryByText("read only")).not.toBeInTheDocument();
  });

  it("advances to the next page using the cursor the engine returned, keeping the same queryId", async () => {
    render(<ConsolePane connection={connection} />);
    await screen.findByText("customers");
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("table");
    mocked.query.mockResolvedValue(page({ pageNumber: 2, rows: [{ id: "3" }], nextCursor: null, hasMore: false }));
    await userEvent.click(screen.getByRole("button", { name: "Next page" }));
    await waitFor(() => {
      const last = mocked.query.mock.calls.at(-1)!;
      expect(last[2]).toMatchObject({ cursor: "2", pageNumber: 2, queryId: "q-1" });
    });
  });

  it("disables Next once the engine reports there are no more rows", async () => {
    mocked.query.mockResolvedValue(page({ hasMore: false, nextCursor: null }));
    render(<ConsolePane connection={connection} />);
    await screen.findByText("customers");
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(await screen.findByRole("button", { name: "Next page" })).toBeDisabled();
  });

  it("restarts at page one when Run is pressed again, so a new statement never inherits an old cursor", async () => {
    render(<ConsolePane connection={connection} />);
    await screen.findByText("customers");
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("table");
    await userEvent.click(screen.getByRole("button", { name: "Next page" }));
    await waitFor(() => expect(mocked.query).toHaveBeenCalledTimes(2));
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => {
      const last = mocked.query.mock.calls.at(-1)!;
      expect(last[2]).toMatchObject({ pageNumber: 1 });
      expect(last[2]?.cursor).toBeUndefined();
    });
  });

  it("shows the connection target so an operator can tell prod from staging at a glance", async () => {
    render(<ConsolePane connection={connection} />);
    expect(await screen.findByText("localhost:5432 / shop / public")).toBeInTheDocument();
  });

  it("surfaces a schema failure instead of rendering an empty tree that looks like an empty database", async () => {
    mocked.schema.mockRejectedValue(new ApiError("connection refused", 500, false));
    render(<ConsolePane connection={connection} />);
    expect(await screen.findByText("connection refused")).toBeInTheDocument();
  });

  it("seeds the editor with a sample statement built from the real schema", async () => {
    render(<ConsolePane connection={connection} />);
    await screen.findByText("customers");
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(mocked.query.mock.calls[0][1]).toBe("SELECT * FROM customers"));
  });

  it("offers a row count only for SQL engines, since counting a Kafka topic is not a count query", async () => {
    render(<ConsolePane connection={connection} />);
    await screen.findByText("customers");
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(await screen.findByRole("button", { name: "count rows" })).toBeInTheDocument();

    mocked.schema.mockResolvedValue([{ name: "orders.events", kind: "topic", detail: "3 partitions" }]);
    render(<ConsolePane connection={{ ...connection, id: 8, kind: "kafka", port: 9092, database: null, keyspace: null }} />);
    await screen.findByText("orders.events");
  });
});
