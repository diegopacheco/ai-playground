import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AskAi } from "./AskAi";
import { api } from "@lib/api";
import { ApiError } from "@lib/types";

jest.mock("@lib/api");
const mocked = api as jest.Mocked<typeof api>;

const suggestion = (overrides = {}) => ({
  statement: "SELECT count(*) FROM customers",
  cli: "claude",
  model: "sonnet",
  readOnlyOk: true,
  denialReason: null,
  ...overrides
});

beforeEach(() => {
  jest.clearAllMocks();
  mocked.aiSettings.mockResolvedValue({
    cli: "claude", label: "claude -p", model: "sonnet", usingDefault: false, installed: true
  });
  mocked.aiQuery.mockResolvedValue(suggestion());
});

async function openAndAsk(onUse = jest.fn(), text = "count the customers") {
  render(<AskAi connectionId={7} onUse={onUse} />);
  await userEvent.click(screen.getByRole("button", { name: /ask ai/i }));
  await userEvent.type(screen.getByLabelText("describe the query you want"), text);
  await userEvent.click(screen.getByRole("button", { name: "Ask" }));
  return onUse;
}

describe("AskAi", () => {
  it("shows which CLI and model will be used, so nobody is surprised where their schema went", async () => {
    render(<AskAi connectionId={7} onUse={jest.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /ask ai/i }));
    expect(await screen.findByText("claude")).toBeInTheDocument();
    expect(screen.getByText("sonnet")).toBeInTheDocument();
  });

  it("sends the prompt for the current connection", async () => {
    await openAndAsk();
    await waitFor(() => expect(mocked.aiQuery).toHaveBeenCalledWith(7, "count the customers"));
  });

  it("shows the suggested statement without running it", async () => {
    await openAndAsk();
    expect(await screen.findByText("SELECT count(*) FROM customers")).toBeInTheDocument();
    expect(mocked.query).not.toHaveBeenCalled();
  });

  it("only loads the statement into the editor when the user chooses to", async () => {
    const onUse = await openAndAsk();
    expect(onUse).not.toHaveBeenCalled();
    await userEvent.click(await screen.findByRole("button", { name: "use this query" }));
    expect(onUse).toHaveBeenCalledWith("SELECT count(*) FROM customers");
  });

  it("refuses to load a suggestion the guard rejected, so a model cannot smuggle a write into the editor", async () => {
    mocked.aiQuery.mockResolvedValue(
      suggestion({ statement: "DELETE FROM customers", readOnlyOk: false, denialReason: "only read statements are allowed, found: DELETE" })
    );
    const onUse = await openAndAsk();
    expect(await screen.findByText("rejected")).toBeInTheDocument();
    expect(screen.getByText(/only read statements are allowed/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "use this query" })).toBeDisabled();
    expect(onUse).not.toHaveBeenCalled();
  });

  it("marks an acceptable suggestion as read-only", async () => {
    await openAndAsk();
    expect(await screen.findByText("read only")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "use this query" })).toBeEnabled();
  });

  it("reports a missing CLI instead of failing silently", async () => {
    mocked.aiQuery.mockRejectedValue(new ApiError("agy was not found on PATH", 500, false));
    await openAndAsk();
    expect(await screen.findByRole("alert")).toHaveTextContent("agy was not found on PATH");
  });

  it("does not ask with an empty prompt", async () => {
    render(<AskAi connectionId={7} onUse={jest.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /ask ai/i }));
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    expect(mocked.aiQuery).not.toHaveBeenCalled();
  });

  it("closes on Escape", async () => {
    render(<AskAi connectionId={7} onUse={jest.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /ask ai/i }));
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renders into document.body so toolbar chrome cannot trap it behind other panels", async () => {
    render(<AskAi connectionId={7} onUse={jest.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /ask ai/i }));
    expect(document.querySelector(".askai-backdrop")?.parentElement).toBe(document.body);
  });
});
