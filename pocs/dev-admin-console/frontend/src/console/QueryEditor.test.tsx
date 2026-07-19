import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryEditor } from "./QueryEditor";
import { postgres } from "@engines/postgres";

function setup(onRun = jest.fn(), onChange = jest.fn()) {
  render(
    <QueryEditor
      engine={postgres}
      value="SELECT 1"
      completions={["SELECT", "customers"]}
      onChange={onChange}
      onRun={onRun}
    />
  );
  const content = screen.getByTestId("query-editor").querySelector(".cm-content") as HTMLElement;
  return { content, onRun, onChange };
}

describe("QueryEditor", () => {
  it("renders the statement it was given", () => {
    const { content } = setup();
    expect(content.textContent).toContain("SELECT 1");
  });

  it("runs the query on CMD+Enter, which is the documented shortcut", async () => {
    const { content, onRun } = setup();
    content.focus();
    await userEvent.keyboard("{Meta>}{Enter}{/Meta}");
    expect(onRun).toHaveBeenCalledTimes(1);
  });

  it("also runs on Ctrl+Enter so the shortcut works off macOS", async () => {
    const { content, onRun } = setup();
    content.focus();
    await userEvent.keyboard("{Control>}{Enter}{/Control}");
    expect(onRun).toHaveBeenCalledTimes(1);
  });

  it("does not run on plain Enter, because Enter must insert a newline in a multi-line statement", async () => {
    const { content, onRun } = setup();
    content.focus();
    await userEvent.keyboard("{Enter}");
    expect(onRun).not.toHaveBeenCalled();
  });

  it("reports edits so the parent always holds the statement that will be executed", async () => {
    const { content, onChange } = setup();
    content.focus();
    await userEvent.type(content, "0");
    expect(onChange).toHaveBeenCalled();
    expect(onChange.mock.calls.at(-1)![0]).toContain("SELECT");
  });
});
