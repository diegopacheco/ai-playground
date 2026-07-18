import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CommandPalette, { DESTINATIONS } from "./CommandPalette";

async function open(onNavigate = jest.fn()) {
  render(<CommandPalette onNavigate={onNavigate} />);
  await userEvent.keyboard("{Meta>}k{/Meta}");
  return onNavigate;
}

describe("CommandPalette", () => {
  it("stays hidden until the shortcut is pressed", () => {
    render(<CommandPalette onNavigate={jest.fn()} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("opens on CMD+K", async () => {
    await open();
    expect(screen.getByRole("dialog", { name: "Go to page" })).toBeInTheDocument();
  });

  it("opens on Ctrl+K so the shortcut works off macOS", async () => {
    render(<CommandPalette onNavigate={jest.fn()} />);
    await userEvent.keyboard("{Control>}k{/Control}");
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("lists every page without needing to scroll", async () => {
    await open();
    expect(screen.getAllByRole("option")).toHaveLength(DESTINATIONS.length);
  });

  it("focuses the search box so the user can type straight away", async () => {
    await open();
    expect(screen.getByLabelText("go to page")).toHaveFocus();
  });

  it("navigates to the top match on Enter", async () => {
    const onNavigate = await open();
    await userEvent.type(screen.getByLabelText("go to page"), "audit");
    await userEvent.keyboard("{Enter}");
    expect(onNavigate).toHaveBeenCalledWith("/audit-trail");
  });

  it("ranks a name match above a keyword match, so typing a page name goes there", async () => {
    const onNavigate = await open();
    await userEvent.type(screen.getByLabelText("go to page"), "users");
    await userEvent.keyboard("{Enter}");
    expect(onNavigate).toHaveBeenCalledWith("/users");
  });

  it("finds a page by what it does, not just its name", async () => {
    const onNavigate = await open();
    await userEvent.type(screen.getByLabelText("go to page"), "openapi");
    await userEvent.keyboard("{Enter}");
    expect(onNavigate).toHaveBeenCalledWith("/swagger");
  });

  it("moves the highlight forward with the right arrow", async () => {
    const onNavigate = await open();
    await userEvent.keyboard("{ArrowRight}{Enter}");
    expect(onNavigate).toHaveBeenCalledWith(DESTINATIONS[1].href);
  });

  it("wraps backwards past the first entry", async () => {
    const onNavigate = await open();
    await userEvent.keyboard("{ArrowLeft}{Enter}");
    expect(onNavigate).toHaveBeenCalledWith(DESTINATIONS.at(-1)!.href);
  });

  it("navigates on click", async () => {
    const onNavigate = await open();
    await userEvent.click(screen.getByText("Projects"));
    expect(onNavigate).toHaveBeenCalledWith("/projects");
  });

  it("closes on Escape without navigating", async () => {
    const onNavigate = await open();
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it("toggles shut when the shortcut is pressed again", async () => {
    await open();
    await userEvent.keyboard("{Meta>}k{/Meta}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("says so when nothing matches instead of navigating somewhere unexpected", async () => {
    const onNavigate = await open();
    await userEvent.type(screen.getByLabelText("go to page"), "kubernetes");
    expect(screen.getByText(/no page matches/)).toBeInTheDocument();
    await userEvent.keyboard("{Enter}");
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it("locks background scrolling while open and restores it after", async () => {
    await open();
    expect(document.body.style.overflow).toBe("hidden");
    await userEvent.keyboard("{Escape}");
    expect(document.body.style.overflow).not.toBe("hidden");
  });

  it("clears the previous search when reopened", async () => {
    await open();
    await userEvent.type(screen.getByLabelText("go to page"), "audit");
    await userEvent.keyboard("{Escape}");
    await userEvent.keyboard("{Meta>}k{/Meta}");
    expect(screen.getAllByRole("option")).toHaveLength(DESTINATIONS.length);
  });
});
