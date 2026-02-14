import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ComposeBox } from "./ComposeBox";

vi.mock("../api/tweets", () => ({
  createTweet: vi.fn().mockResolvedValue({ id: 1, content: "test" }),
}));

function renderWithProviders() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ComposeBox />
    </QueryClientProvider>
  );
}

describe("ComposeBox", () => {
  it("renders textarea and post button", () => {
    renderWithProviders();
    expect(screen.getByPlaceholderText("What's happening?")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Post" })).toBeInTheDocument();
  });

  it("shows 280 as initial character count", () => {
    renderWithProviders();
    expect(screen.getByText("280")).toBeInTheDocument();
  });

  it("post button is disabled when textarea is empty", () => {
    renderWithProviders();
    expect(screen.getByRole("button", { name: "Post" })).toBeDisabled();
  });

  it("enables post button when text is entered", async () => {
    const user = userEvent.setup();
    renderWithProviders();
    await user.type(screen.getByPlaceholderText("What's happening?"), "Hello");
    expect(screen.getByRole("button", { name: "Post" })).toBeEnabled();
  });

  it("updates character counter as user types", async () => {
    const user = userEvent.setup();
    renderWithProviders();
    await user.type(screen.getByPlaceholderText("What's happening?"), "Hello");
    expect(screen.getByText("275")).toBeInTheDocument();
  });

  it("shows negative count when over 280 chars", async () => {
    const user = userEvent.setup();
    renderWithProviders();
    const longText = "a".repeat(285);
    await user.type(screen.getByPlaceholderText("What's happening?"), longText);
    expect(screen.getByText("-5")).toBeInTheDocument();
  });

  it("disables post button when over 280 chars", async () => {
    const user = userEvent.setup();
    renderWithProviders();
    const longText = "a".repeat(281);
    await user.type(screen.getByPlaceholderText("What's happening?"), longText);
    expect(screen.getByRole("button", { name: "Post" })).toBeDisabled();
  });
});
