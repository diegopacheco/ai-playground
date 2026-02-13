import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect } from "vitest";
import { PostComposer } from "../PostComposer";
import { AuthProvider } from "../../context/AuthContext";

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <AuthProvider>{ui}</AuthProvider>
    </QueryClientProvider>
  );
}

describe("PostComposer", () => {
  it("renders textarea and submit button", () => {
    renderWithProviders(<PostComposer />);
    expect(screen.getByPlaceholderText("What's happening?")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Post" })).toBeInTheDocument();
  });

  it("shows character count", () => {
    renderWithProviders(<PostComposer />);
    expect(screen.getByText("280")).toBeInTheDocument();
  });

  it("updates character count as user types", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PostComposer />);
    const textarea = screen.getByPlaceholderText("What's happening?");
    await user.type(textarea, "Hello");
    expect(screen.getByText("275")).toBeInTheDocument();
  });

  it("disables submit button when empty", () => {
    renderWithProviders(<PostComposer />);
    const button = screen.getByRole("button", { name: "Post" });
    expect(button).toBeDisabled();
  });

  it("enables submit button when text is entered", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PostComposer />);
    const textarea = screen.getByPlaceholderText("What's happening?");
    await user.type(textarea, "Hello world");
    const button = screen.getByRole("button", { name: "Post" });
    expect(button).not.toBeDisabled();
  });
});
