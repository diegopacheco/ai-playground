import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi } from "vitest";
import { LoginPage } from "../LoginPage";
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

describe("LoginPage", () => {
  it("renders login form", () => {
    const onNavigate = vi.fn();
    renderWithProviders(<LoginPage onNavigate={onNavigate} />);
    expect(screen.getByRole("heading", { name: "Login" })).toBeInTheDocument();
    expect(screen.getByLabelText("Username")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
  });

  it("renders register link", () => {
    const onNavigate = vi.fn();
    renderWithProviders(<LoginPage onNavigate={onNavigate} />);
    expect(screen.getByText("Register")).toBeInTheDocument();
  });

  it("renders login button", () => {
    const onNavigate = vi.fn();
    renderWithProviders(<LoginPage onNavigate={onNavigate} />);
    expect(screen.getByRole("button", { name: "Login" })).toBeInTheDocument();
  });
});
