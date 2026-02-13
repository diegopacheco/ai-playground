import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi } from "vitest";
import { RegisterPage } from "../RegisterPage";
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

describe("RegisterPage", () => {
  it("renders registration form", () => {
    const onNavigate = vi.fn();
    renderWithProviders(<RegisterPage onNavigate={onNavigate} />);
    expect(screen.getByRole("heading", { name: "Register" })).toBeInTheDocument();
    expect(screen.getByLabelText("Username")).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
  });

  it("renders login link", () => {
    const onNavigate = vi.fn();
    renderWithProviders(<RegisterPage onNavigate={onNavigate} />);
    expect(screen.getByText("Login")).toBeInTheDocument();
  });
});
