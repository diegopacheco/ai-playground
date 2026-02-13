import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi } from "vitest";
import { Navbar } from "../Navbar";
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

describe("Navbar", () => {
  it("renders the app name", () => {
    const onNavigate = vi.fn();
    renderWithProviders(<Navbar onNavigate={onNavigate} />);
    expect(screen.getByText("Chirp")).toBeInTheDocument();
  });
});
