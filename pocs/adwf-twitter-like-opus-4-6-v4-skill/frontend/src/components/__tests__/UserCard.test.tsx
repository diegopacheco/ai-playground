import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect } from "vitest";
import { UserCard } from "../UserCard";
import { AuthProvider } from "../../context/AuthContext";
import type { User } from "../../types";

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

const mockUser: User = {
  id: 2,
  username: "otheruser",
  email: "other@test.com",
  created_at: new Date().toISOString(),
};

describe("UserCard", () => {
  it("renders user username", () => {
    renderWithProviders(<UserCard targetUser={mockUser} />);
    expect(screen.getByText("@otheruser")).toBeInTheDocument();
  });

  it("renders follow button for other users", () => {
    renderWithProviders(<UserCard targetUser={mockUser} isFollowing={false} />);
    expect(screen.getByRole("button", { name: "Follow" })).toBeInTheDocument();
  });

  it("renders unfollow button when following", () => {
    renderWithProviders(<UserCard targetUser={mockUser} isFollowing={true} />);
    expect(screen.getByRole("button", { name: "Unfollow" })).toBeInTheDocument();
  });
});
