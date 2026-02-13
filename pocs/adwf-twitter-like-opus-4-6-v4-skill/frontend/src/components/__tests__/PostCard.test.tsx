import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect } from "vitest";
import { PostCard } from "../PostCard";
import { AuthProvider } from "../../context/AuthContext";
import type { Post } from "../../types";

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

const mockPost: Post = {
  id: 1,
  user_id: 1,
  content: "Test post content",
  created_at: new Date().toISOString(),
  username: "testuser",
  like_count: 5,
  liked_by_me: false,
};

describe("PostCard", () => {
  it("renders post content", () => {
    renderWithProviders(<PostCard post={mockPost} />);
    expect(screen.getByText("Test post content")).toBeInTheDocument();
  });

  it("renders author username", () => {
    renderWithProviders(<PostCard post={mockPost} />);
    expect(screen.getByText("@testuser")).toBeInTheDocument();
  });

  it("renders like count", () => {
    renderWithProviders(<PostCard post={mockPost} />);
    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("shows empty heart when not liked", () => {
    renderWithProviders(<PostCard post={mockPost} />);
    expect(screen.getByText("\u2661")).toBeInTheDocument();
  });

  it("shows filled heart when liked", () => {
    const likedPost = { ...mockPost, liked_by_me: true };
    renderWithProviders(<PostCard post={likedPost} />);
    expect(screen.getByText("\u2665")).toBeInTheDocument();
  });
});
