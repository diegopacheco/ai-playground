import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { AuthContext } from "../context/AuthContext";
import { TweetCard } from "./TweetCard";
import type { Tweet } from "../types";

vi.mock("../api/tweets", () => ({
  likeTweet: vi.fn(),
  unlikeTweet: vi.fn(),
  deleteTweet: vi.fn(),
}));

const mockTweet: Tweet = {
  id: 1,
  user_id: 42,
  content: "This is a test tweet",
  created_at: new Date().toISOString(),
  username: "johndoe",
  display_name: "John Doe",
  like_count: 5,
  liked_by_me: false,
};

function renderTweetCard(tweet: Tweet = mockTweet, userId: number | null = null) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  const authValue = {
    user: userId ? { id: userId, username: "test", email: "t@t.com", display_name: "Test", bio: "", created_at: "" } : null,
    token: userId ? "fake-token" : null,
    setAuth: vi.fn(),
    logout: vi.fn(),
    isAuthenticated: !!userId,
  };

  return render(
    <QueryClientProvider client={queryClient}>
      <AuthContext.Provider value={authValue}>
        <MemoryRouter>
          <TweetCard tweet={tweet} />
        </MemoryRouter>
      </AuthContext.Provider>
    </QueryClientProvider>
  );
}

describe("TweetCard", () => {
  it("renders tweet content", () => {
    renderTweetCard();
    expect(screen.getByText("This is a test tweet")).toBeInTheDocument();
  });

  it("renders author display name", () => {
    renderTweetCard();
    expect(screen.getByText("John Doe")).toBeInTheDocument();
  });

  it("renders author username", () => {
    renderTweetCard();
    expect(screen.getByText("@johndoe")).toBeInTheDocument();
  });

  it("renders like count", () => {
    renderTweetCard();
    expect(screen.getByText(/5/)).toBeInTheDocument();
  });

  it("renders like button with empty heart when not liked", () => {
    renderTweetCard();
    expect(screen.getByRole("button", { name: /5/ })).toBeInTheDocument();
  });

  it("renders filled heart when liked", () => {
    const likedTweet = { ...mockTweet, liked_by_me: true };
    renderTweetCard(likedTweet);
    const likeBtn = screen.getByRole("button", { name: /5/ });
    expect(likeBtn.textContent).toContain("\u2665");
  });

  it("shows delete button when user owns the tweet", () => {
    renderTweetCard(mockTweet, 42);
    expect(screen.getByText("Delete")).toBeInTheDocument();
  });

  it("hides delete button when user does not own the tweet", () => {
    renderTweetCard(mockTweet, 99);
    expect(screen.queryByText("Delete")).not.toBeInTheDocument();
  });
});
