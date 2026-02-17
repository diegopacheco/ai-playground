import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

const API_URL = "http://localhost:8080/api/tweets";

interface Tweet {
  id: string;
  username: string;
  content: string;
  created_at: string;
}

function timeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const seconds = Math.floor((now - then) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function TweetList() {
  const queryClient = useQueryClient();
  const currentUser = localStorage.getItem("tweeter_username") || "";

  const { data: tweets, isLoading, error } = useQuery<Tweet[]>({
    queryKey: ["tweets"],
    queryFn: async () => {
      const res = await fetch(API_URL);
      if (!res.ok) throw new Error("Failed to fetch tweets");
      return res.json();
    },
    refetchInterval: 10000,
  });

  const deleteMutation = useMutation({
    mutationFn: async (tweetId: string) => {
      const res = await fetch(
        `${API_URL}/${tweetId}?username=${encodeURIComponent(currentUser)}`,
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error("Failed to delete tweet");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tweets"] });
    },
  });

  if (isLoading) return <div className="loading">Loading tweets...</div>;
  if (error) return <div className="error">Failed to load tweets</div>;

  return (
    <div className="tweet-list">
      {tweets && tweets.length === 0 && (
        <div className="empty">No tweets yet. Be the first!</div>
      )}
      {tweets?.map((tweet) => (
        <div key={tweet.id} className="tweet-card">
          <div className="tweet-header">
            <span className="tweet-username">@{tweet.username}</span>
            <span className="tweet-time">{timeAgo(tweet.created_at)}</span>
          </div>
          <div className="tweet-content">{tweet.content}</div>
          {tweet.username === currentUser && (
            <button
              className="delete-btn"
              onClick={() => deleteMutation.mutate(tweet.id)}
              disabled={deleteMutation.isPending}
            >
              Delete
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
