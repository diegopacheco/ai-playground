import { Link } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { likeTweet, unlikeTweet, deleteTweet } from "../api/tweets";
import { useAuth } from "../hooks/useAuth";
import type { Tweet } from "../types";

interface TweetCardProps {
  tweet: Tweet;
}

export function TweetCard({ tweet }: TweetCardProps) {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const likeMutation = useMutation({
    mutationFn: () => (tweet.liked_by_me ? unlikeTweet(tweet.id) : likeTweet(tweet.id)),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tweets"] });
      queryClient.invalidateQueries({ queryKey: ["feed"] });
      queryClient.invalidateQueries({ queryKey: ["tweet", tweet.id] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteTweet(tweet.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tweets"] });
      queryClient.invalidateQueries({ queryKey: ["feed"] });
    },
  });

  const timeAgo = (dateStr: string) => {
    const diff = Date.now() - new Date(dateStr).getTime();
    const minutes = Math.floor(diff / 60000);
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h`;
    const days = Math.floor(hours / 24);
    return `${days}d`;
  };

  return (
    <div className="border-b border-gray-200 p-4 hover:bg-gray-50">
      <div className="flex justify-between items-start">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <Link
              to={`/profile/${tweet.user_id}`}
              className="font-bold text-gray-900 hover:underline"
            >
              {tweet.display_name}
            </Link>
            <Link
              to={`/profile/${tweet.user_id}`}
              className="text-gray-500"
            >
              @{tweet.username}
            </Link>
            <span className="text-gray-400">·</span>
            <span className="text-gray-400">{timeAgo(tweet.created_at)}</span>
          </div>
          <Link to={`/tweet/${tweet.id}`} className="block mt-1 text-gray-800">
            {tweet.content}
          </Link>
          <div className="flex items-center gap-4 mt-2">
            <button
              onClick={() => likeMutation.mutate()}
              disabled={likeMutation.isPending}
              className={`flex items-center gap-1 text-sm ${
                tweet.liked_by_me ? "text-red-500" : "text-gray-500 hover:text-red-500"
              }`}
            >
              {tweet.liked_by_me ? "\u2665" : "\u2661"} {tweet.like_count}
            </button>
          </div>
        </div>
        {user && user.id === tweet.user_id && (
          <button
            onClick={() => deleteMutation.mutate()}
            disabled={deleteMutation.isPending}
            className="text-gray-400 hover:text-red-500 text-sm ml-2"
          >
            Delete
          </button>
        )}
      </div>
    </div>
  );
}
