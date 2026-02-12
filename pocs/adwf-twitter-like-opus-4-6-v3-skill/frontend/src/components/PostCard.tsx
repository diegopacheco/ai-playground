import { Link } from "@tanstack/react-router";
import { useToggleLike } from "../hooks/usePosts";
import type { PostResponse } from "../types";

function timeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const seconds = Math.floor((now - then) / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.floor(hours / 24);
  return `${days}d`;
}

interface PostCardProps {
  post: PostResponse;
}

export function PostCard({ post }: PostCardProps) {
  const toggleLike = useToggleLike();

  return (
    <div className="border-b border-gray-200 p-4 hover:bg-gray-50 transition-colors">
      <div className="flex gap-3">
        <Link
          to="/profile/$userId"
          params={{ userId: post.user_id }}
          className="shrink-0"
        >
          <div className="w-10 h-10 rounded-full bg-[#1DA1F2] flex items-center justify-center text-white font-bold">
            {post.username[0].toUpperCase()}
          </div>
        </Link>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <Link
              to="/profile/$userId"
              params={{ userId: post.user_id }}
              className="font-bold text-gray-900 hover:underline"
            >
              {post.username}
            </Link>
            <span className="text-gray-500 text-sm">
              {timeAgo(post.created_at)}
            </span>
          </div>
          <Link to="/post/$postId" params={{ postId: post.id }}>
            <p className="text-gray-900 mt-1 whitespace-pre-wrap break-words">
              {post.content}
            </p>
          </Link>
          <div className="mt-3 flex items-center gap-1">
            <button
              onClick={() => toggleLike.mutate(post)}
              className={`flex items-center gap-1 group cursor-pointer ${
                post.liked_by_me
                  ? "text-red-500"
                  : "text-gray-500 hover:text-red-500"
              }`}
            >
              <span className="p-1.5 rounded-full group-hover:bg-red-50 transition-colors">
                {post.liked_by_me ? (
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                  </svg>
                ) : (
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                  </svg>
                )}
              </span>
              <span className="text-sm">{post.likes_count}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
