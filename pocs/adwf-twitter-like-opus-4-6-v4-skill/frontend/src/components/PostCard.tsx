import { useMutation, useQueryClient } from "@tanstack/react-query";
import { likePost, unlikePost } from "../api/likes";
import type { Post } from "../types";

interface PostCardProps {
  post: Post;
  onAuthorClick?: (userId: number) => void;
  onPostClick?: (postId: number) => void;
}

export function PostCard({ post, onAuthorClick, onPostClick }: PostCardProps) {
  const queryClient = useQueryClient();

  const likeMutation = useMutation({
    mutationFn: () => likePost(post.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["posts"] });
      queryClient.invalidateQueries({ queryKey: ["feed"] });
      queryClient.invalidateQueries({ queryKey: ["post", post.id] });
    },
  });

  const unlikeMutation = useMutation({
    mutationFn: () => unlikePost(post.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["posts"] });
      queryClient.invalidateQueries({ queryKey: ["feed"] });
      queryClient.invalidateQueries({ queryKey: ["post", post.id] });
    },
  });

  const handleLikeToggle = () => {
    if (post.liked_by_me) {
      unlikeMutation.mutate();
    } else {
      likeMutation.mutate();
    }
  };

  const timeAgo = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h`;
    const days = Math.floor(hours / 24);
    return `${days}d`;
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
      <div className="flex items-start gap-3">
        <div
          className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold cursor-pointer"
          onClick={() => onAuthorClick?.(post.user_id)}
        >
          {(post.username ?? "?")[0].toUpperCase()}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <button
              onClick={() => onAuthorClick?.(post.user_id)}
              className="font-semibold text-gray-900 hover:underline"
            >
              @{post.username ?? "unknown"}
            </button>
            <span className="text-gray-500 text-sm">
              {timeAgo(post.created_at)}
            </span>
          </div>
          <p
            className="mt-1 text-gray-800 whitespace-pre-wrap break-words cursor-pointer"
            onClick={() => onPostClick?.(post.id)}
          >
            {post.content}
          </p>
          <div className="mt-2 flex items-center gap-4">
            <button
              onClick={handleLikeToggle}
              disabled={likeMutation.isPending || unlikeMutation.isPending}
              className={`flex items-center gap-1 text-sm transition-colors ${
                post.liked_by_me
                  ? "text-red-500 hover:text-red-600"
                  : "text-gray-500 hover:text-red-500"
              }`}
            >
              <span>{post.liked_by_me ? "\u2665" : "\u2661"}</span>
              <span>{post.like_count ?? 0}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
