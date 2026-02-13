import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getPost } from "../api/posts";
import { likePost, unlikePost } from "../api/likes";

interface PostDetailPageProps {
  postId: number;
  onNavigate: (page: string, params?: Record<string, string>) => void;
}

export function PostDetailPage({ postId, onNavigate }: PostDetailPageProps) {
  const queryClient = useQueryClient();

  const { data: post, isLoading, isError } = useQuery({
    queryKey: ["post", postId],
    queryFn: () => getPost(postId),
  });

  const likeMutation = useMutation({
    mutationFn: () => likePost(postId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["post", postId] });
    },
  });

  const unlikeMutation = useMutation({
    mutationFn: () => unlikePost(postId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["post", postId] });
    },
  });

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (isError || !post) {
    return (
      <div className="text-center py-12 text-red-500">
        Post not found.
      </div>
    );
  }

  const handleLikeToggle = () => {
    if (post.liked_by_me) {
      unlikeMutation.mutate();
    } else {
      likeMutation.mutate();
    }
  };

  return (
    <div className="max-w-2xl mx-auto px-4 py-6">
      <button
        onClick={() => onNavigate("home")}
        className="text-blue-500 hover:underline mb-4 inline-block"
      >
        &larr; Back to feed
      </button>
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <div
            className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center text-white text-lg font-bold cursor-pointer"
            onClick={() =>
              onNavigate("profile", { userId: String(post.user_id) })
            }
          >
            {(post.username ?? "?")[0].toUpperCase()}
          </div>
          <div>
            <button
              onClick={() =>
                onNavigate("profile", { userId: String(post.user_id) })
              }
              className="font-semibold text-gray-900 hover:underline"
            >
              @{post.username ?? "unknown"}
            </button>
            <p className="text-gray-500 text-sm">
              {new Date(post.created_at).toLocaleString()}
            </p>
          </div>
        </div>
        <p className="text-gray-800 text-lg whitespace-pre-wrap break-words mb-4">
          {post.content}
        </p>
        <div className="border-t border-gray-200 pt-3 flex items-center gap-4">
          <button
            onClick={handleLikeToggle}
            disabled={likeMutation.isPending || unlikeMutation.isPending}
            className={`flex items-center gap-2 text-lg transition-colors ${
              post.liked_by_me
                ? "text-red-500 hover:text-red-600"
                : "text-gray-500 hover:text-red-500"
            }`}
          >
            <span>{post.liked_by_me ? "\u2665" : "\u2661"}</span>
            <span className="text-base">{post.like_count ?? 0} likes</span>
          </button>
        </div>
      </div>
    </div>
  );
}
