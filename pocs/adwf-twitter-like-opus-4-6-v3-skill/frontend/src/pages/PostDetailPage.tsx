import { useParams } from "@tanstack/react-router";
import { usePost } from "../hooks/usePosts";
import { PostCard } from "../components/PostCard";

export function PostDetailPage() {
  const { postId } = useParams({ strict: false }) as { postId: string };
  const { data: post, isLoading } = usePost(postId);

  if (isLoading) {
    return (
      <div className="flex justify-center p-8">
        <div className="w-8 h-8 border-4 border-[#1DA1F2] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!post) {
    return (
      <div className="p-8 text-center text-gray-500">Post not found</div>
    );
  }

  return (
    <div>
      <div className="border-b border-gray-200 p-4">
        <h2 className="text-xl font-bold text-gray-900">Post</h2>
      </div>
      <PostCard post={post} />
    </div>
  );
}
