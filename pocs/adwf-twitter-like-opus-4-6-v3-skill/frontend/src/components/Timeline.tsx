import { PostCard } from "./PostCard";
import type { PostResponse } from "../types";

interface TimelineProps {
  posts: PostResponse[];
  isLoading: boolean;
}

export function Timeline({ posts, isLoading }: TimelineProps) {
  if (isLoading) {
    return (
      <div className="flex justify-center p-8">
        <div className="w-8 h-8 border-4 border-[#1DA1F2] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (posts.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        <p className="text-lg">No posts yet</p>
        <p className="text-sm mt-1">Follow some users or create a post!</p>
      </div>
    );
  }

  return (
    <div>
      {posts.map((post) => (
        <PostCard key={post.id} post={post} />
      ))}
    </div>
  );
}
