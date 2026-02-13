import { useInfiniteQuery } from "@tanstack/react-query";
import { PostCard } from "./PostCard";
import type { Post } from "../types";

interface FeedProps {
  queryKey: string[];
  fetchFn: (page: number) => Promise<Post[]>;
  onAuthorClick?: (userId: number) => void;
  onPostClick?: (postId: number) => void;
}

export function Feed({ queryKey, fetchFn, onAuthorClick, onPostClick }: FeedProps) {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
  } = useInfiniteQuery({
    queryKey,
    queryFn: ({ pageParam }) => fetchFn(pageParam),
    initialPageParam: 1,
    getNextPageParam: (lastPage, allPages) => {
      if (lastPage.length === 0) return undefined;
      return allPages.length + 1;
    },
  });

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="text-center py-8 text-red-500">
        Failed to load posts.
      </div>
    );
  }

  const posts = data?.pages.flatMap((page) => page) ?? [];

  if (posts.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No posts yet. Be the first to post!
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {posts.map((post) => (
        <PostCard
          key={post.id}
          post={post}
          onAuthorClick={onAuthorClick}
          onPostClick={onPostClick}
        />
      ))}
      {hasNextPage && (
        <div className="flex justify-center py-4">
          <button
            onClick={() => fetchNextPage()}
            disabled={isFetchingNextPage}
            className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-6 py-2 rounded-full font-semibold text-sm transition-colors"
          >
            {isFetchingNextPage ? "Loading..." : "Load More"}
          </button>
        </div>
      )}
    </div>
  );
}
