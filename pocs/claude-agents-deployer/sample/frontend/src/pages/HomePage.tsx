import { Link } from "@tanstack/react-router";
import { usePosts } from "../hooks/useApi";

export default function HomePage() {
  const { data: posts, isLoading, error } = usePosts();

  if (isLoading) {
    return (
      <div className="flex justify-center py-20">
        <div className="text-gray-500 text-lg">Loading posts...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-red-700">
        Failed to load posts. Make sure the API server is running.
      </div>
    );
  }

  if (!posts || posts.length === 0) {
    return (
      <div className="text-center py-20">
        <h2 className="text-2xl font-semibold text-gray-700 mb-4">
          No posts yet
        </h2>
        <Link
          to="/posts/create"
          className="inline-block bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors"
        >
          Create your first post
        </Link>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold text-gray-900">All Posts</h1>
        <Link
          to="/posts/create"
          className="bg-indigo-600 text-white px-5 py-2.5 rounded-lg hover:bg-indigo-700 transition-colors font-medium"
        >
          New Post
        </Link>
      </div>
      <div className="space-y-6">
        {posts.map((post) => (
          <Link
            key={post.id}
            to="/posts/$postId"
            params={{ postId: String(post.id) }}
            className="block bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
          >
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              {post.title}
            </h2>
            <div className="flex items-center gap-3 text-sm text-gray-500 mb-3">
              <span className="font-medium text-indigo-600">{post.author}</span>
              <span>-</span>
              <span>{new Date(post.createdAt).toLocaleDateString()}</span>
            </div>
            <p className="text-gray-600 leading-relaxed line-clamp-3">{post.content}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
