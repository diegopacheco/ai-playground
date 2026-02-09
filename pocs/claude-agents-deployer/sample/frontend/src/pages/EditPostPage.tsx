import { useNavigate } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { usePost, useUpdatePost } from "../hooks/useApi";

export default function EditPostPage({ postId }: { postId: string }) {
  const navigate = useNavigate();
  const { data: post, isLoading } = usePost(postId);
  const updatePost = useUpdatePost(postId);

  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");

  useEffect(() => {
    if (post) {
      setTitle(post.title);
      setContent(post.content);
    }
  }, [post]);

  if (isLoading) {
    return (
      <div className="flex justify-center py-20">
        <div className="text-gray-500 text-lg">Loading post...</div>
      </div>
    );
  }

  if (!post) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-red-700">
        Post not found.
      </div>
    );
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!title.trim() || !content.trim()) return;
    updatePost.mutate(
      { title, content },
      {
        onSuccess: () => {
          navigate({
            to: "/posts/$postId",
            params: { postId },
          });
        },
      }
    );
  }

  return (
    <div>
      <h1 className="text-3xl font-bold text-gray-900 mb-8">Edit Post</h1>
      <form
        onSubmit={handleSubmit}
        className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 space-y-6"
      >
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Title
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Content
          </label>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            rows={12}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            required
          />
        </div>
        <div className="flex gap-3">
          <button
            type="submit"
            disabled={updatePost.isPending}
            className="bg-indigo-600 text-white px-6 py-2.5 rounded-lg hover:bg-indigo-700 transition-colors font-medium disabled:opacity-50"
          >
            {updatePost.isPending ? "Saving..." : "Save Changes"}
          </button>
          <button
            type="button"
            onClick={() =>
              navigate({
                to: "/posts/$postId",
                params: { postId },
              })
            }
            className="bg-gray-200 text-gray-700 px-6 py-2.5 rounded-lg hover:bg-gray-300 transition-colors font-medium"
          >
            Cancel
          </button>
        </div>
        {updatePost.isError && (
          <p className="text-red-600">
            Failed to update post. Please try again.
          </p>
        )}
      </form>
    </div>
  );
}
