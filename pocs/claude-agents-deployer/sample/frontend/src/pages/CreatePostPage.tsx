import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { useCreatePost } from "../hooks/useApi";

export default function CreatePostPage() {
  const navigate = useNavigate();
  const createPost = useCreatePost();
  const [title, setTitle] = useState("");
  const [author, setAuthor] = useState("");
  const [content, setContent] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!title.trim() || !content.trim() || !author.trim()) return;
    createPost.mutate(
      { title, content, author },
      {
        onSuccess: (post) => {
          navigate({
            to: "/posts/$postId",
            params: { postId: String(post.id) },
          });
        },
      }
    );
  }

  return (
    <div>
      <h1 className="text-3xl font-bold text-gray-900 mb-8">
        Create New Post
      </h1>
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
            Author
          </label>
          <input
            type="text"
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
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
            disabled={createPost.isPending}
            className="bg-indigo-600 text-white px-6 py-2.5 rounded-lg hover:bg-indigo-700 transition-colors font-medium disabled:opacity-50"
          >
            {createPost.isPending ? "Creating..." : "Create Post"}
          </button>
          <button
            type="button"
            onClick={() => navigate({ to: "/" })}
            className="bg-gray-200 text-gray-700 px-6 py-2.5 rounded-lg hover:bg-gray-300 transition-colors font-medium"
          >
            Cancel
          </button>
        </div>
        {createPost.isError && (
          <p className="text-red-600">
            Failed to create post. Please try again.
          </p>
        )}
      </form>
    </div>
  );
}
