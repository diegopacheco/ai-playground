import { useState } from "react";
import { useCreatePost } from "../hooks/usePosts";

const MAX_CHARS = 280;

export function PostComposer() {
  const [content, setContent] = useState("");
  const createPost = useCreatePost();
  const remaining = MAX_CHARS - content.length;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (content.trim().length === 0 || content.length > MAX_CHARS) return;
    createPost.mutate(content.trim(), {
      onSuccess: () => setContent(""),
    });
  }

  return (
    <form onSubmit={handleSubmit} className="border-b border-gray-200 p-4">
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="What's happening?"
        rows={3}
        className="w-full resize-none outline-none text-lg placeholder-gray-500"
        maxLength={MAX_CHARS}
      />
      <div className="flex items-center justify-between mt-2">
        <span
          className={`text-sm ${
            remaining < 0
              ? "text-red-500"
              : remaining < 20
                ? "text-yellow-500"
                : "text-gray-400"
          }`}
        >
          {remaining}
        </span>
        <button
          type="submit"
          disabled={
            content.trim().length === 0 ||
            content.length > MAX_CHARS ||
            createPost.isPending
          }
          className="bg-[#1DA1F2] text-white px-5 py-2 rounded-full font-bold hover:bg-[#1a91da] disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
        >
          Post
        </button>
      </div>
    </form>
  );
}
