import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createPost } from "../api/posts";

const MAX_CHARS = 280;

export function PostComposer() {
  const [content, setContent] = useState("");
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: createPost,
    onSuccess: () => {
      setContent("");
      queryClient.invalidateQueries({ queryKey: ["posts"] });
      queryClient.invalidateQueries({ queryKey: ["feed"] });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (content.trim().length === 0 || content.length > MAX_CHARS) return;
    mutation.mutate({ content: content.trim() });
  };

  const remaining = MAX_CHARS - content.length;

  return (
    <form onSubmit={handleSubmit} className="bg-white border border-gray-200 rounded-lg p-4">
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="What's happening?"
        className="w-full resize-none border-none outline-none text-gray-800 placeholder-gray-400 text-lg min-h-[80px]"
        maxLength={MAX_CHARS}
      />
      <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-100">
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
            mutation.isPending
          }
          className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-4 py-2 rounded-full font-semibold text-sm transition-colors"
        >
          {mutation.isPending ? "Posting..." : "Post"}
        </button>
      </div>
      {mutation.isError && (
        <p className="text-red-500 text-sm mt-2">
          Failed to create post. Try again.
        </p>
      )}
    </form>
  );
}
