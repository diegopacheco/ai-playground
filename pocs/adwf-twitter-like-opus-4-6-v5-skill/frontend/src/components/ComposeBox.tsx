import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createTweet } from "../api/tweets";

const MAX_CHARS = 280;

export function ComposeBox() {
  const [content, setContent] = useState("");
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: () => createTweet(content),
    onSuccess: () => {
      setContent("");
      queryClient.invalidateQueries({ queryKey: ["feed"] });
    },
  });

  const remaining = MAX_CHARS - content.length;
  const isOverLimit = remaining < 0;

  return (
    <div className="border-b border-gray-200 p-4">
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="What's happening?"
        rows={3}
        className="w-full resize-none border-none outline-none text-lg placeholder-gray-400"
      />
      <div className="flex items-center justify-between mt-2">
        <span className={`text-sm ${isOverLimit ? "text-red-500" : "text-gray-400"}`}>
          {remaining}
        </span>
        <button
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending || content.trim().length === 0 || isOverLimit}
          className="bg-blue-600 text-white px-5 py-2 rounded-full font-bold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Post
        </button>
      </div>
    </div>
  );
}
