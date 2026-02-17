import { useState, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

const API_URL = "http://localhost:8080/api/tweets";

export default function TweetForm() {
  const [username, setUsername] = useState("");
  const [content, setContent] = useState("");
  const queryClient = useQueryClient();

  useEffect(() => {
    const saved = localStorage.getItem("tweeter_username");
    if (saved) setUsername(saved);
  }, []);

  const mutation = useMutation({
    mutationFn: async (data: { username: string; content: string }) => {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Failed to post tweet");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tweets"] });
      setContent("");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedUsername = username.trim();
    const trimmedContent = content.trim();
    if (!trimmedUsername || !trimmedContent) return;
    localStorage.setItem("tweeter_username", trimmedUsername);
    mutation.mutate({ username: trimmedUsername, content: trimmedContent });
  };

  return (
    <form className="tweet-form" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        className="input-username"
      />
      <textarea
        placeholder="What's happening?"
        value={content}
        onChange={(e) => setContent(e.target.value)}
        maxLength={280}
        className="input-content"
      />
      <div className="form-footer">
        <span className="char-count">{content.length}/280</span>
        <button type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? "Posting..." : "Tweet"}
        </button>
      </div>
    </form>
  );
}
