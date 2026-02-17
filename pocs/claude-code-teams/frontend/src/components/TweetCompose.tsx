import { useState } from "react";

interface TweetComposeProps {
  onSubmit: (username: string, content: string) => void;
}

export default function TweetCompose({ onSubmit }: TweetComposeProps) {
  const [username, setUsername] = useState("");
  const [content, setContent] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!username.trim() || !content.trim()) return;
    onSubmit(username.trim(), content.trim());
    setContent("");
  }

  const canSubmit = username.trim().length > 0 && content.trim().length > 0;

  return (
    <form onSubmit={handleSubmit} style={{
      padding: "16px 20px",
      borderBottom: "1px solid #38444d",
    }}>
      <input
        type="text"
        placeholder="Username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        style={{
          width: "100%",
          padding: "10px 12px",
          marginBottom: "12px",
          backgroundColor: "#273340",
          border: "1px solid #38444d",
          borderRadius: "8px",
          color: "#e7e9ea",
          fontSize: "15px",
          outline: "none",
          boxSizing: "border-box",
        }}
        onFocus={(e) => e.currentTarget.style.borderColor = "#1d9bf0"}
        onBlur={(e) => e.currentTarget.style.borderColor = "#38444d"}
      />
      <textarea
        placeholder="What's happening?"
        value={content}
        onChange={(e) => setContent(e.target.value)}
        rows={3}
        style={{
          width: "100%",
          padding: "10px 12px",
          backgroundColor: "#273340",
          border: "1px solid #38444d",
          borderRadius: "8px",
          color: "#e7e9ea",
          fontSize: "15px",
          resize: "vertical",
          outline: "none",
          fontFamily: "inherit",
          boxSizing: "border-box",
        }}
        onFocus={(e) => e.currentTarget.style.borderColor = "#1d9bf0"}
        onBlur={(e) => e.currentTarget.style.borderColor = "#38444d"}
      />
      <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "12px" }}>
        <button
          type="submit"
          disabled={!canSubmit}
          style={{
            backgroundColor: canSubmit ? "#1d9bf0" : "#0e4f82",
            color: canSubmit ? "#fff" : "#808080",
            border: "none",
            borderRadius: "9999px",
            padding: "10px 24px",
            fontSize: "15px",
            fontWeight: 700,
            cursor: canSubmit ? "pointer" : "not-allowed",
            transition: "background-color 0.2s",
          }}
          onMouseEnter={(e) => { if (canSubmit) e.currentTarget.style.backgroundColor = "#1a8cd8"; }}
          onMouseLeave={(e) => { if (canSubmit) e.currentTarget.style.backgroundColor = "#1d9bf0"; }}
        >
          Chirp
        </button>
      </div>
    </form>
  );
}
