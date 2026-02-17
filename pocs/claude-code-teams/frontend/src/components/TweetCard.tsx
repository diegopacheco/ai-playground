import type { Tweet } from "../types";

interface TweetCardProps {
  tweet: Tweet;
  onLike: (id: string) => void;
  onDelete: (id: string) => void;
}

function formatTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d`;
}

export default function TweetCard({ tweet, onLike, onDelete }: TweetCardProps) {
  return (
    <div style={{
      padding: "16px 20px",
      borderBottom: "1px solid #38444d",
      transition: "background-color 0.2s",
      cursor: "pointer",
    }}
      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = "#1c2b3a"}
      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = "transparent"}
    >
      <div style={{ display: "flex", gap: "12px" }}>
        <div style={{
          width: "40px",
          height: "40px",
          borderRadius: "50%",
          backgroundColor: "#1d9bf0",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontWeight: 700,
          fontSize: "16px",
          color: "#fff",
          flexShrink: 0,
        }}>
          {tweet.username.charAt(0).toUpperCase()}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontWeight: 700, color: "#e7e9ea", fontSize: "15px" }}>
              {tweet.username}
            </span>
            <span style={{ color: "#71767b", fontSize: "14px" }}>
              @{tweet.username}
            </span>
            <span style={{ color: "#71767b", fontSize: "14px" }}>
              ·
            </span>
            <span style={{ color: "#71767b", fontSize: "14px" }}>
              {formatTime(tweet.created_at)}
            </span>
          </div>
          <p style={{
            margin: "4px 0 12px 0",
            color: "#e7e9ea",
            fontSize: "15px",
            lineHeight: 1.5,
            wordBreak: "break-word",
          }}>
            {tweet.content}
          </p>
          <div style={{ display: "flex", gap: "40px" }}>
            <button
              onClick={(e) => { e.stopPropagation(); onLike(tweet.id); }}
              style={{
                background: "none",
                border: "none",
                color: tweet.likes > 0 ? "#f91880" : "#71767b",
                cursor: "pointer",
                padding: "4px 8px",
                borderRadius: "9999px",
                fontSize: "14px",
                display: "flex",
                alignItems: "center",
                gap: "6px",
                transition: "color 0.2s",
              }}
            >
              {tweet.likes > 0 ? "♥" : "♡"} {tweet.likes > 0 ? tweet.likes : ""}
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(tweet.id); }}
              style={{
                background: "none",
                border: "none",
                color: "#71767b",
                cursor: "pointer",
                padding: "4px 8px",
                borderRadius: "9999px",
                fontSize: "14px",
                transition: "color 0.2s",
              }}
              onMouseEnter={(e) => e.currentTarget.style.color = "#f4212e"}
              onMouseLeave={(e) => e.currentTarget.style.color = "#71767b"}
            >
              🗑
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
