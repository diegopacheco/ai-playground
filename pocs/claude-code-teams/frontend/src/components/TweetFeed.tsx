import { Tweet } from "../types";
import TweetCard from "./TweetCard";

interface TweetFeedProps {
  tweets: Tweet[];
  onLike: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function TweetFeed({ tweets, onLike, onDelete }: TweetFeedProps) {
  if (tweets.length === 0) {
    return (
      <div style={{
        padding: "40px 20px",
        textAlign: "center",
        color: "#71767b",
        fontSize: "15px",
      }}>
        No chirps yet. Be the first to chirp!
      </div>
    );
  }

  return (
    <div>
      {tweets.map((tweet) => (
        <TweetCard
          key={tweet.id}
          tweet={tweet}
          onLike={onLike}
          onDelete={onDelete}
        />
      ))}
    </div>
  );
}
