import "./App.css";
import Layout from "./components/Layout";
import Header from "./components/Header";
import TweetCompose from "./components/TweetCompose";
import TweetFeed from "./components/TweetFeed";
import { Tweet } from "./types";
import { useState } from "react";

const sampleTweets: Tweet[] = [
  {
    id: "1",
    username: "diego",
    content: "Just shipped a new feature! Feeling great about the architecture decisions we made.",
    created_at: new Date(Date.now() - 300000).toISOString(),
    likes: 3,
  },
  {
    id: "2",
    username: "rustacean",
    content: "Rust + Actix Web is such a powerful combo for building APIs. Zero cost abstractions FTW!",
    created_at: new Date(Date.now() - 3600000).toISOString(),
    likes: 7,
  },
];

export default function App() {
  const [tweets, setTweets] = useState<Tweet[]>(sampleTweets);

  function handleSubmit(username: string, content: string) {
    const newTweet: Tweet = {
      id: crypto.randomUUID(),
      username,
      content,
      created_at: new Date().toISOString(),
      likes: 0,
    };
    setTweets([newTweet, ...tweets]);
  }

  function handleLike(id: string) {
    setTweets(tweets.map((t) => t.id === id ? { ...t, likes: t.likes + 1 } : t));
  }

  function handleDelete(id: string) {
    setTweets(tweets.filter((t) => t.id !== id));
  }

  return (
    <Layout>
      <Header />
      <TweetCompose onSubmit={handleSubmit} />
      <TweetFeed tweets={tweets} onLike={handleLike} onDelete={handleDelete} />
    </Layout>
  );
}
