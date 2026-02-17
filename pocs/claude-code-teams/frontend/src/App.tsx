import Layout from "./components/Layout";
import Header from "./components/Header";
import TweetCompose from "./components/TweetCompose";
import TweetFeed from "./components/TweetFeed";
import { useTweets, useCreateTweet, useDeleteTweet, useLikeTweet } from "./hooks";

export default function App() {
  const { data: tweets = [], isLoading } = useTweets();
  const createTweet = useCreateTweet();
  const deleteTweet = useDeleteTweet();
  const likeTweet = useLikeTweet();

  const handleCreateTweet = (username: string, content: string) => {
    createTweet.mutate({ username, content });
  };

  const handleDeleteTweet = (id: string) => {
    deleteTweet.mutate(id);
  };

  const handleLikeTweet = (id: string) => {
    likeTweet.mutate(id);
  };

  if (isLoading) {
    return (
      <Layout>
        <Header />
        <div style={{ padding: "40px 20px", textAlign: "center", color: "#71767b" }}>
          Loading...
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <Header />
      <TweetCompose onSubmit={handleCreateTweet} />
      <TweetFeed tweets={tweets} onLike={handleLikeTweet} onDelete={handleDeleteTweet} />
    </Layout>
  );
}
