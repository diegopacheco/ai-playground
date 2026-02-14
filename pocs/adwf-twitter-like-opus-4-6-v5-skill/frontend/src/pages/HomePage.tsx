import { useQuery } from "@tanstack/react-query";
import { getFeed } from "../api/tweets";
import { ComposeBox } from "../components/ComposeBox";
import { TweetCard } from "../components/TweetCard";

export function HomePage() {
  const { data: tweets, isLoading, error } = useQuery({
    queryKey: ["feed"],
    queryFn: getFeed,
  });

  return (
    <div className="max-w-2xl mx-auto bg-white min-h-screen border-x border-gray-200">
      <h2 className="text-xl font-bold p-4 border-b border-gray-200">Home</h2>
      <ComposeBox />
      {isLoading && <p className="p-4 text-gray-500">Loading feed...</p>}
      {error && <p className="p-4 text-red-500">Failed to load feed</p>}
      {tweets && tweets.length === 0 && (
        <p className="p-4 text-gray-500">No tweets yet. Follow some users or post something!</p>
      )}
      {tweets?.map((tweet) => (
        <TweetCard key={tweet.id} tweet={tweet} />
      ))}
    </div>
  );
}
