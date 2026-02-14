import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getTweet } from "../api/tweets";
import { TweetCard } from "../components/TweetCard";

export function TweetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const tweetId = Number(id);

  const { data: tweet, isLoading, error } = useQuery({
    queryKey: ["tweet", tweetId],
    queryFn: () => getTweet(tweetId),
    enabled: !isNaN(tweetId),
  });

  return (
    <div className="max-w-2xl mx-auto bg-white min-h-screen border-x border-gray-200">
      <div className="flex items-center gap-4 p-4 border-b border-gray-200">
        <button onClick={() => navigate(-1)} className="text-blue-600 hover:underline">
          Back
        </button>
        <h2 className="text-xl font-bold">Tweet</h2>
      </div>
      {isLoading && <p className="p-4 text-gray-500">Loading...</p>}
      {error && <p className="p-4 text-red-500">Tweet not found</p>}
      {tweet && <TweetCard tweet={tweet} />}
    </div>
  );
}
