import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getUser } from "../api/users";
import { getUserTweets } from "../api/tweets";
import { UserCard } from "../components/UserCard";
import { TweetCard } from "../components/TweetCard";

export function ProfilePage() {
  const { id } = useParams<{ id: string }>();
  const userId = Number(id);

  const { data: profile, isLoading: profileLoading } = useQuery({
    queryKey: ["user", userId],
    queryFn: () => getUser(userId),
    enabled: !isNaN(userId),
  });

  const { data: tweets, isLoading: tweetsLoading } = useQuery({
    queryKey: ["tweets", userId],
    queryFn: () => getUserTweets(userId),
    enabled: !isNaN(userId),
  });

  if (profileLoading) {
    return <p className="p-4 text-gray-500">Loading profile...</p>;
  }

  if (!profile) {
    return <p className="p-4 text-red-500">User not found</p>;
  }

  return (
    <div className="max-w-2xl mx-auto bg-white min-h-screen border-x border-gray-200">
      <h2 className="text-xl font-bold p-4 border-b border-gray-200">Profile</h2>
      <div className="p-4">
        <UserCard profile={profile} />
      </div>
      <h3 className="text-lg font-bold px-4 py-2 border-b border-gray-200">Tweets</h3>
      {tweetsLoading && <p className="p-4 text-gray-500">Loading tweets...</p>}
      {tweets && tweets.length === 0 && (
        <p className="p-4 text-gray-500">No tweets yet</p>
      )}
      {tweets?.map((tweet) => (
        <TweetCard key={tweet.id} tweet={tweet} />
      ))}
    </div>
  );
}
