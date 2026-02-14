import { Link } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { followUser, unfollowUser } from "../api/users";
import { useAuth } from "../hooks/useAuth";
import type { UserProfile } from "../types";

interface UserCardProps {
  profile: UserProfile;
}

export function UserCard({ profile }: UserCardProps) {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const followMutation = useMutation({
    mutationFn: () =>
      profile.is_following ? unfollowUser(profile.id) : followUser(profile.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["user", profile.id] });
    },
  });

  const isOwnProfile = user?.id === profile.id;

  return (
    <div className="p-4 border border-gray-200 rounded-lg">
      <Link to={`/profile/${profile.id}`} className="font-bold text-lg hover:underline">
        {profile.display_name}
      </Link>
      <p className="text-gray-500">@{profile.username}</p>
      {profile.bio && <p className="mt-1 text-gray-700">{profile.bio}</p>}
      <div className="flex gap-4 mt-2 text-sm text-gray-500">
        <span>
          <strong className="text-gray-900">{profile.following_count}</strong> Following
        </span>
        <span>
          <strong className="text-gray-900">{profile.follower_count}</strong> Followers
        </span>
      </div>
      {!isOwnProfile && (
        <button
          onClick={() => followMutation.mutate()}
          disabled={followMutation.isPending}
          className={`mt-3 px-4 py-1 rounded-full text-sm font-bold ${
            profile.is_following
              ? "border border-gray-300 text-gray-700 hover:border-red-300 hover:text-red-600"
              : "bg-blue-600 text-white hover:bg-blue-700"
          }`}
        >
          {profile.is_following ? "Unfollow" : "Follow"}
        </button>
      )}
    </div>
  );
}
