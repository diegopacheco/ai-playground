import { useMutation, useQueryClient } from "@tanstack/react-query";
import { followUser, unfollowUser } from "../api/follows";
import { useAuth } from "../hooks/useAuth";
import type { User } from "../types";

interface UserCardProps {
  targetUser: User;
  isFollowing?: boolean;
  onUserClick?: (userId: number) => void;
}

export function UserCard({ targetUser, isFollowing, onUserClick }: UserCardProps) {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const followMutation = useMutation({
    mutationFn: () => followUser(targetUser.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["followers"] });
      queryClient.invalidateQueries({ queryKey: ["following"] });
    },
  });

  const unfollowMutation = useMutation({
    mutationFn: () => unfollowUser(targetUser.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["followers"] });
      queryClient.invalidateQueries({ queryKey: ["following"] });
    },
  });

  const handleFollowToggle = () => {
    if (isFollowing) {
      unfollowMutation.mutate();
    } else {
      followMutation.mutate();
    }
  };

  const isOwnProfile = user?.id === targetUser.id;

  return (
    <div className="flex items-center justify-between p-3 bg-white border border-gray-200 rounded-lg">
      <div
        className="flex items-center gap-3 cursor-pointer"
        onClick={() => onUserClick?.(targetUser.id)}
      >
        <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold">
          {targetUser.username[0].toUpperCase()}
        </div>
        <span className="font-semibold text-gray-900 hover:underline">
          @{targetUser.username}
        </span>
      </div>
      {!isOwnProfile && (
        <button
          onClick={handleFollowToggle}
          disabled={followMutation.isPending || unfollowMutation.isPending}
          className={`px-4 py-1 rounded-full text-sm font-semibold transition-colors ${
            isFollowing
              ? "bg-gray-200 text-gray-700 hover:bg-red-100 hover:text-red-600"
              : "bg-blue-500 text-white hover:bg-blue-600"
          }`}
        >
          {isFollowing ? "Unfollow" : "Follow"}
        </button>
      )}
    </div>
  );
}
