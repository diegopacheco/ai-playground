import { useQuery } from "@tanstack/react-query";
import { getUserFollowers, getUserFollowing } from "../api/users";
import { UserCard } from "./UserCard";

interface FollowersListProps {
  userId: number;
  type: "followers" | "following";
  onUserClick?: (userId: number) => void;
}

export function FollowersList({ userId, type, onUserClick }: FollowersListProps) {
  const { data, isLoading, isError } = useQuery({
    queryKey: [type, userId],
    queryFn: () =>
      type === "followers"
        ? getUserFollowers(userId)
        : getUserFollowing(userId),
  });

  if (isLoading) {
    return (
      <div className="flex justify-center py-4">
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="text-center py-4 text-red-500">
        Failed to load {type}.
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-4 text-gray-500">
        No {type} yet.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {data.map((u) => (
        <UserCard key={u.id} targetUser={u} onUserClick={onUserClick} />
      ))}
    </div>
  );
}
