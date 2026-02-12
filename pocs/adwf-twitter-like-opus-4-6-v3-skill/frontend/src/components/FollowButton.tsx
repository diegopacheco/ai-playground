import { useToggleFollow } from "../hooks/useUsers";

interface FollowButtonProps {
  userId: string;
  isFollowing: boolean;
}

export function FollowButton({ userId, isFollowing }: FollowButtonProps) {
  const toggleFollow = useToggleFollow();

  return (
    <button
      onClick={() => toggleFollow.mutate({ userId, isFollowing })}
      disabled={toggleFollow.isPending}
      className={`px-4 py-1.5 rounded-full font-bold text-sm cursor-pointer transition-colors ${
        isFollowing
          ? "border border-gray-300 text-gray-700 hover:border-red-300 hover:text-red-500 hover:bg-red-50"
          : "bg-[#1DA1F2] text-white hover:bg-[#1a91da]"
      }`}
    >
      {isFollowing ? "Following" : "Follow"}
    </button>
  );
}
