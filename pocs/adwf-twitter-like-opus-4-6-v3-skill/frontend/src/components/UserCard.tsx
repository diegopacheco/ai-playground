import { Link } from "@tanstack/react-router";
import { FollowButton } from "./FollowButton";
import { getUser } from "../auth";
import type { UserResponse } from "../types";

interface UserCardProps {
  user: UserResponse;
  isFollowing?: boolean;
}

export function UserCard({ user, isFollowing = false }: UserCardProps) {
  const currentUser = getUser();
  const isMe = currentUser?.id === user.id;

  return (
    <div className="flex items-center justify-between p-3 hover:bg-gray-50 transition-colors">
      <Link
        to="/profile/$userId"
        params={{ userId: user.id }}
        className="flex items-center gap-3"
      >
        <div className="w-10 h-10 rounded-full bg-[#1DA1F2] flex items-center justify-center text-white font-bold">
          {user.username[0].toUpperCase()}
        </div>
        <div>
          <p className="font-bold text-gray-900 hover:underline">
            {user.username}
          </p>
          <p className="text-gray-500 text-sm">{user.email}</p>
        </div>
      </Link>
      {!isMe && <FollowButton userId={user.id} isFollowing={isFollowing} />}
    </div>
  );
}
