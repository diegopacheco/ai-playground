import { useMutation, useQueryClient } from '@tanstack/react-query';
import { User } from '@/types';
import { usersApi } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';
import { Link } from 'react-router-dom';

interface UserCardProps {
  user: User;
  isFollowing?: boolean;
}

export const UserCard = ({ user, isFollowing = false }: UserCardProps) => {
  const { user: currentUser } = useAuth();
  const queryClient = useQueryClient();

  const followMutation = useMutation({
    mutationFn: () =>
      isFollowing ? usersApi.unfollow(user.id) : usersApi.follow(user.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['followers'] });
      queryClient.invalidateQueries({ queryKey: ['following'] });
      queryClient.invalidateQueries({ queryKey: ['user', user.id] });
    },
  });

  const isCurrentUser = currentUser?.id === user.id;

  return (
    <div className="border border-gray-200 rounded-lg p-4 flex items-start justify-between hover:bg-gray-50 transition">
      <div className="flex items-start space-x-3 flex-1">
        <div className="flex-shrink-0 w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold">
          {user.username.charAt(0).toUpperCase()}
        </div>
        <div className="flex-1 min-w-0">
          <Link to={`/profile/${user.id}`}>
            <h3 className="font-semibold hover:underline">
              {user.display_name || user.username}
            </h3>
            <p className="text-gray-500 text-sm">@{user.username}</p>
          </Link>
          {user.bio && (
            <p className="mt-1 text-sm text-gray-700 line-clamp-2">
              {user.bio}
            </p>
          )}
        </div>
      </div>
      {!isCurrentUser && (
        <button
          onClick={() => followMutation.mutate()}
          disabled={followMutation.isPending}
          className={`px-4 py-2 rounded-full font-semibold transition ${
            isFollowing
              ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              : 'bg-blue-500 text-white hover:bg-blue-600'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
        >
          {followMutation.isPending
            ? '...'
            : isFollowing
            ? 'Unfollow'
            : 'Follow'}
        </button>
      )}
    </div>
  );
};
