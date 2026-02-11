import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { usersApi, tweetsApi } from '@/lib/api';
import { NavigationBar } from '@/components/NavigationBar';
import { TweetCard } from '@/components/TweetCard';
import { UserCard } from '@/components/UserCard';
import { useAuth } from '@/contexts/AuthContext';

export const ProfilePage = () => {
  const { userId } = useParams<{ userId: string }>();
  const [activeTab, setActiveTab] = useState<'tweets' | 'followers' | 'following'>('tweets');
  const { user: currentUser } = useAuth();
  const queryClient = useQueryClient();

  const userIdNum = parseInt(userId || '0', 10);

  const { data: user, isLoading: userLoading } = useQuery({
    queryKey: ['user', userIdNum],
    queryFn: () => usersApi.getUser(userIdNum),
    enabled: !!userIdNum,
  });

  const { data: tweets, isLoading: tweetsLoading } = useQuery({
    queryKey: ['tweets', 'user', userIdNum],
    queryFn: () => tweetsApi.getUserTweets(userIdNum),
    enabled: !!userIdNum && activeTab === 'tweets',
  });

  const { data: followers, isLoading: followersLoading } = useQuery({
    queryKey: ['followers', userIdNum],
    queryFn: () => usersApi.getFollowers(userIdNum),
    enabled: !!userIdNum && activeTab === 'followers',
  });

  const { data: following, isLoading: followingLoading } = useQuery({
    queryKey: ['following', userIdNum],
    queryFn: () => usersApi.getFollowing(userIdNum),
    enabled: !!userIdNum && activeTab === 'following',
  });

  const followMutation = useMutation({
    mutationFn: (isFollowing: boolean) =>
      isFollowing ? usersApi.unfollow(userIdNum) : usersApi.follow(userIdNum),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['user', userIdNum] });
      queryClient.invalidateQueries({ queryKey: ['followers'] });
      queryClient.invalidateQueries({ queryKey: ['following'] });
    },
  });

  const isCurrentUser = currentUser?.id === userIdNum;

  if (userLoading) {
    return (
      <div className="min-h-screen bg-gray-100">
        <NavigationBar />
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-100">
        <NavigationBar />
        <div className="container mx-auto max-w-2xl py-6 px-4">
          <div className="text-center text-red-500">User not found</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <NavigationBar />
      <div className="container mx-auto max-w-2xl py-6 px-4">
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0 w-20 h-20 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-2xl">
                {user.username.charAt(0).toUpperCase()}
              </div>
              <div>
                <h1 className="text-2xl font-bold">
                  {user.display_name || user.username}
                </h1>
                <p className="text-gray-500">@{user.username}</p>
                {user.bio && <p className="mt-2 text-gray-700">{user.bio}</p>}
                <p className="mt-2 text-sm text-gray-500">
                  Joined {new Date(user.created_at).toLocaleDateString('en-US', {
                    month: 'long',
                    year: 'numeric',
                  })}
                </p>
              </div>
            </div>
            {!isCurrentUser && (
              <button
                onClick={() => followMutation.mutate(false)}
                disabled={followMutation.isPending}
                className="bg-blue-500 text-white px-6 py-2 rounded-full font-semibold hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                {followMutation.isPending ? '...' : 'Follow'}
              </button>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md mb-6">
          <div className="flex border-b">
            <button
              onClick={() => setActiveTab('tweets')}
              className={`flex-1 py-4 text-center font-semibold transition ${
                activeTab === 'tweets'
                  ? 'border-b-2 border-blue-500 text-blue-500'
                  : 'text-gray-500 hover:bg-gray-50'
              }`}
            >
              Tweets
            </button>
            <button
              onClick={() => setActiveTab('followers')}
              className={`flex-1 py-4 text-center font-semibold transition ${
                activeTab === 'followers'
                  ? 'border-b-2 border-blue-500 text-blue-500'
                  : 'text-gray-500 hover:bg-gray-50'
              }`}
            >
              Followers
            </button>
            <button
              onClick={() => setActiveTab('following')}
              className={`flex-1 py-4 text-center font-semibold transition ${
                activeTab === 'following'
                  ? 'border-b-2 border-blue-500 text-blue-500'
                  : 'text-gray-500 hover:bg-gray-50'
              }`}
            >
              Following
            </button>
          </div>

          <div className="p-4">
            {activeTab === 'tweets' && (
              <>
                {tweetsLoading ? (
                  <div className="flex justify-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                  </div>
                ) : tweets && tweets.length > 0 ? (
                  <div className="space-y-4">
                    {tweets.map((tweet) => (
                      <TweetCard key={tweet.id} tweet={tweet} />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No tweets yet
                  </div>
                )}
              </>
            )}

            {activeTab === 'followers' && (
              <>
                {followersLoading ? (
                  <div className="flex justify-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                  </div>
                ) : followers && followers.length > 0 ? (
                  <div className="space-y-3">
                    {followers.map((follower) => (
                      <UserCard key={follower.id} user={follower} />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No followers yet
                  </div>
                )}
              </>
            )}

            {activeTab === 'following' && (
              <>
                {followingLoading ? (
                  <div className="flex justify-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                  </div>
                ) : following && following.length > 0 ? (
                  <div className="space-y-3">
                    {following.map((followedUser) => (
                      <UserCard key={followedUser.id} user={followedUser} isFollowing={true} />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    Not following anyone yet
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
