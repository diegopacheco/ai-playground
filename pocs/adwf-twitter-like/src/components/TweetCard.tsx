import { Link } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Tweet } from '@/types';
import { tweetsApi } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';

interface TweetCardProps {
  tweet: Tweet;
}

export const TweetCard = ({ tweet }: TweetCardProps) => {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const likeMutation = useMutation({
    mutationFn: () =>
      tweet.is_liked
        ? tweetsApi.unlikeTweet(tweet.id)
        : tweetsApi.likeTweet(tweet.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tweets'] });
      queryClient.invalidateQueries({ queryKey: ['tweet', tweet.id] });
    },
  });

  const retweetMutation = useMutation({
    mutationFn: () =>
      tweet.is_retweeted
        ? tweetsApi.unretweetTweet(tweet.id)
        : tweetsApi.retweetTweet(tweet.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tweets'] });
      queryClient.invalidateQueries({ queryKey: ['tweet', tweet.id] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => tweetsApi.deleteTweet(tweet.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tweets'] });
    },
  });

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition">
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold">
          {tweet.author_username?.charAt(0).toUpperCase()}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center space-x-2">
            <Link
              to={`/profile/${tweet.user_id}`}
              className="font-semibold hover:underline"
            >
              {tweet.author_display_name || tweet.author_username}
            </Link>
            <Link
              to={`/profile/${tweet.user_id}`}
              className="text-gray-500 text-sm hover:underline"
            >
              @{tweet.author_username}
            </Link>
            <span className="text-gray-500 text-sm">·</span>
            <span className="text-gray-500 text-sm">
              {formatDate(tweet.created_at)}
            </span>
          </div>
          <Link to={`/tweet/${tweet.id}`}>
            <p className="mt-2 text-gray-900 whitespace-pre-wrap break-words">
              {tweet.content}
            </p>
          </Link>
          <div className="flex items-center space-x-6 mt-3 text-gray-500">
            <button
              onClick={() => likeMutation.mutate()}
              disabled={likeMutation.isPending}
              className={`flex items-center space-x-2 hover:text-red-500 transition ${
                tweet.is_liked ? 'text-red-500' : ''
              }`}
              aria-label={tweet.is_liked ? 'Unlike' : 'Like'}
            >
              <svg
                className="w-5 h-5"
                fill={tweet.is_liked ? 'currentColor' : 'none'}
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                />
              </svg>
              <span>{tweet.likes_count || 0}</span>
            </button>
            <button
              onClick={() => retweetMutation.mutate()}
              disabled={retweetMutation.isPending}
              className={`flex items-center space-x-2 hover:text-green-500 transition ${
                tweet.is_retweeted ? 'text-green-500' : ''
              }`}
              aria-label={tweet.is_retweeted ? 'Undo Retweet' : 'Retweet'}
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              <span>{tweet.retweets_count || 0}</span>
            </button>
            <Link
              to={`/tweet/${tweet.id}`}
              className="flex items-center space-x-2 hover:text-blue-500 transition"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
              <span>{tweet.comments_count || 0}</span>
            </Link>
            {user && user.id === tweet.user_id && (
              <button
                onClick={() => deleteMutation.mutate()}
                disabled={deleteMutation.isPending}
                className="flex items-center space-x-2 hover:text-red-500 transition"
                aria-label="Delete tweet"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
