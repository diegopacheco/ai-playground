import { useQuery } from '@tanstack/react-query';
import { tweetsApi } from '@/lib/api';
import { TweetCard } from './TweetCard';

export const FeedList = () => {
  const { data: tweets, isLoading, isError } = useQuery({
    queryKey: ['tweets', 'feed'],
    queryFn: () => tweetsApi.getFeed(),
  });

  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="text-center py-8 text-red-500">
        Failed to load tweets. Please try again.
      </div>
    );
  }

  if (!tweets || tweets.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No tweets to display. Follow some users to see their tweets here.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {tweets.map((tweet) => (
        <TweetCard key={tweet.id} tweet={tweet} />
      ))}
    </div>
  );
};
