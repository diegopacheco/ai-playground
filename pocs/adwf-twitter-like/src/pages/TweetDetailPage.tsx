import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { tweetsApi } from '@/lib/api';
import { NavigationBar } from '@/components/NavigationBar';
import { TweetCard } from '@/components/TweetCard';
import { CommentList } from '@/components/CommentList';

export const TweetDetailPage = () => {
  const { tweetId } = useParams<{ tweetId: string }>();
  const tweetIdNum = parseInt(tweetId || '0', 10);

  const { data: tweet, isLoading, isError } = useQuery({
    queryKey: ['tweet', tweetIdNum],
    queryFn: () => tweetsApi.getTweet(tweetIdNum),
    enabled: !!tweetIdNum,
  });

  return (
    <div className="min-h-screen bg-gray-100">
      <NavigationBar />
      <div className="container mx-auto max-w-2xl py-6 px-4">
        {isLoading ? (
          <div className="flex justify-center items-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          </div>
        ) : isError || !tweet ? (
          <div className="bg-white rounded-lg shadow-md p-6 text-center text-red-500">
            Tweet not found or failed to load
          </div>
        ) : (
          <>
            <div className="mb-6">
              <TweetCard tweet={tweet} />
            </div>
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-bold mb-4">Comments</h2>
              <CommentList tweetId={tweetIdNum} />
            </div>
          </>
        )}
      </div>
    </div>
  );
};
