import { NavigationBar } from '@/components/NavigationBar';
import { TweetComposer } from '@/components/TweetComposer';
import { FeedList } from '@/components/FeedList';

export const HomePage = () => {
  return (
    <div className="min-h-screen bg-gray-100">
      <NavigationBar />
      <div className="container mx-auto max-w-2xl py-6 px-4">
        <div className="mb-6">
          <TweetComposer />
        </div>
        <FeedList />
      </div>
    </div>
  );
};
