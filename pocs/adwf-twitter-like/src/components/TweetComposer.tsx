import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { tweetsApi } from '@/lib/api';

export const TweetComposer = () => {
  const [content, setContent] = useState('');
  const queryClient = useQueryClient();

  const createTweetMutation = useMutation({
    mutationFn: (content: string) => tweetsApi.createTweet(content),
    onSuccess: () => {
      setContent('');
      queryClient.invalidateQueries({ queryKey: ['tweets'] });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (content.trim() && content.length <= 280) {
      createTweetMutation.mutate(content);
    }
  };

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      <form onSubmit={handleSubmit}>
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="What's happening?"
          className="w-full p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={3}
          maxLength={280}
          aria-label="Tweet content"
        />
        <div className="flex justify-between items-center mt-3">
          <span
            className={`text-sm ${
              content.length > 280 ? 'text-red-500' : 'text-gray-500'
            }`}
          >
            {content.length}/280
          </span>
          <button
            type="submit"
            disabled={
              !content.trim() ||
              content.length > 280 ||
              createTweetMutation.isPending
            }
            className="bg-blue-500 text-white px-6 py-2 rounded-full font-semibold hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
          >
            {createTweetMutation.isPending ? 'Posting...' : 'Tweet'}
          </button>
        </div>
        {createTweetMutation.isError && (
          <div className="mt-2 text-red-500 text-sm">
            Failed to post tweet. Please try again.
          </div>
        )}
      </form>
    </div>
  );
};
