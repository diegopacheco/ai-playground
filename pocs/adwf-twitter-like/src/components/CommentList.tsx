import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { commentsApi } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';

interface CommentListProps {
  tweetId: number;
}

export const CommentList = ({ tweetId }: CommentListProps) => {
  const [newComment, setNewComment] = useState('');
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const { data: comments, isLoading } = useQuery({
    queryKey: ['comments', tweetId],
    queryFn: () => commentsApi.getComments(tweetId),
  });

  const addCommentMutation = useMutation({
    mutationFn: (content: string) => commentsApi.addComment(tweetId, content),
    onSuccess: () => {
      setNewComment('');
      queryClient.invalidateQueries({ queryKey: ['comments', tweetId] });
      queryClient.invalidateQueries({ queryKey: ['tweet', tweetId] });
    },
  });

  const deleteCommentMutation = useMutation({
    mutationFn: (commentId: number) => commentsApi.deleteComment(commentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['comments', tweetId] });
      queryClient.invalidateQueries({ queryKey: ['tweet', tweetId] });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (newComment.trim() && newComment.length <= 280) {
      addCommentMutation.mutate(newComment);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="border-b pb-4">
        <textarea
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          placeholder="Write a comment..."
          className="w-full p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={2}
          maxLength={280}
          aria-label="Comment content"
        />
        <div className="flex justify-between items-center mt-2">
          <span
            className={`text-sm ${
              newComment.length > 280 ? 'text-red-500' : 'text-gray-500'
            }`}
          >
            {newComment.length}/280
          </span>
          <button
            type="submit"
            disabled={
              !newComment.trim() ||
              newComment.length > 280 ||
              addCommentMutation.isPending
            }
            className="bg-blue-500 text-white px-4 py-2 rounded-full hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
          >
            {addCommentMutation.isPending ? 'Posting...' : 'Comment'}
          </button>
        </div>
      </form>

      {isLoading ? (
        <div className="flex justify-center py-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
        </div>
      ) : comments && comments.length > 0 ? (
        <div className="space-y-3">
          {comments.map((comment) => (
            <div
              key={comment.id}
              className="border border-gray-200 rounded-lg p-3"
            >
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                  {comment.author_username?.charAt(0).toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold">
                      {comment.author_display_name || comment.author_username}
                    </span>
                    <span className="text-gray-500 text-sm">
                      @{comment.author_username}
                    </span>
                    <span className="text-gray-500 text-sm">·</span>
                    <span className="text-gray-500 text-sm">
                      {formatDate(comment.created_at)}
                    </span>
                  </div>
                  <p className="mt-1 text-gray-900 whitespace-pre-wrap break-words">
                    {comment.content}
                  </p>
                  {user && user.id === comment.user_id && (
                    <button
                      onClick={() => deleteCommentMutation.mutate(comment.id)}
                      disabled={deleteCommentMutation.isPending}
                      className="mt-2 text-sm text-red-500 hover:text-red-700"
                      aria-label="Delete comment"
                    >
                      Delete
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-4 text-gray-500">
          No comments yet. Be the first to comment!
        </div>
      )}
    </div>
  );
};
