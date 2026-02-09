import { Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import {
  usePost,
  useComments,
  useCreateComment,
  useDeletePost,
} from "../hooks/useApi";

export default function PostDetailPage({ postId }: { postId: string }) {
  const navigate = useNavigate();
  const { data: post, isLoading, error } = usePost(postId);
  const { data: comments } = useComments(postId);
  const createComment = useCreateComment(postId);
  const deletePost = useDeletePost();

  const [commentAuthor, setCommentAuthor] = useState("");
  const [commentContent, setCommentContent] = useState("");

  if (isLoading) {
    return (
      <div className="flex justify-center py-20">
        <div className="text-gray-500 text-lg">Loading post...</div>
      </div>
    );
  }

  if (error || !post) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-red-700">
        Post not found or failed to load.
      </div>
    );
  }

  function handleSubmitComment(e: React.FormEvent) {
    e.preventDefault();
    if (!commentAuthor.trim() || !commentContent.trim()) return;
    createComment.mutate(
      { author: commentAuthor, content: commentContent },
      {
        onSuccess: () => {
          setCommentAuthor("");
          setCommentContent("");
        },
      }
    );
  }

  function handleDelete() {
    if (!confirm("Are you sure you want to delete this post?")) return;
    deletePost.mutate(postId, {
      onSuccess: () => navigate({ to: "/" }),
    });
  }

  return (
    <div>
      <Link
        to="/"
        className="text-indigo-600 hover:text-indigo-800 font-medium mb-6 inline-block"
      >
        Back to all posts
      </Link>

      <article className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">{post.title}</h1>
        <div className="flex items-center gap-3 text-sm text-gray-500 mb-6">
          <span className="font-medium text-indigo-600">{post.author}</span>
          <span>-</span>
          <span>{new Date(post.createdAt).toLocaleDateString()}</span>
          {post.updatedAt !== post.createdAt && (
            <>
              <span>-</span>
              <span>
                Updated {new Date(post.updatedAt).toLocaleDateString()}
              </span>
            </>
          )}
        </div>
        <div className="text-gray-700 leading-relaxed whitespace-pre-wrap">
          {post.content}
        </div>
        <div className="mt-8 flex gap-3">
          <Link
            to="/posts/$postId/edit"
            params={{ postId: String(post.id) }}
            className="bg-indigo-600 text-white px-5 py-2 rounded-lg hover:bg-indigo-700 transition-colors font-medium"
          >
            Edit
          </Link>
          <button
            onClick={handleDelete}
            className="bg-red-600 text-white px-5 py-2 rounded-lg hover:bg-red-700 transition-colors font-medium"
          >
            Delete
          </button>
        </div>
      </article>

      <section className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
        <h2 className="text-2xl font-semibold text-gray-900 mb-6">Comments</h2>

        {comments && comments.length > 0 ? (
          <div className="space-y-4 mb-8">
            {comments.map((comment) => (
              <div
                key={comment.id}
                className="border-l-4 border-indigo-200 pl-4 py-2"
              >
                <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
                  <span className="font-medium text-gray-700">
                    {comment.author}
                  </span>
                  <span>-</span>
                  <span>
                    {new Date(comment.createdAt).toLocaleDateString()}
                  </span>
                </div>
                <p className="text-gray-600">{comment.content}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 mb-8">No comments yet.</p>
        )}

        <form onSubmit={handleSubmitComment} className="space-y-4">
          <h3 className="text-lg font-medium text-gray-800">Add a comment</h3>
          <input
            type="text"
            placeholder="Your name"
            value={commentAuthor}
            onChange={(e) => setCommentAuthor(e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
          <textarea
            placeholder="Your comment"
            value={commentContent}
            onChange={(e) => setCommentContent(e.target.value)}
            rows={4}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
          <button
            type="submit"
            disabled={createComment.isPending}
            className="bg-indigo-600 text-white px-5 py-2 rounded-lg hover:bg-indigo-700 transition-colors font-medium disabled:opacity-50"
          >
            {createComment.isPending ? "Posting..." : "Post Comment"}
          </button>
        </form>
      </section>
    </div>
  );
}
