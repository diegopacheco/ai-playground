import React, { useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

interface Snark {
  id: number;
  content: string;
  createdAt: string;
  parentId: number | null;
  likeCount: number;
  replyCount: number;
  likedByMe: boolean;
  author: {
    id: number;
    username: string;
    displayName: string;
  };
}

interface SnarkDetail extends Snark {
  replies: Snark[];
}

function timeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const seconds = Math.floor((now - then) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function SnarkDetailPage({ snarkId, onNavigate }: { snarkId: number; onNavigate: (path: string) => void }) {
  const { token } = useAuth();
  const [detail, setDetail] = useState<SnarkDetail | null>(null);
  const [error, setError] = useState('');
  const [replyContent, setReplyContent] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;
    fetch(`/api/snarks/${snarkId}`, { headers })
      .then(res => {
        if (!res.ok) throw new Error('Not found');
        return res.json();
      })
      .then(data => setDetail(data))
      .catch(() => setError('Snark not found'));
  }, [snarkId, token]);

  async function handleLikeToggle(id: number) {
    if (!detail || !token) return;
    const target = id === detail.id ? detail : detail.replies.find(r => r.id === id);
    if (!target) return;
    const method = target.likedByMe ? 'DELETE' : 'POST';
    try {
      const res = await fetch(`/api/snarks/${id}/like`, {
        method,
        headers: { 'Authorization': `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        if (id === detail.id) {
          setDetail(prev => prev ? { ...prev, likedByMe: data.liked, likeCount: data.likeCount } : null);
        } else {
          setDetail(prev => prev ? {
            ...prev,
            replies: prev.replies.map(r =>
              r.id === id ? { ...r, likedByMe: data.liked, likeCount: data.likeCount } : r
            ),
          } : null);
        }
      }
    } catch {}
  }

  async function handleSubmitReply(e: React.FormEvent) {
    e.preventDefault();
    if (!replyContent.trim() || !token || submitting) return;
    setSubmitting(true);
    try {
      const res = await fetch('/api/snarks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ content: replyContent.trim(), parentId: snarkId }),
      });
      if (res.ok) {
        const newReply = await res.json();
        setDetail(prev => prev ? {
          ...prev,
          replyCount: prev.replyCount + 1,
          replies: [...prev.replies, newReply],
        } : null);
        setReplyContent('');
      }
    } finally {
      setSubmitting(false);
    }
  }

  if (error) {
    return (
      <div>
        <button className="back-btn" onClick={() => onNavigate('/')}>Back</button>
        <p className="empty-state">{error}</p>
      </div>
    );
  }

  if (!detail) {
    return <p className="empty-state">Loading...</p>;
  }

  const remaining = 280 - replyContent.length;

  return (
    <div>
      <button className="back-btn" onClick={() => onNavigate('/')}>Back</button>
      <div className="snark-card snark-detail-card">
        <div className="snark-author">
          <span className="snark-display-name clickable" onClick={() => onNavigate(`/profile/${detail.author.username}`)}>{detail.author.displayName}</span>
          <span className="snark-username clickable" onClick={() => onNavigate(`/profile/${detail.author.username}`)}>@{detail.author.username}</span>
          <span className="snark-time">{timeAgo(detail.createdAt)}</span>
        </div>
        <div className="snark-content">{detail.content}</div>
        <div className="snark-actions">
          <span className="reply-count">Replies ({detail.replyCount})</span>
          <button
            className={`like-btn ${detail.likedByMe ? 'liked' : ''}`}
            onClick={() => handleLikeToggle(detail.id)}
          >
            {detail.likedByMe ? 'Liked' : 'Like'} ({detail.likeCount})
          </button>
        </div>
      </div>
      <form className="reply-form" onSubmit={handleSubmitReply}>
        <textarea
          className="compose-input"
          placeholder="Write a reply..."
          value={replyContent}
          onChange={e => setReplyContent(e.target.value)}
          maxLength={280}
          rows={3}
        />
        <div className="compose-footer">
          <span className={`char-count ${remaining < 20 ? (remaining < 0 ? 'char-count-over' : 'char-count-warning') : ''}`}>{remaining}</span>
          <button type="submit" className="compose-btn" disabled={!replyContent.trim() || remaining < 0 || submitting}>Reply</button>
        </div>
      </form>
      <h3 className="replies-title">Replies</h3>
      <div className="timeline">
        {detail.replies.map(r => (
          <div key={r.id} className="snark-card reply-card">
            <div className="snark-author">
              <span className="snark-display-name clickable" onClick={() => onNavigate(`/profile/${r.author.username}`)}>{r.author.displayName}</span>
              <span className="snark-username clickable" onClick={() => onNavigate(`/profile/${r.author.username}`)}>@{r.author.username}</span>
              <span className="snark-time">{timeAgo(r.createdAt)}</span>
            </div>
            <div className="snark-content">{r.content}</div>
            <div className="snark-actions">
              <button
                className={`like-btn ${r.likedByMe ? 'liked' : ''}`}
                onClick={() => handleLikeToggle(r.id)}
              >
                {r.likedByMe ? 'Liked' : 'Like'} ({r.likeCount})
              </button>
            </div>
          </div>
        ))}
        {detail.replies.length === 0 && (
          <p className="empty-state">No replies yet. Be the first to reply!</p>
        )}
      </div>
    </div>
  );
}
