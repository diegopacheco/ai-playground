import React, { useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

interface SearchUser {
  id: number;
  username: string;
  displayName: string;
  bio: string;
  followerCount: number;
  followingCount: number;
  followedByMe: boolean;
}

interface SearchSnark {
  id: number;
  content: string;
  createdAt: string;
  likeCount: number;
  replyCount: number;
  likedByMe: boolean;
  author: {
    id: number;
    username: string;
    displayName: string;
  };
}

interface SearchResults {
  users: SearchUser[];
  snarks: SearchSnark[];
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

export default function SearchPage({ query, onNavigate }: { query: string; onNavigate: (path: string) => void }) {
  const { user, token } = useAuth();
  const [results, setResults] = useState<SearchResults>({ users: [], snarks: [] });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!query) return;
    setLoading(true);
    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;
    fetch(`/api/search?q=${encodeURIComponent(query)}`, { headers })
      .then(res => res.json())
      .then(data => { setResults(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [query, token]);

  async function handleFollow(userId: number, followed: boolean) {
    if (!token) return;
    const method = followed ? 'DELETE' : 'POST';
    try {
      const res = await fetch(`/api/users/${userId}/follow`, {
        method,
        headers: { 'Authorization': `Bearer ${token}` },
      });
      if (res.ok) {
        setResults(prev => ({
          ...prev,
          users: prev.users.map(u =>
            u.id === userId ? { ...u, followedByMe: !followed, followerCount: u.followerCount + (followed ? -1 : 1) } : u
          ),
        }));
      }
    } catch {}
  }

  async function handleLikeToggle(snarkId: number) {
    const snark = results.snarks.find(s => s.id === snarkId);
    if (!snark || !token) return;
    const method = snark.likedByMe ? 'DELETE' : 'POST';
    try {
      const res = await fetch(`/api/snarks/${snarkId}/like`, {
        method,
        headers: { 'Authorization': `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setResults(prev => ({
          ...prev,
          snarks: prev.snarks.map(s =>
            s.id === snarkId ? { ...s, likedByMe: data.liked, likeCount: data.likeCount } : s
          ),
        }));
      }
    } catch {}
  }

  if (!query) {
    return <p className="empty-state">Enter a search term to find users and snarks.</p>;
  }

  if (loading) {
    return <p className="empty-state">Searching...</p>;
  }

  return (
    <div className="search-results">
      <h3 className="search-section-title">Users</h3>
      {results.users.length === 0 ? (
        <p className="empty-state">No users found.</p>
      ) : (
        results.users.map(u => (
          <div key={u.id} className="search-user-card">
            <div className="search-user-info">
              <span className="snark-display-name clickable" onClick={() => onNavigate(`/profile/${u.username}`)}>{u.displayName}</span>
              <span className="snark-username clickable" onClick={() => onNavigate(`/profile/${u.username}`)}>@{u.username}</span>
              {u.bio && <p className="search-user-bio">{u.bio}</p>}
            </div>
            {user && u.id !== user.id && (
              <button
                className={`follow-btn ${u.followedByMe ? 'following' : ''}`}
                onClick={() => handleFollow(u.id, u.followedByMe)}
              >
                {u.followedByMe ? 'Unfollow' : 'Follow'}
              </button>
            )}
          </div>
        ))
      )}

      <h3 className="search-section-title">Snarks</h3>
      {results.snarks.length === 0 ? (
        <p className="empty-state">No snarks found.</p>
      ) : (
        results.snarks.map(s => (
          <div key={s.id} className="snark-card">
            <div className="snark-author">
              <span className="snark-display-name clickable" onClick={() => onNavigate(`/profile/${s.author.username}`)}>{s.author.displayName}</span>
              <span className="snark-username clickable" onClick={() => onNavigate(`/profile/${s.author.username}`)}>@{s.author.username}</span>
              <span className="snark-time">{timeAgo(s.createdAt)}</span>
            </div>
            <div className="snark-content">{s.content}</div>
            <div className="snark-actions">
              <button className="reply-btn" onClick={() => onNavigate(`/snark/${s.id}`)}>
                Reply ({s.replyCount})
              </button>
              <button
                className={`like-btn ${s.likedByMe ? 'liked' : ''}`}
                onClick={() => handleLikeToggle(s.id)}
              >
                {s.likedByMe ? 'Liked' : 'Like'} ({s.likeCount})
              </button>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
