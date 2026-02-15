import React, { useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

interface ProfileSnark {
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

interface ProfileData {
  id: number;
  username: string;
  displayName: string;
  bio: string;
  createdAt: string;
  followerCount: number;
  followingCount: number;
  followedByMe: boolean;
  snarks: ProfileSnark[];
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

export default function ProfilePage({ username, onNavigate }: { username: string; onNavigate: (path: string) => void }) {
  const { user, token } = useAuth();
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [error, setError] = useState('');
  const [editingBio, setEditingBio] = useState(false);
  const [bioText, setBioText] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;
    fetch(`/api/users/${username}`, { headers })
      .then(res => {
        if (!res.ok) throw new Error('User not found');
        return res.json();
      })
      .then(data => {
        setProfile(data);
        setBioText(data.bio);
      })
      .catch(() => setError('User not found'));
  }, [username, token]);

  async function handleSaveBio() {
    if (!token) return;
    setSaving(true);
    try {
      const res = await fetch('/api/users/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ bio: bioText }),
      });
      if (res.ok) {
        const data = await res.json();
        setProfile(prev => prev ? { ...prev, bio: data.bio } : null);
        setEditingBio(false);
      }
    } finally {
      setSaving(false);
    }
  }

  async function handleLikeToggle(snarkId: number) {
    if (!profile || !token) return;
    const snark = profile.snarks.find(s => s.id === snarkId);
    if (!snark) return;
    const method = snark.likedByMe ? 'DELETE' : 'POST';
    try {
      const res = await fetch(`/api/snarks/${snarkId}/like`, {
        method,
        headers: { 'Authorization': `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setProfile(prev => prev ? {
          ...prev,
          snarks: prev.snarks.map(s =>
            s.id === snarkId ? { ...s, likedByMe: data.liked, likeCount: data.likeCount } : s
          ),
        } : null);
      }
    } catch {}
  }

  if (error) {
    return (
      <div className="profile-page">
        <button className="back-btn" onClick={() => onNavigate('/')}>Back</button>
        <p className="empty-state">{error}</p>
      </div>
    );
  }

  if (!profile) {
    return <div className="profile-page"><p className="empty-state">Loading...</p></div>;
  }

  async function handleFollowToggle() {
    if (!profile || !token) return;
    const method = profile.followedByMe ? 'DELETE' : 'POST';
    try {
      const res = await fetch(`/api/users/${profile.id}/follow`, {
        method,
        headers: { 'Authorization': `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setProfile(prev => prev ? {
          ...prev,
          followedByMe: data.following,
          followerCount: data.followerCount,
          followingCount: data.followingCount,
        } : null);
      }
    } catch {}
  }

  const isOwnProfile = user && user.username === profile.username;
  const joinDate = new Date(profile.createdAt).toLocaleDateString('en-US', { month: 'long', year: 'numeric' });

  return (
    <div className="profile-page">
      <button className="back-btn" onClick={() => onNavigate('/')}>Back</button>
      <div className="profile-header">
        <div className="profile-names-row">
          <div className="profile-names">
            <h2 className="profile-display-name">{profile.displayName}</h2>
            <span className="profile-username">@{profile.username}</span>
          </div>
          {!isOwnProfile && user && (
            <button
              className={`follow-btn ${profile.followedByMe ? 'following' : ''}`}
              onClick={handleFollowToggle}
            >
              {profile.followedByMe ? 'Unfollow' : 'Follow'}
            </button>
          )}
        </div>
        <div className="profile-bio-section">
          {editingBio ? (
            <div className="bio-edit">
              <textarea
                className="bio-input"
                value={bioText}
                onChange={e => setBioText(e.target.value)}
                maxLength={160}
                rows={3}
              />
              <div className="bio-edit-actions">
                <span className="char-count">{160 - bioText.length}</span>
                <button className="cancel-btn" onClick={() => { setEditingBio(false); setBioText(profile.bio); }}>Cancel</button>
                <button className="save-btn" onClick={handleSaveBio} disabled={saving}>Save</button>
              </div>
            </div>
          ) : (
            <div className="bio-display">
              <p className="profile-bio">{profile.bio || (isOwnProfile ? 'No bio yet.' : '')}</p>
              {isOwnProfile && (
                <button className="edit-bio-btn" onClick={() => setEditingBio(true)}>Edit bio</button>
              )}
            </div>
          )}
        </div>
        <div className="profile-stats">
          <span><strong>{profile.followingCount}</strong> Following</span>
          <span><strong>{profile.followerCount}</strong> Followers</span>
        </div>
        <p className="profile-joined">Joined {joinDate}</p>
      </div>
      <h3 className="profile-snarks-title">Snarks</h3>
      <div className="timeline">
        {profile.snarks.map(s => (
          <div key={s.id} className="snark-card">
            <div className="snark-author">
              <span className="snark-display-name">{s.author.displayName}</span>
              <span className="snark-username">@{s.author.username}</span>
              <span className="snark-time">{timeAgo(s.createdAt)}</span>
            </div>
            <div className="snark-content">{s.content}</div>
            <div className="snark-actions">
              <button
                className={`like-btn ${s.likedByMe ? 'liked' : ''}`}
                onClick={() => handleLikeToggle(s.id)}
              >
                {s.likedByMe ? 'Liked' : 'Like'} ({s.likeCount})
              </button>
            </div>
          </div>
        ))}
        {profile.snarks.length === 0 && (
          <p className="empty-state">No snarks yet.</p>
        )}
      </div>
    </div>
  );
}
