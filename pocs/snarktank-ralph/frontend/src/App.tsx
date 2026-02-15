import React, { useState, useEffect, useCallback } from 'react';
import { AuthProvider, useAuth } from './AuthContext';
import LoginPage from './LoginPage';
import RegisterPage from './RegisterPage';
import ComposeSnark from './ComposeSnark';
import ProfilePage from './ProfilePage';
import './App.css';

interface Snark {
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

function useRoute() {
  const [path, setPath] = useState(window.location.hash.slice(1) || '/');

  useEffect(() => {
    function onHashChange() {
      setPath(window.location.hash.slice(1) || '/');
    }
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  function navigate(newPath: string) {
    window.location.hash = newPath;
  }

  return { path, navigate };
}

function SnarkCard({ snark, onLikeToggle, onNavigate }: { snark: Snark; onLikeToggle: (id: number) => void; onNavigate: (path: string) => void }) {
  return (
    <div className="snark-card">
      <div className="snark-author">
        <span className="snark-display-name clickable" onClick={() => onNavigate(`/profile/${snark.author.username}`)}>{snark.author.displayName}</span>
        <span className="snark-username clickable" onClick={() => onNavigate(`/profile/${snark.author.username}`)}>@{snark.author.username}</span>
        <span className="snark-time">{timeAgo(snark.createdAt)}</span>
      </div>
      <div className="snark-content">{snark.content}</div>
      <div className="snark-actions">
        <button
          className={`like-btn ${snark.likedByMe ? 'liked' : ''}`}
          onClick={() => onLikeToggle(snark.id)}
        >
          {snark.likedByMe ? 'Liked' : 'Like'} ({snark.likeCount})
        </button>
      </div>
    </div>
  );
}

function AppContent() {
  const { user, token, logout } = useAuth();
  const [page, setPage] = useState<'login' | 'register'>('login');
  const [snarks, setSnarks] = useState<Snark[]>([]);
  const { path, navigate } = useRoute();

  const loadSnarks = useCallback(async () => {
    try {
      const headers: Record<string, string> = {};
      if (token) headers['Authorization'] = `Bearer ${token}`;
      const res = await fetch('/api/snarks', { headers });
      if (res.ok) {
        const data = await res.json();
        setSnarks(data);
      }
    } catch {}
  }, [token]);

  useEffect(() => {
    if (user && path === '/') loadSnarks();
  }, [user, loadSnarks, path]);

  if (!user) {
    if (page === 'register') {
      return <RegisterPage onSwitchToLogin={() => setPage('login')} />;
    }
    return <LoginPage onSwitchToRegister={() => setPage('register')} />;
  }

  function handleSnarkPosted(snark: Snark) {
    setSnarks(prev => [{ ...snark, likeCount: 0, replyCount: 0, likedByMe: false }, ...prev]);
  }

  async function handleLikeToggle(snarkId: number) {
    const snark = snarks.find(s => s.id === snarkId);
    if (!snark || !token) return;
    const method = snark.likedByMe ? 'DELETE' : 'POST';
    try {
      const res = await fetch(`/api/snarks/${snarkId}/like`, {
        method,
        headers: { 'Authorization': `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setSnarks(prev => prev.map(s =>
          s.id === snarkId ? { ...s, likedByMe: data.liked, likeCount: data.likeCount } : s
        ));
      }
    } catch {}
  }

  const profileMatch = path.match(/^\/profile\/(.+)$/);

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="clickable" onClick={() => navigate('/')}>SnarkTank</h1>
        <div className="header-user">
          <span className="clickable" onClick={() => navigate(`/profile/${user.username}`)}>@{user.username}</span>
          <button onClick={logout} className="logout-btn">Logout</button>
        </div>
      </header>
      <main className="app-main">
        {profileMatch ? (
          <ProfilePage username={profileMatch[1]} onNavigate={navigate} />
        ) : (
          <>
            <ComposeSnark onSnarkPosted={handleSnarkPosted} />
            <div className="timeline">
              {snarks.map(s => (
                <SnarkCard key={s.id} snark={s} onLikeToggle={handleLikeToggle} onNavigate={navigate} />
              ))}
              {snarks.length === 0 && (
                <p className="empty-state">No snarks yet. Be the first to post!</p>
              )}
            </div>
          </>
        )}
      </main>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
