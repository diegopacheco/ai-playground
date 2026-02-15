import React, { useState, useEffect, useCallback } from 'react';
import { AuthProvider, useAuth } from './AuthContext';
import LoginPage from './LoginPage';
import RegisterPage from './RegisterPage';
import ComposeSnark from './ComposeSnark';
import './App.css';

interface Snark {
  id: number;
  content: string;
  createdAt: string;
  likeCount: number;
  replyCount: number;
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

function SnarkCard({ snark }: { snark: Snark }) {
  return (
    <div className="snark-card">
      <div className="snark-author">
        <span className="snark-display-name">{snark.author.displayName}</span>
        <span className="snark-username">@{snark.author.username}</span>
        <span className="snark-time">{timeAgo(snark.createdAt)}</span>
      </div>
      <div className="snark-content">{snark.content}</div>
    </div>
  );
}

function AppContent() {
  const { user, token, logout } = useAuth();
  const [page, setPage] = useState<'login' | 'register'>('login');
  const [snarks, setSnarks] = useState<Snark[]>([]);

  const loadSnarks = useCallback(async () => {
    try {
      const res = await fetch('/api/snarks');
      if (res.ok) {
        const data = await res.json();
        setSnarks(data);
      }
    } catch {}
  }, []);

  useEffect(() => {
    if (user) loadSnarks();
  }, [user, loadSnarks]);

  if (!user) {
    if (page === 'register') {
      return <RegisterPage onSwitchToLogin={() => setPage('login')} />;
    }
    return <LoginPage onSwitchToRegister={() => setPage('register')} />;
  }

  function handleSnarkPosted(snark: Snark) {
    setSnarks(prev => [{ ...snark, likeCount: 0, replyCount: 0 }, ...prev]);
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>SnarkTank</h1>
        <div className="header-user">
          <span>@{user.username}</span>
          <button onClick={logout} className="logout-btn">Logout</button>
        </div>
      </header>
      <main className="app-main">
        <ComposeSnark onSnarkPosted={handleSnarkPosted} />
        <div className="timeline">
          {snarks.map(s => (
            <SnarkCard key={s.id} snark={s} />
          ))}
          {snarks.length === 0 && (
            <p className="empty-state">No snarks yet. Be the first to post!</p>
          )}
        </div>
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
