import React, { useState } from 'react';
import { AuthProvider, useAuth } from './AuthContext';
import LoginPage from './LoginPage';
import RegisterPage from './RegisterPage';
import './App.css';

function AppContent() {
  const { user, logout } = useAuth();
  const [page, setPage] = useState<'login' | 'register'>('login');

  if (!user) {
    if (page === 'register') {
      return <RegisterPage onSwitchToLogin={() => setPage('login')} />;
    }
    return <LoginPage onSwitchToRegister={() => setPage('register')} />;
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
        <p>Welcome, {user.displayName}!</p>
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
