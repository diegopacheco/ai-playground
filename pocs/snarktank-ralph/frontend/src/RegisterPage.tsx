import React, { useState } from 'react';
import { useAuth } from './AuthContext';

interface RegisterPageProps {
  onSwitchToLogin: () => void;
}

function RegisterPage({ onSwitchToLogin }: RegisterPageProps) {
  const { register } = useAuth();
  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setLoading(true);
    const err = await register(username, displayName, password);
    setLoading(false);
    if (err) setError(err);
  }

  return (
    <div className="auth-page">
      <h2>Join SnarkTank</h2>
      <form onSubmit={handleSubmit} className="auth-form">
        {error && <div className="auth-error">{error}</div>}
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={e => setUsername(e.target.value)}
          required
          minLength={3}
          maxLength={30}
        />
        <input
          type="text"
          placeholder="Display Name"
          value={displayName}
          onChange={e => setDisplayName(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password (min 6 characters)"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
          minLength={6}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Creating account...' : 'Register'}
        </button>
      </form>
      <p className="auth-switch">
        Already have an account?{' '}
        <button onClick={onSwitchToLogin} className="auth-link">Login</button>
      </p>
    </div>
  );
}

export default RegisterPage;
