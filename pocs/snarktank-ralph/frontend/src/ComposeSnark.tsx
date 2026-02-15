import React, { useState } from 'react';
import { useAuth } from './AuthContext';

interface ComposeSnarkProps {
  onSnarkPosted: (snark: any) => void;
}

export default function ComposeSnark({ onSnarkPosted }: ComposeSnarkProps) {
  const { token } = useAuth();
  const [content, setContent] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const maxLength = 280;
  const remaining = maxLength - content.length;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!content.trim() || submitting) return;

    setSubmitting(true);
    setError('');

    try {
      const res = await fetch('/api/snarks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ content }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error);
        return;
      }
      setContent('');
      onSnarkPosted(data);
    } catch {
      setError('Failed to post snark');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form className="compose-form" onSubmit={handleSubmit}>
      <textarea
        className="compose-input"
        placeholder="What's on your mind?"
        value={content}
        onChange={e => setContent(e.target.value)}
        maxLength={maxLength}
        rows={3}
      />
      <div className="compose-footer">
        <span className={`char-count ${remaining < 20 ? 'char-count-warning' : ''} ${remaining < 0 ? 'char-count-over' : ''}`}>
          {remaining}
        </span>
        {error && <span className="compose-error">{error}</span>}
        <button
          type="submit"
          className="compose-btn"
          disabled={!content.trim() || content.length > maxLength || submitting}
        >
          {submitting ? 'Posting...' : 'Snark'}
        </button>
      </div>
    </form>
  );
}
