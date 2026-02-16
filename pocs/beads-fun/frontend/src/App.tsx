import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import './App.css'

const API = 'http://localhost:8080/api'

interface Tweet {
  id: number
  username: string
  content: string
  likes: number
  created_at: string
}

function timeAgo(dateStr: string): string {
  const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  const days = Math.floor(hours / 24)
  return `${days}d`
}

function App() {
  const [username, setUsername] = useState('')
  const [content, setContent] = useState('')
  const queryClient = useQueryClient()

  const { data: tweets = [], isLoading } = useQuery<Tweet[]>({
    queryKey: ['tweets'],
    queryFn: () => fetch(`${API}/tweets`).then(r => r.json()),
    refetchInterval: 5000,
  })

  const createMutation = useMutation({
    mutationFn: (body: { username: string; content: string }) =>
      fetch(`${API}/tweets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tweets'] })
      setContent('')
    },
  })

  const likeMutation = useMutation({
    mutationFn: (id: number) =>
      fetch(`${API}/tweets/${id}/like`, { method: 'POST' }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['tweets'] }),
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) =>
      fetch(`${API}/tweets/${id}`, { method: 'DELETE' }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['tweets'] }),
  })

  const handleSubmit = () => {
    if (!username.trim() || !content.trim()) return
    createMutation.mutate({ username: username.trim(), content: content.trim() })
  }

  return (
    <div className="app">
      <div className="header">
        <h1>Home</h1>
      </div>

      <div className="compose-box">
        <input
          type="text"
          placeholder="Your username"
          value={username}
          onChange={e => setUsername(e.target.value)}
        />
        <textarea
          placeholder="What is happening?!"
          value={content}
          onChange={e => setContent(e.target.value)}
          maxLength={280}
        />
        <div className="compose-actions">
          <button
            className="btn-tweet"
            onClick={handleSubmit}
            disabled={!username.trim() || !content.trim() || createMutation.isPending}
          >
            Post
          </button>
        </div>
      </div>

      {isLoading && <div className="loading">Loading tweets...</div>}

      {!isLoading && tweets.length === 0 && (
        <div className="empty-state">No tweets yet. Be the first to post!</div>
      )}

      {tweets.map(tweet => (
        <div key={tweet.id} className="tweet">
          <div className="tweet-header">
            <div className="tweet-avatar">
              {tweet.username.charAt(0).toUpperCase()}
            </div>
            <span className="tweet-username">{tweet.username}</span>
            <span className="tweet-handle">@{tweet.username.toLowerCase()}</span>
            <span className="tweet-time">{timeAgo(tweet.created_at)}</span>
          </div>
          <div className="tweet-content">{tweet.content}</div>
          <div className="tweet-actions">
            <button
              className={`tweet-action ${tweet.likes > 0 ? 'liked' : ''}`}
              onClick={() => likeMutation.mutate(tweet.id)}
            >
              {tweet.likes > 0 ? '\u2764\uFE0F' : '\u2661'} {tweet.likes > 0 ? tweet.likes : ''}
            </button>
            <button
              className="tweet-action delete"
              onClick={() => deleteMutation.mutate(tweet.id)}
            >
              \u2715 Delete
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}

export default App
