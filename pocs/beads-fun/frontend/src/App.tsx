import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import './App.css'

const API = 'http://localhost:8080/api'

interface Tweet {
  id: number
  username: string
  content: string
  likes: number
  image_url: string | null
  created_at: string
}

interface AuthState {
  token: string
  username: string
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

function AuthForm({ onAuth }: { onAuth: (auth: AuthState) => void }) {
  const [isLogin, setIsLogin] = useState(true)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = async () => {
    setError('')
    const endpoint = isLogin ? '/login' : '/register'
    const res = await fetch(`${API}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    const data = await res.json()
    if (!res.ok) {
      setError(data.error || 'Something went wrong')
      return
    }
    localStorage.setItem('auth', JSON.stringify(data))
    onAuth(data)
  }

  return (
    <div className="auth-container">
      <div className="auth-box">
        <h1 className="auth-title">Twitter Clone</h1>
        <h2>{isLogin ? 'Login' : 'Register'}</h2>
        {error && <div className="auth-error">{error}</div>}
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={e => setUsername(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSubmit()}
        />
        <button className="btn-tweet" onClick={handleSubmit}>
          {isLogin ? 'Login' : 'Register'}
        </button>
        <button className="btn-switch" onClick={() => { setIsLogin(!isLogin); setError('') }}>
          {isLogin ? 'Need an account? Register' : 'Have an account? Login'}
        </button>
      </div>
    </div>
  )
}

function TweetImage({ url }: { url: string }) {
  const src = url.startsWith('/') ? `http://localhost:8080${url}` : url
  return <img className="tweet-image" src={src} alt="tweet attachment" />
}

function Feed({ auth, onLogout }: { auth: AuthState; onLogout: () => void }) {
  const [content, setContent] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [activeSearch, setActiveSearch] = useState('')
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const queryClient = useQueryClient()

  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${auth.token}`,
  }

  const { data: tweets = [], isLoading } = useQuery<Tweet[]>({
    queryKey: ['tweets', activeSearch],
    queryFn: () => {
      const url = activeSearch
        ? `${API}/tweets/search?q=${encodeURIComponent(activeSearch)}`
        : `${API}/tweets`
      return fetch(url).then(r => r.json())
    },
    refetchInterval: 5000,
  })

  const createMutation = useMutation({
    mutationFn: (body: { content: string; image_url: string | null }) =>
      fetch(`${API}/tweets`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ ...body, username: auth.username }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tweets'] })
      setContent('')
      setImageUrl(null)
    },
  })

  const likeMutation = useMutation({
    mutationFn: (id: number) =>
      fetch(`${API}/tweets/${id}/like`, { method: 'POST' }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['tweets'] }),
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) =>
      fetch(`${API}/tweets/${id}`, { method: 'DELETE', headers }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['tweets'] }),
  })

  const handleSubmit = () => {
    if (!content.trim()) return
    createMutation.mutate({ content: content.trim(), image_url: imageUrl })
  }

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)
    const res = await fetch(`${API}/upload`, { method: 'POST', body: formData })
    const data = await res.json()
    setImageUrl(data.url)
    setUploading(false)
  }

  const handleSearch = () => {
    setActiveSearch(searchQuery.trim())
  }

  const clearSearch = () => {
    setSearchQuery('')
    setActiveSearch('')
  }

  return (
    <div className="app">
      <div className="header">
        <div className="header-row">
          <h1>Home</h1>
          <div className="header-user">
            <span>@{auth.username}</span>
            <button className="btn-logout" onClick={onLogout}>Logout</button>
          </div>
        </div>
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search tweets..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
          />
          <button className="btn-search" onClick={handleSearch}>Search</button>
          {activeSearch && (
            <button className="btn-clear" onClick={clearSearch}>Clear</button>
          )}
        </div>
        {activeSearch && (
          <div className="search-info">Results for: "{activeSearch}"</div>
        )}
      </div>

      <div className="compose-box">
        <textarea
          placeholder="What is happening?!"
          value={content}
          onChange={e => setContent(e.target.value)}
          maxLength={280}
        />
        {imageUrl && (
          <div className="image-preview">
            <TweetImage url={imageUrl} />
            <button className="btn-remove-image" onClick={() => setImageUrl(null)}>Remove</button>
          </div>
        )}
        <div className="compose-actions">
          <label className="btn-image-upload">
            {uploading ? 'Uploading...' : 'Add Image'}
            <input type="file" accept="image/*" onChange={handleImageUpload} hidden />
          </label>
          <button
            className="btn-tweet"
            onClick={handleSubmit}
            disabled={!content.trim() || createMutation.isPending}
          >
            Post
          </button>
        </div>
      </div>

      {isLoading && <div className="loading">Loading tweets...</div>}

      {!isLoading && tweets.length === 0 && (
        <div className="empty-state">
          {activeSearch ? 'No tweets match your search.' : 'No tweets yet. Be the first to post!'}
        </div>
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
          {tweet.image_url && (
            <div className="tweet-image-container">
              <TweetImage url={tweet.image_url} />
            </div>
          )}
          <div className="tweet-actions">
            <button
              className={`tweet-action ${tweet.likes > 0 ? 'liked' : ''}`}
              onClick={() => likeMutation.mutate(tweet.id)}
            >
              {tweet.likes > 0 ? '\u2764\uFE0F' : '\u2661'} {tweet.likes > 0 ? tweet.likes : ''}
            </button>
            {tweet.username === auth.username && (
              <button
                className="tweet-action delete"
                onClick={() => deleteMutation.mutate(tweet.id)}
              >
                \u2715 Delete
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function App() {
  const [auth, setAuth] = useState<AuthState | null>(() => {
    const stored = localStorage.getItem('auth')
    return stored ? JSON.parse(stored) : null
  })

  const handleLogout = () => {
    localStorage.removeItem('auth')
    setAuth(null)
  }

  if (!auth) {
    return <AuthForm onAuth={setAuth} />
  }

  return <Feed auth={auth} onLogout={handleLogout} />
}

export default App
