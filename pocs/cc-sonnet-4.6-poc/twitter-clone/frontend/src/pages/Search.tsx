import { useState } from 'react'
import { Link } from 'react-router-dom'
import { apiSearch } from '../api'
import { useAuth } from '../AuthContext'
import { Post, SearchResult } from '../types'
import PostCard from '../components/Post'

const s: Record<string, React.CSSProperties> = {
  searchBar: { display: 'flex', gap: 8, marginBottom: 24 },
  input: {
    flex: 1,
    background: '#16181c',
    border: '1px solid #536471',
    borderRadius: 24,
    padding: '10px 20px',
    color: '#e7e9ea',
    fontSize: 16,
    outline: 'none',
  },
  btn: {
    background: '#1d9bf0',
    border: 'none',
    borderRadius: 24,
    padding: '10px 20px',
    color: '#fff',
    fontWeight: 700,
    cursor: 'pointer',
  },
  section: { marginBottom: 24 },
  sectionTitle: { fontSize: 18, fontWeight: 700, marginBottom: 12, color: '#e7e9ea' },
  userCard: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: 12,
    border: '1px solid #2f3336',
    borderRadius: 12,
    marginBottom: 8,
    textDecoration: 'none',
    color: '#e7e9ea',
  },
  avatar: {
    width: 44,
    height: 44,
    borderRadius: '50%',
    background: '#1d9bf0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 700,
    fontSize: 18,
    color: '#fff',
    flexShrink: 0,
  },
  empty: { color: '#71767b', textAlign: 'center', padding: 24 },
}

export default function Search() {
  const { user } = useAuth()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [posts, setPosts] = useState<Post[]>([])

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!user || !query.trim()) return
    setLoading(true)
    try {
      const data = await apiSearch(user.token, query)
      setResults(data)
      setPosts(data.posts)
    } catch {
    } finally {
      setLoading(false)
    }
  }

  const updatePost = (updated: Post) => {
    setPosts(prev => prev.map(p => p.id === updated.id ? updated : p))
  }

  return (
    <div>
      <form onSubmit={handleSearch} style={s.searchBar}>
        <input
          style={s.input}
          placeholder="Search posts and users..."
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        <button style={s.btn} type="submit" disabled={loading}>
          {loading ? '...' : 'Search'}
        </button>
      </form>

      {results && (
        <>
          {results.users.length > 0 && (
            <div style={s.section}>
              <h2 style={s.sectionTitle}>People</h2>
              {results.users.map(u => (
                <Link key={u.id} to={`/profile/${u.id}`} style={s.userCard}>
                  <div style={s.avatar}>{u.username[0].toUpperCase()}</div>
                  <div>
                    <div style={{ fontWeight: 700 }}>@{u.username}</div>
                    {u.bio && <div style={{ color: '#71767b', fontSize: 14 }}>{u.bio}</div>}
                  </div>
                </Link>
              ))}
            </div>
          )}

          {posts.length > 0 && (
            <div style={s.section}>
              <h2 style={s.sectionTitle}>Posts</h2>
              {posts.map(post => (
                <PostCard key={post.id} post={post} onUpdate={updatePost} />
              ))}
            </div>
          )}

          {results.users.length === 0 && results.posts.length === 0 && (
            <p style={s.empty}>No results for "{query}"</p>
          )}
        </>
      )}
    </div>
  )
}
