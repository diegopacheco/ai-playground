import { useState, useEffect, useCallback } from 'react'
import { apiTimeline, apiCreatePost } from '../api'
import { useAuth } from '../AuthContext'
import { Post } from '../types'
import PostCard from '../components/Post'

const s: Record<string, React.CSSProperties> = {
  compose: {
    border: '1px solid #2f3336',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  textarea: {
    width: '100%',
    background: 'transparent',
    border: 'none',
    color: '#e7e9ea',
    fontSize: 18,
    resize: 'none',
    outline: 'none',
    fontFamily: 'inherit',
    minHeight: 80,
  },
  composeFooter: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTop: '1px solid #2f3336',
    paddingTop: 12,
    marginTop: 8,
  },
  charCount: { fontSize: 14, color: '#71767b' },
  postBtn: {
    background: '#1d9bf0',
    border: 'none',
    borderRadius: 24,
    padding: '8px 20px',
    color: '#fff',
    fontSize: 15,
    fontWeight: 700,
    cursor: 'pointer',
  },
  empty: { textAlign: 'center', color: '#71767b', padding: 40 },
}

export default function Timeline() {
  const { user } = useAuth()
  const [posts, setPosts] = useState<Post[]>([])
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const fetchTimeline = useCallback(async () => {
    if (!user) return
    setLoading(true)
    try {
      const data = await apiTimeline(user.token)
      setPosts(data)
    } catch {
    } finally {
      setLoading(false)
    }
  }, [user])

  useEffect(() => { fetchTimeline() }, [fetchTimeline])

  const handlePost = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!user || !content.trim()) return
    setSubmitting(true)
    try {
      await apiCreatePost(user.token, content.trim())
      setContent('')
      await fetchTimeline()
    } catch {
    } finally {
      setSubmitting(false)
    }
  }

  const updatePost = (updated: Post) => {
    setPosts(prev => prev.map(p => p.id === updated.id ? updated : p))
  }

  return (
    <div>
      <div style={s.compose}>
        <form onSubmit={handlePost}>
          <textarea
            style={s.textarea}
            placeholder="What is happening?!"
            value={content}
            onChange={e => setContent(e.target.value)}
            maxLength={280}
          />
          <div style={s.composeFooter}>
            <span style={{ ...s.charCount, color: content.length > 260 ? '#f4212e' : '#71767b' }}>
              {280 - content.length}
            </span>
            <button
              style={s.postBtn}
              type="submit"
              disabled={submitting || !content.trim()}
            >
              {submitting ? 'Posting...' : 'Post'}
            </button>
          </div>
        </form>
      </div>

      {loading && <p style={s.empty}>Loading...</p>}
      {!loading && posts.length === 0 && (
        <p style={s.empty}>No posts yet. Follow some people or post something!</p>
      )}
      {posts.map(post => (
        <PostCard key={post.id} post={post} onUpdate={updatePost} />
      ))}
    </div>
  )
}
