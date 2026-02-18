import { Link } from 'react-router-dom'
import { Post } from '../types'
import { apiLike, apiUnlike } from '../api'
import { useAuth } from '../AuthContext'

interface Props {
  post: Post
  onUpdate: (updated: Post) => void
}

const s: Record<string, React.CSSProperties> = {
  card: {
    border: '1px solid #2f3336',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    transition: 'background 0.15s',
  },
  header: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: '50%',
    background: '#1d9bf0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 700,
    fontSize: 16,
    color: '#fff',
    flexShrink: 0,
  },
  username: { fontWeight: 700, color: '#e7e9ea', textDecoration: 'none', fontSize: 15 },
  date: { color: '#71767b', fontSize: 13 },
  content: { fontSize: 15, lineHeight: 1.5, color: '#e7e9ea', marginBottom: 12 },
  actions: { display: 'flex', gap: 24 },
  likeBtn: {
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    fontSize: 14,
    borderRadius: 20,
    padding: '4px 8px',
  },
}

function formatDate(d: string | null) {
  if (!d) return ''
  const date = new Date(d + 'Z')
  return date.toLocaleString()
}

export default function PostCard({ post, onUpdate }: Props) {
  const { user } = useAuth()

  const toggleLike = async () => {
    if (!user) return
    try {
      if (post.liked_by_me) {
        await apiUnlike(user.token, post.id)
        onUpdate({ ...post, liked_by_me: false, likes_count: post.likes_count - 1 })
      } else {
        await apiLike(user.token, post.id)
        onUpdate({ ...post, liked_by_me: true, likes_count: post.likes_count + 1 })
      }
    } catch {
    }
  }

  return (
    <div style={s.card}>
      <div style={s.header}>
        <div style={s.avatar}>{post.username[0].toUpperCase()}</div>
        <div>
          <Link to={`/profile/${post.user_id}`} style={s.username}>@{post.username}</Link>
          <div style={s.date}>{formatDate(post.created_at)}</div>
        </div>
      </div>
      <p style={s.content}>{post.content}</p>
      <div style={s.actions}>
        <button
          onClick={toggleLike}
          style={{
            ...s.likeBtn,
            color: post.liked_by_me ? '#f91880' : '#71767b',
          }}
        >
          {post.liked_by_me ? '♥' : '♡'} {post.likes_count}
        </button>
      </div>
    </div>
  )
}
