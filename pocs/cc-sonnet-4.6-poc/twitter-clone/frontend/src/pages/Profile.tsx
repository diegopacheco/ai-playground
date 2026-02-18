import { useState, useEffect, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { apiGetProfile, apiUserPosts, apiFollow, apiUnfollow } from '../api'
import { useAuth } from '../AuthContext'
import { Post, UserProfile } from '../types'
import PostCard from '../components/Post'

const s: Record<string, React.CSSProperties> = {
  header: {
    border: '1px solid #2f3336',
    borderRadius: 12,
    padding: 24,
    marginBottom: 20,
  },
  topRow: { display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 16 },
  avatar: {
    width: 72,
    height: 72,
    borderRadius: '50%',
    background: '#1d9bf0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 700,
    fontSize: 28,
    color: '#fff',
  },
  username: { fontSize: 20, fontWeight: 800, color: '#e7e9ea', marginBottom: 4 },
  bio: { color: '#71767b', fontSize: 15, marginBottom: 16 },
  stats: { display: 'flex', gap: 24 },
  stat: { display: 'flex', gap: 4 },
  statNum: { fontWeight: 700, color: '#e7e9ea' },
  statLabel: { color: '#71767b' },
  followBtn: {
    borderRadius: 24,
    padding: '8px 20px',
    fontSize: 15,
    fontWeight: 700,
    cursor: 'pointer',
  },
  empty: { color: '#71767b', textAlign: 'center', padding: 40 },
}

export default function Profile() {
  const { userId } = useParams<{ userId: string }>()
  const { user } = useAuth()
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [posts, setPosts] = useState<Post[]>([])
  const [loading, setLoading] = useState(true)

  const uid = parseInt(userId ?? '0')

  const fetchData = useCallback(async () => {
    if (!user || !uid) return
    setLoading(true)
    try {
      const [prof, postsData] = await Promise.all([
        apiGetProfile(user.token, uid),
        apiUserPosts(user.token, uid),
      ])
      setProfile(prof)
      setPosts(postsData)
    } catch {
    } finally {
      setLoading(false)
    }
  }, [user, uid])

  useEffect(() => { fetchData() }, [fetchData])

  const toggleFollow = async () => {
    if (!user || !profile) return
    try {
      if (profile.is_following) {
        await apiUnfollow(user.token, uid)
        setProfile(p => p ? { ...p, is_following: false, followers_count: p.followers_count - 1 } : p)
      } else {
        await apiFollow(user.token, uid)
        setProfile(p => p ? { ...p, is_following: true, followers_count: p.followers_count + 1 } : p)
      }
    } catch {
    }
  }

  const updatePost = (updated: Post) => {
    setPosts(prev => prev.map(p => p.id === updated.id ? updated : p))
  }

  if (loading) return <p style={s.empty}>Loading...</p>
  if (!profile) return <p style={s.empty}>User not found</p>

  const isOwnProfile = user?.user_id === uid

  return (
    <div>
      <div style={s.header}>
        <div style={s.topRow}>
          <div style={s.avatar}>{profile.username[0].toUpperCase()}</div>
          {!isOwnProfile && (
            <button
              onClick={toggleFollow}
              style={{
                ...s.followBtn,
                background: profile.is_following ? 'transparent' : '#e7e9ea',
                color: profile.is_following ? '#e7e9ea' : '#000',
                border: profile.is_following ? '1px solid #536471' : 'none',
              }}
            >
              {profile.is_following ? 'Unfollow' : 'Follow'}
            </button>
          )}
        </div>
        <div style={s.username}>@{profile.username}</div>
        {profile.bio && <p style={s.bio}>{profile.bio}</p>}
        <div style={s.stats}>
          <div style={s.stat}>
            <span style={s.statNum}>{profile.following_count}</span>
            <span style={s.statLabel}> Following</span>
          </div>
          <div style={s.stat}>
            <span style={s.statNum}>{profile.followers_count}</span>
            <span style={s.statLabel}> Followers</span>
          </div>
          <div style={s.stat}>
            <span style={s.statNum}>{profile.posts_count}</span>
            <span style={s.statLabel}> Posts</span>
          </div>
        </div>
      </div>

      {posts.length === 0 && <p style={s.empty}>No posts yet</p>}
      {posts.map(post => (
        <PostCard key={post.id} post={post} onUpdate={updatePost} />
      ))}
    </div>
  )
}
