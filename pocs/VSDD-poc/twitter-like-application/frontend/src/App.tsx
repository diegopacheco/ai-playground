import { useState, useEffect, createContext, useContext } from 'react'
import { createRouter, RouterProvider, createRoute, createRootRoute, Outlet, Link, useNavigate, useParams, useSearch } from '@tanstack/react-router'
import * as api from './api'

interface User {
  id: number
  username: string
  display_name: string
  bio: string
  created_at: number
}

interface Post {
  id: number
  author_id: number
  author_username: string
  author_display_name: string
  content: string
  image_url: string
  like_count: number
  liked_by_me: boolean
  created_at: number
}

const AuthContext = createContext<{ user: User | null; setUser: (u: User | null) => void }>({ user: null, setUser: () => {} })

function useAuth() { return useContext(AuthContext) }

function timeAgo(ts: number) {
  const diff = Math.floor(Date.now() / 1000) - ts
  if (diff < 60) return `${diff}s`
  if (diff < 3600) return `${Math.floor(diff / 60)}m`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`
  return `${Math.floor(diff / 86400)}d`
}

function PostCard({ post, onDelete }: { post: Post; onDelete?: () => void }) {
  const { user } = useAuth()
  const [liked, setLiked] = useState(post.liked_by_me)
  const [likeCount, setLikeCount] = useState(post.like_count)
  const navigate = useNavigate()

  async function toggleLike() {
    if (!user) return
    if (liked) {
      const r = await api.unlikePost(post.id)
      setLiked(false)
      setLikeCount(r.like_count)
    } else {
      const r = await api.likePost(post.id)
      setLiked(true)
      setLikeCount(r.like_count)
    }
  }

  async function handleDelete() {
    await api.deletePost(post.id)
    onDelete?.()
  }

  return (
    <div className="post">
      <div className="post-header">
        <a className="post-author" onClick={() => navigate({ to: '/profile/$id', params: { id: String(post.author_id) } })}>{post.author_display_name}</a>
        <span className="post-username">@{post.author_username}</span>
        <span className="post-time">{timeAgo(post.created_at)}</span>
      </div>
      <div className="post-content">{post.content}</div>
      {post.image_url && <img className="post-image" src={`/uploads/${post.image_url}`} alt="" />}
      <div className="post-actions">
        <button className={liked ? 'liked' : ''} onClick={toggleLike}>
          {liked ? 'Liked' : 'Like'} {likeCount}
        </button>
        {user && user.id === post.author_id && (
          <button onClick={handleDelete}>Delete</button>
        )}
      </div>
    </div>
  )
}

function PostList({ posts, onRefresh }: { posts: Post[]; onRefresh?: () => void }) {
  if (posts.length === 0) return <div className="empty">No posts yet</div>
  return <>{posts.map(p => <PostCard key={p.id} post={p} onDelete={onRefresh} />)}</>
}

function ComposeBox({ onPosted }: { onPosted: () => void }) {
  const [content, setContent] = useState('')
  const [image, setImage] = useState<File | null>(null)
  const [posting, setPosting] = useState(false)

  async function submit() {
    if (!content.trim() || posting) return
    setPosting(true)
    try {
      await api.createPost(content, image || undefined)
      setContent('')
      setImage(null)
      onPosted()
    } catch (e: any) {
      alert(e.message)
    }
    setPosting(false)
  }

  return (
    <div className="compose">
      <textarea placeholder="What's happening?" value={content} onChange={e => setContent(e.target.value)} maxLength={280} />
      <div className="compose-actions">
        <div>
          <input type="file" accept="image/jpeg,image/png" onChange={e => setImage(e.target.files?.[0] || null)} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span className="char-count">{content.length}/280</span>
          <button className="btn-primary" onClick={submit} disabled={!content.trim() || posting}>Post</button>
        </div>
      </div>
    </div>
  )
}

function LoginPage() {
  const { setUser } = useAuth()
  const navigate = useNavigate()
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [error, setError] = useState('')

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    try {
      if (mode === 'login') {
        const u = await api.login(username, password)
        setUser(u)
      } else {
        const u = await api.register(username, password, displayName)
        setUser(u)
      }
      navigate({ to: '/' })
    } catch (err: any) {
      setError(err.message)
    }
  }

  return (
    <div className="auth-page">
      <form className="auth-form" onSubmit={submit}>
        <h1>{mode === 'login' ? 'Login' : 'Register'}</h1>
        {error && <div className="error">{error}</div>}
        <div className="field">
          <label>Username</label>
          <input value={username} onChange={e => setUsername(e.target.value)} />
        </div>
        <div className="field">
          <label>Password</label>
          <input type="password" value={password} onChange={e => setPassword(e.target.value)} />
        </div>
        {mode === 'register' && (
          <div className="field">
            <label>Display Name</label>
            <input value={displayName} onChange={e => setDisplayName(e.target.value)} />
          </div>
        )}
        <button type="submit" className="btn-primary" style={{ width: '100%', padding: 12, fontSize: 16 }}>
          {mode === 'login' ? 'Login' : 'Register'}
        </button>
        <div className="switch">
          {mode === 'login' ? (
            <>No account? <a onClick={() => setMode('register')}>Register</a></>
          ) : (
            <>Have an account? <a onClick={() => setMode('login')}>Login</a></>
          )}
        </div>
      </form>
    </div>
  )
}

function HomePage() {
  const { user } = useAuth()
  const [posts, setPosts] = useState<Post[]>([])
  const [tab, setTab] = useState<'timeline' | 'all'>('timeline')

  function load() {
    if (tab === 'timeline' && user) {
      api.getTimeline().then(r => setPosts(r.posts))
    } else {
      api.getPosts().then(r => setPosts(r.posts))
    }
  }

  useEffect(() => { load() }, [tab, user])

  return (
    <>
      <div className="header"><h2>Home</h2></div>
      <div className="nav">
        <a className={tab === 'timeline' ? 'active' : ''} onClick={() => setTab('timeline')}>Timeline</a>
        <a className={tab === 'all' ? 'active' : ''} onClick={() => setTab('all')}>All Posts</a>
      </div>
      {user && <ComposeBox onPosted={load} />}
      <PostList posts={posts} onRefresh={load} />
    </>
  )
}

function ProfilePage() {
  const { user, setUser } = useAuth()
  const { id } = useParams({ from: '/profile/$id' })
  const [profile, setProfile] = useState<any>(null)
  const [posts, setPosts] = useState<Post[]>([])
  const [following, setFollowing] = useState(false)
  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [editBio, setEditBio] = useState('')

  function load() {
    api.getUserProfile(Number(id)).then(p => {
      setProfile(p)
      setEditName(p.display_name)
      setEditBio(p.bio)
    })
    api.getUserPosts(Number(id)).then(r => setPosts(r.posts))
    if (user) {
      api.getFollowing(user.id).then(r => {
        setFollowing(r.users.some((u: any) => u.id === Number(id)))
      })
    }
  }

  useEffect(() => { load() }, [id])

  async function toggleFollow() {
    if (!user) return
    if (following) {
      await api.unfollowUser(Number(id))
      setFollowing(false)
    } else {
      await api.followUser(Number(id))
      setFollowing(true)
    }
    load()
  }

  async function saveProfile() {
    await api.updateProfile(Number(id), editName, editBio)
    setEditing(false)
    const me = await api.getMe()
    setUser(me)
    load()
  }

  if (!profile) return <div className="loading">Loading...</div>

  const isMe = user && user.id === Number(id)

  return (
    <>
      <div className="header"><h2>Profile</h2></div>
      {editing ? (
        <div className="edit-profile">
          <div className="field">
            <label>Display Name</label>
            <input value={editName} onChange={e => setEditName(e.target.value)} />
          </div>
          <div className="field">
            <label>Bio</label>
            <textarea value={editBio} onChange={e => setEditBio(e.target.value)} />
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn-primary" onClick={saveProfile}>Save</button>
            <button className="btn-outline" onClick={() => setEditing(false)}>Cancel</button>
          </div>
        </div>
      ) : (
        <div className="profile-header">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div className="display-name">{profile.display_name}</div>
              <div className="username">@{profile.username}</div>
            </div>
            {isMe ? (
              <button className="btn-outline" onClick={() => setEditing(true)}>Edit Profile</button>
            ) : user ? (
              <button className={following ? 'btn-unfollow' : 'btn-follow'} onClick={toggleFollow}>
                {following ? 'Unfollow' : 'Follow'}
              </button>
            ) : null}
          </div>
          {profile.bio && <div className="bio">{profile.bio}</div>}
          <div className="stats">
            <span><strong>{profile.post_count}</strong> Posts</span>
            <span><strong>{profile.follower_count}</strong> Followers</span>
            <span><strong>{profile.following_count}</strong> Following</span>
          </div>
        </div>
      )}
      <PostList posts={posts} onRefresh={load} />
    </>
  )
}

function SearchPage() {
  const [query, setQuery] = useState('')
  const [type, setType] = useState<'posts' | 'users'>('posts')
  const [postResults, setPostResults] = useState<Post[]>([])
  const [userResults, setUserResults] = useState<any[]>([])
  const [searched, setSearched] = useState(false)
  const navigate = useNavigate()

  async function doSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setSearched(true)
    if (type === 'posts') {
      const r = await api.search(query, 'posts')
      setPostResults(r.posts)
      setUserResults([])
    } else {
      const r = await api.search(query, 'users')
      setUserResults(r.users)
      setPostResults([])
    }
  }

  return (
    <>
      <div className="header"><h2>Search</h2></div>
      <div className="search-bar">
        <form onSubmit={doSearch}>
          <input placeholder="Search..." value={query} onChange={e => setQuery(e.target.value)} />
          <select value={type} onChange={e => setType(e.target.value as 'posts' | 'users')}>
            <option value="posts">Posts</option>
            <option value="users">Users</option>
          </select>
          <button className="btn-primary" type="submit">Search</button>
        </form>
      </div>
      {type === 'posts' && <PostList posts={postResults} />}
      {type === 'users' && userResults.map(u => (
        <div key={u.id} className="user-card" onClick={() => navigate({ to: '/profile/$id', params: { id: String(u.id) } })}>
          <div className="user-info">
            <span className="name">{u.display_name}</span>
            <span className="uname">@{u.username}</span>
          </div>
        </div>
      ))}
      {searched && postResults.length === 0 && userResults.length === 0 && (
        <div className="empty">No results found</div>
      )}
    </>
  )
}

function HotPage() {
  const [posts, setPosts] = useState<Post[]>([])
  useEffect(() => { api.getHot().then(r => setPosts(r.posts)) }, [])
  return (
    <>
      <div className="header"><h2>Hot Topics</h2></div>
      <div className="section-title">Trending in the last 24 hours</div>
      <PostList posts={posts} />
    </>
  )
}

function Layout() {
  const { user, setUser } = useAuth()
  const navigate = useNavigate()

  async function handleLogout() {
    await api.logout()
    setUser(null)
    navigate({ to: '/login' })
  }

  return (
    <div className="container">
      <div className="nav">
        <Link to="/" className={({ isActive }: any) => isActive ? 'active' : ''}>Home</Link>
        <Link to="/search" className={({ isActive }: any) => isActive ? 'active' : ''}>Search</Link>
        <Link to="/hot" className={({ isActive }: any) => isActive ? 'active' : ''}>Hot</Link>
        {user ? (
          <>
            <Link to="/profile/$id" params={{ id: String(user.id) }} className={({ isActive }: any) => isActive ? 'active' : ''}>Profile</Link>
            <a onClick={handleLogout}>Logout</a>
          </>
        ) : (
          <Link to="/login" className={({ isActive }: any) => isActive ? 'active' : ''}>Login</Link>
        )}
      </div>
      <Outlet />
    </div>
  )
}

const rootRoute = createRootRoute({ component: Layout })
const homeRoute = createRoute({ getParentRoute: () => rootRoute, path: '/', component: HomePage })
const loginRoute = createRoute({ getParentRoute: () => rootRoute, path: '/login', component: LoginPage })
const profileRoute = createRoute({ getParentRoute: () => rootRoute, path: '/profile/$id', component: ProfilePage })
const searchRoute = createRoute({ getParentRoute: () => rootRoute, path: '/search', component: SearchPage })
const hotRoute = createRoute({ getParentRoute: () => rootRoute, path: '/hot', component: HotPage })

const routeTree = rootRoute.addChildren([homeRoute, loginRoute, profileRoute, searchRoute, hotRoute])
const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register { router: typeof router }
}

export default function App() {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.getMe().then(u => setUser(u)).catch(() => {}).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading">Loading...</div>

  return (
    <AuthContext.Provider value={{ user, setUser }}>
      <RouterProvider router={router} />
    </AuthContext.Provider>
  )
}
