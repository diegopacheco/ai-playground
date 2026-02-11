import React, { useEffect, useState } from "react"
import { createRoot } from "react-dom/client"

const api = "http://127.0.0.1:3001"

function App() {
  const [activeUser, setActiveUser] = useState(1)
  const [username, setUsername] = useState("")
  const [content, setContent] = useState("")
  const [posts, setPosts] = useState([])
  const [error, setError] = useState("")

  async function loadPosts() {
    const res = await fetch(`${api}/api/posts`)
    const data = await res.json()
    setPosts(data)
  }

  useEffect(() => {
    loadPosts().catch(() => setError("failed to load posts"))
  }, [])

  async function createUser(e) {
    e.preventDefault()
    setError("")
    const res = await fetch(`${api}/api/users`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username })
    })
    if (!res.ok) {
      setError("failed to create user")
      return
    }
    const data = await res.json()
    setActiveUser(data.id)
    setUsername("")
  }

  async function createPost(e) {
    e.preventDefault()
    setError("")
    const res = await fetch(`${api}/api/posts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: activeUser, content })
    })
    if (!res.ok) {
      setError("failed to create post")
      return
    }
    setContent("")
    await loadPosts()
  }

  async function likePost(postId) {
    const res = await fetch(`${api}/api/posts/${postId}/likes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: activeUser })
    })
    if (!res.ok) {
      setError("failed to like post")
      return
    }
    await loadPosts()
  }

  return (
    <main className="page">
      <h1>Twitter Like Clone</h1>
      <section className="card">
        <h2>Create User</h2>
        <form onSubmit={createUser}>
          <input value={username} onChange={e => setUsername(e.target.value)} placeholder="username" required />
          <button type="submit">Create</button>
        </form>
        <p>Active user id: {activeUser}</p>
      </section>
      <section className="card">
        <h2>Create Post</h2>
        <form onSubmit={createPost}>
          <textarea value={content} onChange={e => setContent(e.target.value)} placeholder="what is happening" required maxLength={280} />
          <button type="submit">Post</button>
        </form>
      </section>
      <section className="card">
        <h2>Timeline</h2>
        {error && <p className="error">{error}</p>}
        <ul>
          {posts.map(p => (
            <li key={p.id}>
              <p>{p.content}</p>
              <small>{p.username} - {p.created_at} - likes: {p.likes}</small>
              <button onClick={() => likePost(p.id)}>Like</button>
            </li>
          ))}
        </ul>
      </section>
    </main>
  )
}

createRoot(document.getElementById("root")).render(<App />)
