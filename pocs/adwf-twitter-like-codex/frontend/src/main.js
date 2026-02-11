const api = "http://127.0.0.1:3001"

const root = document.getElementById("root")
let activeUser = 1
let posts = []
let error = ""

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
}

function render() {
  const items = posts
    .map(
      p => `<li data-post-id="${p.id}"><p>${escapeHtml(p.content)}</p><small>${escapeHtml(p.username)} - ${escapeHtml(p.created_at)} - likes: ${p.likes}</small><button class="like-btn" data-post-id="${p.id}">Like</button></li>`
    )
    .join("")

  root.innerHTML = `<main class="page"><h1>Twitter Like Clone</h1><section class="card"><h2>Create User</h2><form id="user-form"><input id="username" placeholder="username" required /><button type="submit">Create</button></form><p id="active-user">Active user id: ${activeUser}</p></section><section class="card"><h2>Create Post</h2><form id="post-form"><textarea id="content" placeholder="what is happening" required maxlength="280"></textarea><button type="submit">Post</button></form></section><section class="card"><h2>Timeline</h2>${error ? `<p class="error">${escapeHtml(error)}</p>` : ""}<ul>${items}</ul></section></main>`

  const userForm = document.getElementById("user-form")
  const postForm = document.getElementById("post-form")

  userForm.addEventListener("submit", onCreateUser)
  postForm.addEventListener("submit", onCreatePost)
  for (const button of document.querySelectorAll(".like-btn")) {
    button.addEventListener("click", onLikePost)
  }
}

async function loadPosts() {
  const res = await fetch(`${api}/api/posts`)
  if (!res.ok) {
    throw new Error("failed")
  }
  posts = await res.json()
  render()
}

async function onCreateUser(event) {
  event.preventDefault()
  error = ""
  const input = document.getElementById("username")
  const username = input.value
  const res = await fetch(`${api}/api/users`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username })
  })
  if (!res.ok) {
    error = "failed to create user"
    render()
    return
  }
  const data = await res.json()
  activeUser = data.id
  input.value = ""
  render()
}

async function onCreatePost(event) {
  event.preventDefault()
  error = ""
  const input = document.getElementById("content")
  const content = input.value
  const res = await fetch(`${api}/api/posts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: activeUser, content })
  })
  if (!res.ok) {
    error = "failed to create post"
    render()
    return
  }
  input.value = ""
  await loadPosts()
}

async function onLikePost(event) {
  error = ""
  const postId = Number(event.currentTarget.dataset.postId)
  const res = await fetch(`${api}/api/posts/${postId}/likes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: activeUser })
  })
  if (!res.ok) {
    error = "failed to like post"
    render()
    return
  }
  await loadPosts()
}

render()
loadPosts().catch(() => {
  error = "failed to load posts"
  render()
})
