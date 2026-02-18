const BASE = '/api'

function headers(token?: string): HeadersInit {
  const h: HeadersInit = { 'Content-Type': 'application/json' }
  if (token) h['Authorization'] = `Bearer ${token}`
  return h
}

export async function apiRegister(username: string, email: string, password: string) {
  const res = await fetch(`${BASE}/auth/register`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ username, email, password }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function apiLogin(username: string, password: string) {
  const res = await fetch(`${BASE}/auth/login`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function apiTimeline(token: string) {
  const res = await fetch(`${BASE}/posts/timeline`, { headers: headers(token) })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function apiCreatePost(token: string, content: string) {
  const res = await fetch(`${BASE}/posts`, {
    method: 'POST',
    headers: headers(token),
    body: JSON.stringify({ content }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function apiUserPosts(token: string, userId: number) {
  const res = await fetch(`${BASE}/posts/${userId}`, { headers: headers(token) })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function apiGetProfile(token: string, userId: number) {
  const res = await fetch(`${BASE}/users/${userId}/profile`, { headers: headers(token) })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function apiFollow(token: string, userId: number) {
  const res = await fetch(`${BASE}/users/${userId}/follow`, {
    method: 'POST',
    headers: headers(token),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function apiUnfollow(token: string, userId: number) {
  const res = await fetch(`${BASE}/users/${userId}/unfollow`, {
    method: 'DELETE',
    headers: headers(token),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function apiLike(token: string, postId: number) {
  const res = await fetch(`${BASE}/likes/${postId}`, {
    method: 'POST',
    headers: headers(token),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function apiUnlike(token: string, postId: number) {
  const res = await fetch(`${BASE}/likes/${postId}`, {
    method: 'DELETE',
    headers: headers(token),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function apiSearch(token: string, q: string) {
  const res = await fetch(`${BASE}/search?q=${encodeURIComponent(q)}`, {
    headers: headers(token),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
