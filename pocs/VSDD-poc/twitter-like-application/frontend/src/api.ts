const BASE = '/api'

async function request(path: string, opts: RequestInit = {}) {
  const headers: Record<string, string> = { ...opts.headers as Record<string, string> }
  if (opts.body) headers['Content-Type'] = 'application/json'
  const res = await fetch(`${BASE}${path}`, { credentials: 'include', headers, ...opts })
  if (res.status === 204) return null
  let data
  try { data = await res.json() } catch { data = { error: res.statusText } }
  if (!res.ok) throw new Error(data.error || 'Request failed')
  return data
}

export async function register(username: string, password: string, display_name: string) {
  return request('/auth/register', { method: 'POST', body: JSON.stringify({ username, password, display_name }) })
}

export async function login(username: string, password: string) {
  return request('/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) })
}

export async function logout() {
  return request('/auth/logout', { method: 'POST' })
}

export async function getMe() {
  return request('/auth/me')
}

export async function getPosts(page = 1, limit = 20) {
  return request(`/posts?page=${page}&limit=${limit}`)
}

export async function getPost(id: number) {
  return request(`/posts/${id}`)
}

export async function createPost(content: string, image?: File) {
  const form = new FormData()
  form.append('content', content)
  if (image) form.append('image', image)
  const res = await fetch(`${BASE}/posts`, { method: 'POST', credentials: 'include', body: form })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'Request failed')
  return data
}

export async function deletePost(id: number) {
  return request(`/posts/${id}`, { method: 'DELETE' })
}

export async function likePost(id: number) {
  return request(`/posts/${id}/like`, { method: 'POST' })
}

export async function unlikePost(id: number) {
  return request(`/posts/${id}/like`, { method: 'DELETE' })
}

export async function followUser(id: number) {
  return request(`/users/${id}/follow`, { method: 'POST' })
}

export async function unfollowUser(id: number) {
  return request(`/users/${id}/follow`, { method: 'DELETE' })
}

export async function getFollowers(id: number) {
  return request(`/users/${id}/followers`)
}

export async function getFollowing(id: number) {
  return request(`/users/${id}/following`)
}

export async function getTimeline(page = 1, limit = 20) {
  return request(`/timeline?page=${page}&limit=${limit}`)
}

export async function getUserProfile(id: number) {
  return request(`/users/${id}`)
}

export async function updateProfile(id: number, display_name: string, bio: string) {
  return request(`/users/${id}`, { method: 'PUT', body: JSON.stringify({ display_name, bio }) })
}

export async function getUserPosts(id: number, page = 1, limit = 20) {
  return request(`/users/${id}/posts?page=${page}&limit=${limit}`)
}

export async function search(q: string, type: string, page = 1, limit = 20) {
  return request(`/search?q=${encodeURIComponent(q)}&type=${type}&page=${page}&limit=${limit}`)
}

export async function getHot() {
  return request('/hot')
}
