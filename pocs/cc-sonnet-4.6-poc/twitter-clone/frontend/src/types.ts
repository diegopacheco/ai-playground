export interface Post {
  id: number
  content: string
  created_at: string | null
  user_id: number
  username: string
  likes_count: number
  liked_by_me: boolean
}

export interface UserProfile {
  id: number
  username: string
  bio: string | null
  followers_count: number
  following_count: number
  posts_count: number
  is_following: boolean
}

export interface AuthUser {
  token: string
  user_id: number
  username: string
}

export interface SearchResult {
  posts: Post[]
  users: Array<{ id: number; username: string; bio: string | null }>
}
