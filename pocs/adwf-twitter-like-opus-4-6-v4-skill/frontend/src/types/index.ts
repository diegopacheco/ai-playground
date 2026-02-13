export interface User {
  id: number;
  username: string;
  email: string;
  created_at: string;
}

export interface Post {
  id: number;
  user_id: number;
  content: string;
  created_at: string;
  username?: string;
  like_count?: number;
  liked_by_me?: boolean;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface CreatePostRequest {
  content: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  page: number;
  per_page: number;
  total: number;
}

export interface LikeCount {
  count: number;
  liked_by_me: boolean;
}

export interface FollowInfo {
  followers_count: number;
  following_count: number;
  is_following: boolean;
}
