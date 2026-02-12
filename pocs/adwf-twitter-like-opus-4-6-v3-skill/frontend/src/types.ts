export interface UserResponse {
  id: string;
  username: string;
  email: string;
  created_at: string;
}

export interface PostResponse {
  id: string;
  user_id: string;
  username: string;
  content: string;
  likes_count: number;
  liked_by_me: boolean;
  created_at: string;
}

export interface AuthResponse {
  token: string;
  user: UserResponse;
}

export interface ProfileData {
  user: UserResponse;
  followers_count: number;
  following_count: number;
  is_following: boolean;
}
