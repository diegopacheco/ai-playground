export interface User {
  id: number;
  username: string;
  email: string;
  display_name?: string;
  bio?: string;
  created_at: string;
  updated_at: string;
}

export interface Tweet {
  id: number;
  user_id: number;
  content: string;
  created_at: string;
  updated_at: string;
  author_username?: string;
  author_display_name?: string | null;
  likes_count?: number;
  retweets_count?: number;
  comments_count?: number;
  is_liked?: boolean;
  is_retweeted?: boolean;
}

export interface Comment {
  id: number;
  user_id: number;
  tweet_id: number;
  content: string;
  created_at: string;
  updated_at: string;
  author_username?: string;
  author_display_name?: string | null;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterCredentials {
  username: string;
  email: string;
  password: string;
}
