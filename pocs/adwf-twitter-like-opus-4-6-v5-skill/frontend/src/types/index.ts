export interface User {
  id: number;
  username: string;
  email: string;
  display_name: string;
  bio: string;
  created_at: string;
}

export interface Tweet {
  id: number;
  user_id: number;
  content: string;
  created_at: string;
  username: string;
  display_name: string;
  like_count: number;
  liked_by_me: boolean;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface UserProfile extends User {
  follower_count: number;
  following_count: number;
  is_following: boolean;
}
