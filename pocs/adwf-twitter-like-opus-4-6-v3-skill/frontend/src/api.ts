import { getToken } from "./auth";
import type {
  AuthResponse,
  PostResponse,
  UserResponse,
  ProfileData,
} from "./types";

const BASE = "/api";

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  const res = await fetch(`${BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export function login(email: string, password: string): Promise<AuthResponse> {
  return request("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function register(
  username: string,
  email: string,
  password: string
): Promise<AuthResponse> {
  return request("/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, email, password }),
  });
}

export function getMe(): Promise<UserResponse> {
  return request("/auth/me");
}

export function getTimeline(): Promise<PostResponse[]> {
  return request("/posts");
}

export function getPost(id: string): Promise<PostResponse> {
  return request(`/posts/${id}`);
}

export function createPost(content: string): Promise<PostResponse> {
  return request("/posts", {
    method: "POST",
    body: JSON.stringify({ content }),
  });
}

export function deletePost(id: string): Promise<void> {
  return request(`/posts/${id}`, { method: "DELETE" });
}

export function likePost(id: string): Promise<void> {
  return request(`/posts/${id}/like`, { method: "POST" });
}

export function unlikePost(id: string): Promise<void> {
  return request(`/posts/${id}/like`, { method: "DELETE" });
}

export function getUserProfile(id: string): Promise<ProfileData> {
  return request(`/users/${id}`);
}

export function getUserPosts(id: string): Promise<PostResponse[]> {
  return request(`/users/${id}/posts`);
}

export function getFollowers(id: string): Promise<UserResponse[]> {
  return request(`/users/${id}/followers`);
}

export function getFollowing(id: string): Promise<UserResponse[]> {
  return request(`/users/${id}/following`);
}

export function followUser(id: string): Promise<void> {
  return request(`/users/${id}/follow`, { method: "POST" });
}

export function unfollowUser(id: string): Promise<void> {
  return request(`/users/${id}/follow`, { method: "DELETE" });
}
