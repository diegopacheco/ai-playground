import { apiClient } from "./client";
import type { Post, CreatePostRequest } from "../types";

export function createPost(data: CreatePostRequest): Promise<Post> {
  return apiClient<Post>("/api/posts", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function getPost(id: number): Promise<Post> {
  return apiClient<Post>(`/api/posts/${id}`);
}

export function deletePost(id: number): Promise<void> {
  return apiClient<void>(`/api/posts/${id}`, {
    method: "DELETE",
  });
}

export function getPosts(page: number = 1): Promise<Post[]> {
  return apiClient<Post[]>(`/api/posts?page=${page}`);
}

export function getFeed(page: number = 1): Promise<Post[]> {
  return apiClient<Post[]>(`/api/feed?page=${page}`);
}

export function getUserPosts(userId: number, page: number = 1): Promise<Post[]> {
  return apiClient<Post[]>(`/api/posts?user_id=${userId}&page=${page}`);
}
