import { apiClient } from "./client";
import type { LikeCount } from "../types";

export function likePost(id: number): Promise<void> {
  return apiClient<void>(`/api/posts/${id}/like`, {
    method: "POST",
  });
}

export function unlikePost(id: number): Promise<void> {
  return apiClient<void>(`/api/posts/${id}/like`, {
    method: "DELETE",
  });
}

export function getLikeCount(id: number): Promise<LikeCount> {
  return apiClient<LikeCount>(`/api/posts/${id}/likes`);
}
