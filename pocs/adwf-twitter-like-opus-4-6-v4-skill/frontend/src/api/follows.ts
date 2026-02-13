import { apiClient } from "./client";

export function followUser(id: number): Promise<void> {
  return apiClient<void>(`/api/users/${id}/follow`, {
    method: "POST",
  });
}

export function unfollowUser(id: number): Promise<void> {
  return apiClient<void>(`/api/users/${id}/follow`, {
    method: "DELETE",
  });
}
