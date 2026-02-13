import { apiClient } from "./client";
import type {
  User,
  LoginRequest,
  RegisterRequest,
  AuthResponse,
} from "../types";

export function registerUser(data: RegisterRequest): Promise<AuthResponse> {
  return apiClient<AuthResponse>("/api/users", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function loginUser(data: LoginRequest): Promise<AuthResponse> {
  return apiClient<AuthResponse>("/api/users/login", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function getUser(id: number): Promise<User> {
  return apiClient<User>(`/api/users/${id}`);
}

export function getUserFollowers(id: number): Promise<User[]> {
  return apiClient<User[]>(`/api/users/${id}/followers`);
}

export function getUserFollowing(id: number): Promise<User[]> {
  return apiClient<User[]>(`/api/users/${id}/following`);
}
