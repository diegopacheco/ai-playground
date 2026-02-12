import type { UserResponse } from "./types";

const TOKEN_KEY = "token";
const USER_KEY = "user";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function getUser(): UserResponse | null {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  return JSON.parse(raw);
}

export function setAuth(token: string, user: UserResponse) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function isAuthenticated(): boolean {
  return !!getToken();
}
