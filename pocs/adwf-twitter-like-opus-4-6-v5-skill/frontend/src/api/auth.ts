import client from "./client";
import type { AuthResponse } from "../types";

export const login = async (email: string, password: string): Promise<AuthResponse> => {
  const res = await client.post<AuthResponse>("/auth/login", { email, password });
  return res.data;
};

export const register = async (
  username: string,
  email: string,
  password: string
): Promise<AuthResponse> => {
  const res = await client.post<AuthResponse>("/auth/register", { username, email, password });
  return res.data;
};
