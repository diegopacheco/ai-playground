import client from "./client";
import type { UserProfile, User } from "../types";

export const getUser = async (id: number): Promise<UserProfile> => {
  const res = await client.get<UserProfile>(`/users/${id}`);
  return res.data;
};

export const getFollowers = async (id: number): Promise<User[]> => {
  const res = await client.get<User[]>(`/users/${id}/followers`);
  return res.data;
};

export const getFollowing = async (id: number): Promise<User[]> => {
  const res = await client.get<User[]>(`/users/${id}/following`);
  return res.data;
};

export const followUser = async (id: number): Promise<void> => {
  await client.post(`/users/${id}/follow`);
};

export const unfollowUser = async (id: number): Promise<void> => {
  await client.delete(`/users/${id}/follow`);
};
