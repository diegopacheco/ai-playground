import client from "./client";
import type { Tweet } from "../types";

export const getFeed = async (): Promise<Tweet[]> => {
  const res = await client.get<Tweet[]>("/tweets/feed");
  return res.data;
};

export const getTweet = async (id: number): Promise<Tweet> => {
  const res = await client.get<Tweet>(`/tweets/${id}`);
  return res.data;
};

export const getUserTweets = async (userId: number): Promise<Tweet[]> => {
  const res = await client.get<Tweet[]>(`/users/${userId}/tweets`);
  return res.data;
};

export const createTweet = async (content: string): Promise<Tweet> => {
  const res = await client.post<Tweet>("/tweets", { content });
  return res.data;
};

export const deleteTweet = async (id: number): Promise<void> => {
  await client.delete(`/tweets/${id}`);
};

export const likeTweet = async (id: number): Promise<void> => {
  await client.post(`/tweets/${id}/like`);
};

export const unlikeTweet = async (id: number): Promise<void> => {
  await client.delete(`/tweets/${id}/like`);
};
