import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type {
  Post,
  Comment,
  User,
  CreatePostPayload,
  UpdatePostPayload,
  CreateCommentPayload,
} from "../types";

const API_BASE = "/api";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return res.json();
}

export function usePosts() {
  return useQuery<Post[]>({
    queryKey: ["posts"],
    queryFn: () => fetchJson<Post[]>("/posts"),
  });
}

export function usePost(id: number) {
  return useQuery<Post>({
    queryKey: ["posts", id],
    queryFn: () => fetchJson<Post>(`/posts/${id}`),
    enabled: !!id,
  });
}

export function useCreatePost() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: CreatePostPayload) =>
      fetchJson<Post>("/posts", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["posts"] });
    },
  });
}

export function useUpdatePost(id: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: UpdatePostPayload) =>
      fetchJson<Post>(`/posts/${id}`, {
        method: "PUT",
        body: JSON.stringify(payload),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["posts"] });
      queryClient.invalidateQueries({ queryKey: ["posts", id] });
    },
  });
}

export function useDeletePost() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) =>
      fetchJson<void>(`/posts/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["posts"] });
    },
  });
}

export function useComments(postId: number) {
  return useQuery<Comment[]>({
    queryKey: ["comments", postId],
    queryFn: () => fetchJson<Comment[]>(`/posts/${postId}/comments`),
    enabled: !!postId,
  });
}

export function useCreateComment(postId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: CreateCommentPayload) =>
      fetchJson<Comment>(`/posts/${postId}/comments`, {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["comments", postId] });
    },
  });
}

export function useUser(id: number) {
  return useQuery<User>({
    queryKey: ["users", id],
    queryFn: () => fetchJson<User>(`/users/${id}`),
    enabled: !!id,
  });
}

export function useCurrentUser() {
  return useQuery<User>({
    queryKey: ["users", "me"],
    queryFn: () => fetchJson<User>("/users/me"),
  });
}
