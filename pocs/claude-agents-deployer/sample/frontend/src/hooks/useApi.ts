import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type {
  Post,
  Comment,
  User,
  CreatePostPayload,
  UpdatePostPayload,
  CreateCommentPayload,
  AppSettings,
  UpdateSettingsPayload,
} from "../types";

const API_BASE = "/api";
const LOCAL_SETTINGS_KEY = "blog_platform_admin_settings";

function defaultSettings(): AppSettings {
  return {
    id: 1,
    commentsEnabled: true,
    backgroundTheme: "classic",
    updatedAt: new Date().toISOString(),
  };
}

function readLocalSettings(): AppSettings {
  if (typeof window === "undefined") return defaultSettings();
  const raw = window.localStorage.getItem(LOCAL_SETTINGS_KEY);
  if (!raw) return defaultSettings();
  try {
    return JSON.parse(raw) as AppSettings;
  } catch {
    return defaultSettings();
  }
}

function writeLocalSettings(settings: AppSettings): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LOCAL_SETTINGS_KEY, JSON.stringify(settings));
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error: ${res.status} ${body}`);
  }
  return res.json();
}

export function usePosts() {
  return useQuery<Post[]>({
    queryKey: ["posts"],
    queryFn: () => fetchJson<Post[]>("/posts"),
  });
}

export function usePost(id: string) {
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

export function useUpdatePost(id: string) {
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
    mutationFn: (id: string) =>
      fetchJson<void>(`/posts/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["posts"] });
    },
  });
}

export function useComments(postId: string) {
  return useQuery<Comment[]>({
    queryKey: ["comments", postId],
    queryFn: () => fetchJson<Comment[]>(`/posts/${postId}/comments`),
    enabled: !!postId,
  });
}

export function useCreateComment(postId: string) {
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

export function useUsers() {
  return useQuery<User[]>({
    queryKey: ["users"],
    queryFn: () => fetchJson<User[]>("/users"),
  });
}

export function useSettings() {
  return useQuery<AppSettings>({
    queryKey: ["settings"],
    queryFn: async () => {
      try {
        const settings = await fetchJson<AppSettings>("/settings");
        writeLocalSettings(settings);
        return settings;
      } catch (e) {
        if (e instanceof Error && e.message.includes("API error: 404")) {
          return readLocalSettings();
        }
        throw e;
      }
    },
  });
}

export function useUpdateSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: UpdateSettingsPayload) => {
      try {
        const settings = await fetchJson<AppSettings>("/settings", {
          method: "PUT",
          body: JSON.stringify(payload),
        });
        writeLocalSettings(settings);
        return settings;
      } catch (e) {
        if (e instanceof Error && e.message.includes("API error: 404")) {
          const current = readLocalSettings();
          const next: AppSettings = {
            ...current,
            commentsEnabled:
              payload.commentsEnabled ?? current.commentsEnabled,
            backgroundTheme:
              payload.backgroundTheme ?? current.backgroundTheme,
            updatedAt: new Date().toISOString(),
          };
          writeLocalSettings(next);
          return next;
        }
        throw e;
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings"] });
      queryClient.invalidateQueries({ queryKey: ["comments"] });
    },
  });
}
