import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";
import type { PostResponse } from "../types";

export function useTimeline() {
  return useQuery({
    queryKey: ["timeline"],
    queryFn: api.getTimeline,
  });
}

export function usePost(id: string) {
  return useQuery({
    queryKey: ["post", id],
    queryFn: () => api.getPost(id),
  });
}

export function useUserPosts(userId: string) {
  return useQuery({
    queryKey: ["userPosts", userId],
    queryFn: () => api.getUserPosts(userId),
  });
}

export function useCreatePost() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (content: string) => api.createPost(content),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["timeline"] });
    },
  });
}

export function useToggleLike() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (post: PostResponse) =>
      post.liked_by_me ? api.unlikePost(post.id) : api.likePost(post.id),
    onMutate: async (post: PostResponse) => {
      await queryClient.cancelQueries({ queryKey: ["timeline"] });
      const previous = queryClient.getQueryData<PostResponse[]>(["timeline"]);
      queryClient.setQueryData<PostResponse[]>(["timeline"], (old) =>
        old?.map((p) =>
          p.id === post.id
            ? {
                ...p,
                liked_by_me: !p.liked_by_me,
                likes_count: p.liked_by_me
                  ? p.likes_count - 1
                  : p.likes_count + 1,
              }
            : p
        )
      );
      return { previous };
    },
    onError: (_err, _post, context) => {
      if (context?.previous) {
        queryClient.setQueryData(["timeline"], context.previous);
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["timeline"] });
      queryClient.invalidateQueries({ queryKey: ["userPosts"] });
      queryClient.invalidateQueries({ queryKey: ["post"] });
    },
  });
}
