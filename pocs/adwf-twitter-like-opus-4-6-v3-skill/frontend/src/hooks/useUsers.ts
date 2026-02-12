import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";

export function useUserProfile(userId: string) {
  return useQuery({
    queryKey: ["profile", userId],
    queryFn: () => api.getUserProfile(userId),
  });
}

export function useFollowers(userId: string) {
  return useQuery({
    queryKey: ["followers", userId],
    queryFn: () => api.getFollowers(userId),
  });
}

export function useFollowing(userId: string) {
  return useQuery({
    queryKey: ["following", userId],
    queryFn: () => api.getFollowing(userId),
  });
}

export function useToggleFollow() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      userId,
      isFollowing,
    }: {
      userId: string;
      isFollowing: boolean;
    }) => (isFollowing ? api.unfollowUser(userId) : api.followUser(userId)),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["profile", variables.userId],
      });
      queryClient.invalidateQueries({ queryKey: ["timeline"] });
      queryClient.invalidateQueries({ queryKey: ["followers"] });
      queryClient.invalidateQueries({ queryKey: ["following"] });
    },
  });
}
