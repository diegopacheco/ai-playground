import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getLeaderboard,
  getLeaderboardWeek,
  trackUser,
  untrackUser,
} from "../api/client";

export function useLeaderboard() {
  return useQuery({
    queryKey: ["leaderboard"],
    queryFn: getLeaderboard,
  });
}

export function useLeaderboardWeek(weekStart: string | null) {
  return useQuery({
    queryKey: ["leaderboardWeek", weekStart],
    queryFn: () => getLeaderboardWeek(weekStart!),
    enabled: !!weekStart,
  });
}

export function useTrackUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (githubUser: string) => trackUser(githubUser),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["leaderboard"] });
    },
  });
}

export function useUntrackUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (username: string) => untrackUser(username),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["leaderboard"] });
    },
  });
}
