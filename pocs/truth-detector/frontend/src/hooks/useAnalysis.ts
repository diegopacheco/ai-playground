import { useMutation, useQuery } from "@tanstack/react-query";
import {
  analyzeUser,
  getAgents,
  getAnalysis,
  getLatestAnalysis,
  getWeeklyScores,
} from "../api/client";

export function useAgents() {
  return useQuery({
    queryKey: ["agents"],
    queryFn: () => getAgents(),
  });
}

export function useAnalyze() {
  return useMutation({
    mutationFn: (params: { githubUser: string; cli: string; model: string }) =>
      analyzeUser(params.githubUser, params.cli, params.model),
  });
}

export function useAnalysisQuery(id: string | null) {
  return useQuery({
    queryKey: ["analysis", id],
    queryFn: () => getAnalysis(id!),
    enabled: !!id,
  });
}

export function useLatestAnalysis(username: string | null) {
  return useQuery({
    queryKey: ["latestAnalysis", username],
    queryFn: () => getLatestAnalysis(username!),
    enabled: !!username,
  });
}

export function useWeeklyScores(username: string | null) {
  return useQuery({
    queryKey: ["weeklyScores", username],
    queryFn: () => getWeeklyScores(username!),
    enabled: !!username,
  });
}
