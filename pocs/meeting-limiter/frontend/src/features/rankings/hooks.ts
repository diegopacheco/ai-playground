import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { rankingsApi } from "./api";
import type { Period } from "./range";

export const rankingKeys = {
  all: ["rankings"] as const,
  period: (p: Period) => [...rankingKeys.all, p.from.toISOString(), p.to.toISOString()] as const,
};

export function useRankings(p: Period) {
  return useQuery({ queryKey: rankingKeys.period(p), queryFn: () => rankingsApi.get(p), placeholderData: keepPreviousData });
}
