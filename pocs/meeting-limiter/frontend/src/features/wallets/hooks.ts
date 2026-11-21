import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { walletsApi } from "./api";

export const walletKeys = {
  all: ["wallets"] as const,
  week: (weekStart: Date) => [...walletKeys.all, weekStart.toISOString()] as const,
};

export function useWallets(weekStart: Date) {
  return useQuery({
    queryKey: walletKeys.week(weekStart),
    queryFn: () => walletsApi.week(weekStart),
    placeholderData: keepPreviousData,
  });
}
