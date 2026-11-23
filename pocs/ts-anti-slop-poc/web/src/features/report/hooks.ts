import { useQuery } from "@tanstack/react-query";
import { reportApi } from "./api.ts";

export const reportKeys = {
  all: ["report"] as const,
};

export function useReport() {
  return useQuery({ queryKey: reportKeys.all, queryFn: reportApi.get });
}
