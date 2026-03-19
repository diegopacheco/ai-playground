import { useQuery, useMutation } from "@tanstack/react-query";
import {
  getAuctions,
  getAuction,
  createAuction,
  getAgents,
} from "../api/auctions.ts";
import type { Agent } from "../types/index.ts";

export function useAuctions() {
  return useQuery({
    queryKey: ["auctions"],
    queryFn: getAuctions,
  });
}

export function useAuction(id: string) {
  return useQuery({
    queryKey: ["auction", id],
    queryFn: () => getAuction(id),
    enabled: !!id,
  });
}

export function useCreateAuction() {
  return useMutation({
    mutationFn: (agents: Agent[]) => createAuction(agents),
  });
}

export function useAgents() {
  return useQuery({
    queryKey: ["agents"],
    queryFn: getAgents,
  });
}
