import { useQuery } from "@tanstack/react-query";
import { peopleApi } from "./api";
import type { User } from "./types";

export const peopleKeys = {
  users: ["users"] as const,
  policies: ["policies"] as const,
};

export function useUsers() {
  return useQuery({ queryKey: peopleKeys.users, queryFn: peopleApi.users, staleTime: Infinity });
}

export function usePolicies() {
  return useQuery({ queryKey: peopleKeys.policies, queryFn: peopleApi.policies, staleTime: Infinity });
}

export function useUserMap(): ReadonlyMap<string, User> {
  const { data } = useUsers();
  return new Map((data ?? []).map((u) => [u.id, u]));
}
