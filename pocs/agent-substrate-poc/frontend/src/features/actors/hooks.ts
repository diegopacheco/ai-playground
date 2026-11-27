import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { config } from "../../config";
import { logger } from "../../logger";
import { actorsApi } from "./api";
import type { Action } from "./rules";

export const actorKeys = {
  all: ["actors"] as const,
  list: () => [...actorKeys.all, "list"] as const,
  detail: (name: string) => [...actorKeys.all, "detail", name] as const,
  state: (name: string) => [...actorKeys.all, "state", name] as const,
  workers: ["workers"] as const,
};

export function useActors() {
  return useQuery({ queryKey: actorKeys.list(), queryFn: actorsApi.list, refetchInterval: config.refreshMs });
}

export function useActor(name: string) {
  return useQuery({ queryKey: actorKeys.detail(name), queryFn: () => actorsApi.get(name), refetchInterval: config.refreshMs });
}

export function useAgentState(name: string, enabled: boolean) {
  return useQuery({ queryKey: actorKeys.state(name), queryFn: () => actorsApi.state(name), enabled, retry: 0 });
}

export function useWorkers() {
  return useQuery({ queryKey: actorKeys.workers, queryFn: actorsApi.workers, refetchInterval: config.refreshMs });
}

export function useCreateActor() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => actorsApi.create(name),
    onSuccess: (actor) => logger.info("actor created", { name: actor.name }),
    onSettled: () => queryClient.invalidateQueries({ queryKey: actorKeys.all }),
  });
}

export function useActorAction() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ name, action }: { readonly name: string; readonly action: Action }) => actorsApi.act(name, action),
    onSuccess: (_, { name, action }) => logger.info("actor action", { name, action }),
    onSettled: () => queryClient.invalidateQueries({ queryKey: actorKeys.all }),
  });
}

export function useRemember(name: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (text: string) => actorsApi.remember(name, text),
    onSettled: () => queryClient.invalidateQueries({ queryKey: actorKeys.all }),
  });
}
