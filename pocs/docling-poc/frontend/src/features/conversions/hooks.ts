import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { logger } from "../../logger";
import { conversionsApi, type UploadInput } from "./api";

export const conversionKeys = {
  all: ["conversions"] as const,
  history: () => [...conversionKeys.all, "history"] as const,
  detail: (id: string) => [...conversionKeys.all, "detail", id] as const,
  formats: ["formats"] as const,
};

export function useHistory() {
  return useQuery({ queryKey: conversionKeys.history(), queryFn: conversionsApi.history });
}

export function useConversion(id: string) {
  return useQuery({ queryKey: conversionKeys.detail(id), queryFn: () => conversionsApi.get(id) });
}

export function useFormats() {
  return useQuery({ queryKey: conversionKeys.formats, queryFn: conversionsApi.formats, staleTime: Infinity });
}

export function useUpload() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: UploadInput) => conversionsApi.upload(input),
    onSuccess: (conversion) => {
      logger.info("document converted", { id: conversion.id, ms: conversion.durationMs });
      queryClient.setQueryData(conversionKeys.detail(conversion.id), conversion);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: conversionKeys.history() }),
  });
}
