import { useMutation } from "@tanstack/react-query";
import { triageApi } from "./api";

export function useTriage() {
  return useMutation({ mutationFn: (message: string) => triageApi.decide(message) });
}
