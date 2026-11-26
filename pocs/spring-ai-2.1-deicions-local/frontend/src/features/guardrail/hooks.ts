import { useMutation } from "@tanstack/react-query";
import { guardrailApi } from "./api";
import type { Battery } from "./types";

export function useScreen() {
  return useMutation({
    mutationFn: (input: { readonly battery: Battery; readonly text: string }) => guardrailApi.screen(input.battery, input.text),
  });
}
