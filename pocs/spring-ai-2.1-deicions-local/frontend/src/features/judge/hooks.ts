import { useMutation } from "@tanstack/react-query";
import { judgeApi } from "./api";

export function useJudge() {
  return useMutation({
    mutationFn: (input: { readonly question: string; readonly answer: string }) => judgeApi.judge(input.question, input.answer),
  });
}
