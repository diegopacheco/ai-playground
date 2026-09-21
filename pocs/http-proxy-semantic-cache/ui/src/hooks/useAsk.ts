import { useActionState } from "react";
import { api } from "../api/client";
import type { AskResult } from "../types";

interface AskState {
  history: AskResult[];
  error: string | null;
}

const initial: AskState = { history: [], error: null };

export function useAsk(onAnswered: () => void) {
  const [state, submit, pending] = useActionState(async (previous: AskState, form: FormData): Promise<AskState> => {
    const question = String(form.get("question") ?? "").trim();
    if (!question) {
      return { ...previous, error: "Type a question first" };
    }
    try {
      const result = await api.ask(question);
      onAnswered();
      return { history: [result, ...previous.history], error: null };
    } catch (error) {
      return { ...previous, error: (error as Error).message };
    }
  }, initial);

  return { ...state, submit, pending };
}
