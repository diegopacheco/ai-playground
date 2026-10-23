import { useCallback, useState } from "react";
import { useApp, type App } from "@modelcontextprotocol/ext-apps/react";
import { toOutcome, type Calculation, type ToolResult } from "../shared/calculation";

const APP_INFO = { name: "calculator-view", version: "0.1.0" };

export type CalculatorState = {
  expression: string;
  setExpression: (expression: string) => void;
  current: Calculation | null;
  error: string | null;
  history: Calculation[];
  busy: boolean;
  connected: boolean;
  evaluate: () => Promise<void>;
};

export function useCalculator(): CalculatorState {
  const [expression, setExpression] = useState("");
  const [current, setCurrent] = useState<Calculation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<Calculation[]>([]);
  const [busy, setBusy] = useState(false);

  const show = useCallback((result: ToolResult) => {
    const outcome = toOutcome(result);
    if (outcome.kind === "error") {
      setError(outcome.message);
      return null;
    }
    setError(null);
    setCurrent(outcome.calculation);
    setExpression(outcome.calculation.result);
    setHistory((previous) => [outcome.calculation, ...previous].slice(0, 8));
    return outcome.calculation;
  }, []);

  const onAppCreated = useCallback(
    (created: App) => {
      created.ontoolinput = ({ arguments: input }) => {
        const value = input?.expression;
        if (typeof value === "string") setExpression(value);
      };
      created.ontoolresult = (result) => {
        show(result);
      };
    },
    [show],
  );

  const { app, isConnected } = useApp({ appInfo: APP_INFO, capabilities: {}, onAppCreated });

  const evaluate = useCallback(async () => {
    if (!app || expression.trim() === "") return;
    setBusy(true);
    try {
      const calculation = show(await app.callServerTool({ name: "calculate", arguments: { expression } }));
      if (calculation && app.getHostCapabilities()?.updateModelContext) {
        await app.updateModelContext({
          content: [{ type: "text", text: `The user calculated ${calculation.expression} = ${calculation.result}` }],
        });
      }
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }, [app, expression, show]);

  return { expression, setExpression, current, error, history, busy, connected: isConnected, evaluate };
}
