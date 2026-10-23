export type Calculation = {
  expression: string;
  result: string;
  steps: string[];
};

export type ToolResult = {
  isError?: boolean;
  content?: ReadonlyArray<{ type: string; text?: string }>;
  structuredContent?: unknown;
};

export type Outcome =
  | { kind: "ok"; calculation: Calculation }
  | { kind: "error"; message: string };

const OPERATOR_SYMBOLS: Record<string, string> = { "*": "×", "/": "÷", "-": "−" };

function isCalculation(value: unknown): value is Calculation {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.expression === "string" &&
    typeof candidate.result === "string" &&
    Array.isArray(candidate.steps) &&
    candidate.steps.every((step) => typeof step === "string")
  );
}

function contentText(result: ToolResult): string {
  return (result.content ?? [])
    .flatMap((block) => (block.type === "text" && block.text ? [block.text] : []))
    .join(" ")
    .trim();
}

export function toOutcome(result: ToolResult): Outcome {
  if (!result.isError && isCalculation(result.structuredContent)) {
    return { kind: "ok", calculation: result.structuredContent };
  }
  return { kind: "error", message: contentText(result) || "calculation failed" };
}

export function prettyStep(step: string): string {
  return step.replace(/ ([*/-]) /g, (_, operator: string) => ` ${OPERATOR_SYMBOLS[operator]} `);
}
