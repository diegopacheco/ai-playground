export type KeyKind = "digit" | "operator" | "paren" | "clear" | "back" | "equals";

export type Key = {
  label: string;
  kind: KeyKind;
  wide?: boolean;
};

export const KEYS: Key[] = [
  { label: "C", kind: "clear" },
  { label: "(", kind: "paren" },
  { label: ")", kind: "paren" },
  { label: "⌫", kind: "back" },
  { label: "7", kind: "digit" },
  { label: "8", kind: "digit" },
  { label: "9", kind: "digit" },
  { label: "÷", kind: "operator" },
  { label: "4", kind: "digit" },
  { label: "5", kind: "digit" },
  { label: "6", kind: "digit" },
  { label: "×", kind: "operator" },
  { label: "1", kind: "digit" },
  { label: "2", kind: "digit" },
  { label: "3", kind: "digit" },
  { label: "−", kind: "operator" },
  { label: "0", kind: "digit" },
  { label: ".", kind: "digit" },
  { label: "^", kind: "operator" },
  { label: "+", kind: "operator" },
  { label: "%", kind: "operator" },
  { label: "=", kind: "equals", wide: true },
];

const OPERATORS = new Set(["+", "−", "×", "÷", "^", "%"]);

function currentNumber(expression: string): string {
  const match = expression.match(/[\d.]*$/);
  return match ? match[0] : "";
}

function pressOperator(expression: string, operator: string): string {
  const last = expression.at(-1);
  if (last === undefined || last === "(") return operator === "−" ? expression + operator : expression;
  if (OPERATORS.has(last)) return expression.slice(0, -1) + operator;
  return expression + operator;
}

function pressDigit(expression: string, digit: string): string {
  if (digit !== ".") return expression + digit;
  const number = currentNumber(expression);
  if (number.includes(".")) return expression;
  return expression + (number === "" ? "0." : ".");
}

export function press(expression: string, key: Key): string {
  switch (key.kind) {
    case "clear":
      return "";
    case "back":
      return expression.slice(0, -1);
    case "digit":
      return pressDigit(expression, key.label);
    case "operator":
      return pressOperator(expression, key.label);
    case "paren":
      return expression + key.label;
    case "equals":
      return expression;
  }
}

export function keyForInput(input: string): Key | undefined {
  const aliases: Record<string, string> = {
    "*": "×",
    "/": "÷",
    "-": "−",
    Enter: "=",
    "=": "=",
    Backspace: "⌫",
    Escape: "C",
  };
  const label = aliases[input] ?? input;
  return KEYS.find((key) => key.label === label);
}
