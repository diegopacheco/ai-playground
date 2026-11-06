const numberPair = {
  type: "object",
  properties: { a: { type: "number" }, b: { type: "number" } },
  required: ["a", "b"],
};

function requireNumbers({ a, b }) {
  if (typeof a !== "number" || typeof b !== "number") throw new Error("a and b must be numbers");
  return [a, b];
}

export const tools = [
  {
    name: "add",
    description: "Adds two numbers",
    inputSchema: numberPair,
    handler: (args) => {
      const [a, b] = requireNumbers(args);
      return a + b;
    },
  },
  {
    name: "multiply",
    description: "Multiplies two numbers",
    inputSchema: numberPair,
    handler: (args) => {
      const [a, b] = requireNumbers(args);
      return a * b;
    },
  },
];
