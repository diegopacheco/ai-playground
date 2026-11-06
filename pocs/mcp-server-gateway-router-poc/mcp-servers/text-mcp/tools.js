const textInput = {
  type: "object",
  properties: { text: { type: "string" } },
  required: ["text"],
};

function requireText({ text }) {
  if (typeof text !== "string") throw new Error("text must be a string");
  return text;
}

export const tools = [
  {
    name: "uppercase",
    description: "Converts text to uppercase",
    inputSchema: textInput,
    handler: (args) => requireText(args).toUpperCase(),
  },
  {
    name: "reverse",
    description: "Reverses the characters of the text",
    inputSchema: textInput,
    handler: (args) => [...requireText(args)].reverse().join(""),
  },
  {
    name: "word_count",
    description: "Counts the words in the text",
    inputSchema: textInput,
    handler: (args) => {
      const words = requireText(args).trim().split(/\s+/).filter(Boolean);
      return words.length;
    },
  },
];
