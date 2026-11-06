import { createHash } from "node:crypto";

const textInput = { type: "object", properties: { text: { type: "string" } }, required: ["text"] };

function requireText({ text }) {
  if (typeof text !== "string") throw new Error("text must be a string");
  return text;
}

export const tools = [
  {
    name: "sha256",
    description: "Returns the SHA-256 hex digest of a text",
    inputSchema: textInput,
    handler: (args) => createHash("sha256").update(requireText(args)).digest("hex"),
  },
];
