import { tool } from "@strands-agents/sdk";
import { z } from "zod";

export const calculator = tool({
  name: "calculator",
  description: "Evaluate math expressions. Supports basic arithmetic, powers, roots, etc.",
  inputSchema: z.object({
    expression: z.string().describe("The math expression to evaluate, e.g. '2 + 2', 'Math.sqrt(144)', 'Math.pow(2,10)'"),
  }),
  callback: (input) => {
    try {
      const result = new Function(`return ${input.expression}`)();
      return `Result: ${result}`;
    } catch (e: any) {
      return `Error evaluating expression: ${e.message}`;
    }
  },
});

export const editor = tool({
  name: "editor",
  description: "Check grammar and style of text. Returns suggestions for improvement.",
  inputSchema: z.object({
    text: z.string().describe("The text to check for grammar and style"),
  }),
  callback: (input) => {
    const issues: string[] = [];
    if (input.text.length === 0) {
      issues.push("Text is empty");
    }
    if (input.text[0] !== input.text[0].toUpperCase()) {
      issues.push("Sentence should start with a capital letter");
    }
    if (!/[.!?]$/.test(input.text.trim())) {
      issues.push("Sentence should end with punctuation");
    }
    const doubleSpaces = input.text.match(/  +/g);
    if (doubleSpaces) {
      issues.push(`Found ${doubleSpaces.length} double-space(s)`);
    }
    if (issues.length === 0) {
      return "Text looks good! No grammar or style issues found.";
    }
    return `Issues found:\n${issues.map((i) => `- ${i}`).join("\n")}`;
  },
});

export const jsRepl = tool({
  name: "js_repl",
  description: "Execute JavaScript code and return the result.",
  inputSchema: z.object({
    code: z.string().describe("The JavaScript code to execute"),
  }),
  callback: (input) => {
    try {
      const result = new Function(input.code)();
      return `Output: ${JSON.stringify(result)}`;
    } catch (e: any) {
      return `Error: ${e.message}`;
    }
  },
});

export const translator = tool({
  name: "translator",
  description: "Translate common words/phrases between languages. Supports basic translations.",
  inputSchema: z.object({
    text: z.string().describe("The text to translate"),
    fromLang: z.string().describe("Source language"),
    toLang: z.string().describe("Target language"),
  }),
  callback: (input) => {
    return `Please translate "${input.text}" from ${input.fromLang} to ${input.toLang}. (Use your language knowledge to provide the translation.)`;
  },
});
