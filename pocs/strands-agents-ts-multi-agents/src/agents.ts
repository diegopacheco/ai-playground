import { Agent, tool } from "@strands-agents/sdk";
import { OpenAIModel } from "@strands-agents/sdk/openai";
import { z } from "zod";
import { calculator, editor, jsRepl, translator } from "./tools.js";

function createModel(): OpenAIModel {
  return new OpenAIModel({
    apiKey: process.env.OPENAI_API_KEY!,
    modelId: "gpt-4o",
  });
}

const MATH_PROMPT = `You are a Math Assistant. You help students with math problems.
You can solve arithmetic, algebra, calculus, geometry, and other math topics.
Use the calculator tool for computations when needed.
Always show your work step by step.`;

const ENGLISH_PROMPT = `You are an English Assistant. You help students with English grammar,
writing, comprehension, and literature analysis.
Use the editor tool to check grammar and style when needed.
Provide clear explanations and suggestions.`;

const LANGUAGE_PROMPT = `You are a Language Assistant. You help students with language translation
and learning foreign languages.
Use the translator tool when needed.
Provide cultural context when relevant.`;

const CS_PROMPT = `You are a Computer Science Assistant. You help students with programming,
algorithms, data structures, and computer science concepts.
Use the js_repl tool to execute and demonstrate code when needed.
Explain concepts clearly with code snippets.`;

const GENERAL_PROMPT = `You are a General Knowledge Assistant. You help students with questions
that don't fall into math, english, language, or computer science categories.
Provide helpful, accurate, and educational responses.`;

export const mathAssistant = tool({
  name: "math_assistant",
  description: "Handles math-related queries: arithmetic, algebra, calculus, geometry, equations.",
  inputSchema: z.object({
    query: z.string().describe("The math question or problem to solve"),
  }),
  callback: async (input) => {
    const agent = new Agent({
      model: createModel(),
      systemPrompt: MATH_PROMPT,
      tools: [calculator],
      printer: false,
    });
    const result = await agent.invoke(input.query);
    return String(result);
  },
});

export const englishAssistant = tool({
  name: "english_assistant",
  description: "Handles English-related queries: grammar, writing, comprehension, literature.",
  inputSchema: z.object({
    query: z.string().describe("The English question or writing to review"),
  }),
  callback: async (input) => {
    const agent = new Agent({
      model: createModel(),
      systemPrompt: ENGLISH_PROMPT,
      tools: [editor],
      printer: false,
    });
    const result = await agent.invoke(input.query);
    return String(result);
  },
});

export const languageAssistant = tool({
  name: "language_assistant",
  description: "Handles language translation and foreign language learning queries.",
  inputSchema: z.object({
    query: z.string().describe("The translation or language learning question"),
  }),
  callback: async (input) => {
    const agent = new Agent({
      model: createModel(),
      systemPrompt: LANGUAGE_PROMPT,
      tools: [translator],
      printer: false,
    });
    const result = await agent.invoke(input.query);
    return String(result);
  },
});

export const csAssistant = tool({
  name: "computer_science_assistant",
  description: "Handles programming and computer science queries: algorithms, data structures, coding.",
  inputSchema: z.object({
    query: z.string().describe("The CS or programming question"),
  }),
  callback: async (input) => {
    const agent = new Agent({
      model: createModel(),
      systemPrompt: CS_PROMPT,
      tools: [jsRepl],
      printer: false,
    });
    const result = await agent.invoke(input.query);
    return String(result);
  },
});

export const generalAssistant = tool({
  name: "general_assistant",
  description: "Handles general knowledge queries that don't fit other categories.",
  inputSchema: z.object({
    query: z.string().describe("The general knowledge question"),
  }),
  callback: async (input) => {
    const agent = new Agent({
      model: createModel(),
      systemPrompt: GENERAL_PROMPT,
      tools: [],
      printer: false,
    });
    const result = await agent.invoke(input.query);
    return String(result);
  },
});
