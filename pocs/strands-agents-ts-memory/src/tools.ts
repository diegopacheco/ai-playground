import { tool } from "@strands-agents/sdk";
import { z } from "zod";

interface Memory {
  id: string;
  content: string;
  timestamp: number;
}

const memoryStore: Map<string, Memory[]> = new Map();

function getUserMemories(userId: string): Memory[] {
  if (!memoryStore.has(userId)) {
    memoryStore.set(userId, []);
  }
  return memoryStore.get(userId)!;
}

export const storeMemory = tool({
  name: "store_memory",
  description: "Store a piece of information in memory for a user. Use this when the user asks you to remember something.",
  inputSchema: z.object({
    userId: z.string().describe("The user ID to store the memory for"),
    content: z.string().describe("The information to remember"),
  }),
  callback: (input) => {
    const memories = getUserMemories(input.userId);
    const memory: Memory = {
      id: `mem_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      content: input.content,
      timestamp: Date.now(),
    };
    memories.push(memory);
    return `Stored memory: "${input.content}"`;
  },
});

export const retrieveMemories = tool({
  name: "retrieve_memories",
  description: "Search memories for a user by keyword matching. Use this to find relevant memories before answering questions.",
  inputSchema: z.object({
    userId: z.string().describe("The user ID to search memories for"),
    query: z.string().describe("The search query to match against stored memories"),
  }),
  callback: (input) => {
    const memories = getUserMemories(input.userId);
    if (memories.length === 0) {
      return "No memories found for this user.";
    }
    const queryWords = input.query.toLowerCase().split(/\s+/);
    const scored = memories.map((m) => {
      const contentLower = m.content.toLowerCase();
      const matches = queryWords.filter((w) => contentLower.includes(w)).length;
      return { memory: m, score: matches / queryWords.length };
    });
    const relevant = scored
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
    if (relevant.length === 0) {
      return "No relevant memories found for this query.";
    }
    const results = relevant.map((r) => `- ${r.memory.content}`).join("\n");
    return `Found ${relevant.length} relevant memories:\n${results}`;
  },
});

export const listMemories = tool({
  name: "list_memories",
  description: "List all stored memories for a user.",
  inputSchema: z.object({
    userId: z.string().describe("The user ID to list memories for"),
  }),
  callback: (input) => {
    const memories = getUserMemories(input.userId);
    if (memories.length === 0) {
      return "No memories stored for this user.";
    }
    const list = memories.map((m) => `- ${m.content}`).join("\n");
    return `All memories (${memories.length}):\n${list}`;
  },
});

export const deleteMemory = tool({
  name: "delete_memory",
  description: "Delete all memories for a user.",
  inputSchema: z.object({
    userId: z.string().describe("The user ID to delete memories for"),
  }),
  callback: (input) => {
    memoryStore.delete(input.userId);
    return "All memories deleted for this user.";
  },
});
