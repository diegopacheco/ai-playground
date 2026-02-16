import { Agent } from "@strands-agents/sdk";
import { OpenAIModel } from "@strands-agents/sdk/openai";
import { storeMemory, retrieveMemories, listMemories, deleteMemory } from "./tools.js";

const MEMORY_SYSTEM_PROMPT = `You are a Memory Assistant. You can store, retrieve, and manage
information for users across conversations.

When a user says "remember that..." or "note that..." or "I want you to know...",
use the store_memory tool to save that information.

When a user asks a question, first use retrieve_memories to check if you have
relevant stored information, then answer using those memories.

When a user asks to see all memories or says "show memories", use list_memories.

When a user asks to forget everything or clear memories, use delete_memory.

Always use userId "user_1" for all memory operations.
Be helpful and reference stored memories when answering questions.`;

async function main() {
  const model = new OpenAIModel({
    apiKey: process.env.OPENAI_API_KEY!,
    modelId: "gpt-4o",
  });

  const agent = new Agent({
    model,
    systemPrompt: MEMORY_SYSTEM_PROMPT,
    tools: [storeMemory, retrieveMemories, listMemories, deleteMemory],
  });

  const seedMemories = [
    "Remember that my name is Diego",
    "Remember that I live in POA, Brazil",
    "Remember that my favorite programming language is Scala",
    "Remember that I enjoy hiking and trail running",
    "Remember that I prefer window seats on flights",
  ];

  console.log("=== Seeding memories ===\n");
  for (const mem of seedMemories) {
    console.log(`> ${mem}`);
    const result = await agent.invoke(mem);
    console.log(String(result));
    console.log();
  }

  const questions = [
    "What is my name and where do I live?",
    "What are my hobbies?",
    "What programming language do I prefer?",
    "Show me all my memories",
    "What do you know about my travel preferences?",
  ];

  console.log("\n=== Querying memories ===\n");
  for (const question of questions) {
    console.log("=".repeat(70));
    console.log(`QUESTION: ${question}`);
    console.log("=".repeat(70));
    const result = await agent.invoke(question);
    console.log("\nANSWER:");
    console.log(String(result));
    console.log();
  }
}

main().catch(console.error);
