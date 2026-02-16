import { Agent } from "@strands-agents/sdk";
import { OpenAIModel } from "@strands-agents/sdk/openai";
import {
  mathAssistant,
  englishAssistant,
  languageAssistant,
  csAssistant,
  generalAssistant,
} from "./agents.js";

const TEACHER_PROMPT = `You are a Teacher's Assistant orchestrator. Your job is to analyze student
questions and route them to the appropriate specialist agent.

Available specialists:
- math_assistant: For math problems, calculations, equations, geometry, calculus
- english_assistant: For grammar, writing, comprehension, literature
- language_assistant: For translation and foreign language learning
- computer_science_assistant: For programming, algorithms, data structures, CS concepts
- general_assistant: For everything else

Analyze each question carefully and delegate to the most appropriate specialist.
Always use exactly one specialist tool per question.
Present the specialist's response clearly to the student.`;

async function main() {
  const model = new OpenAIModel({
    apiKey: process.env.OPENAI_API_KEY!,
    modelId: "gpt-4o",
  });

  const teacherAgent = new Agent({
    model,
    systemPrompt: TEACHER_PROMPT,
    tools: [mathAssistant, englishAssistant, languageAssistant, csAssistant, generalAssistant],
  });

  const questions = [
    "What is the derivative of x^3 + 2x^2 - 5x + 3?",
    "Can you check this sentence for grammar: 'me and him went to the store yesterday'",
    "How do you say 'Good morning, how are you?' in French?",
    "Explain what a binary search tree is and its time complexity",
    "What caused World War I?",
  ];

  for (const question of questions) {
    console.log("\n" + "=".repeat(80));
    console.log(`STUDENT QUESTION: ${question}`);
    console.log("=".repeat(80));
    const result = await teacherAgent.invoke(question);
    console.log("\nFINAL ANSWER:");
    console.log(String(result));
    console.log("\n");
  }
}

main().catch(console.error);
