import { OpenAIChatModel } from "beeai-framework/adapters/openai/backend/chat";
import { UserMessage } from "beeai-framework/backend/message";

const llm = new OpenAIChatModel("gpt-4o-mini", {});

const response = await llm.create({
  messages: [new UserMessage("What are the top 3 programming languages in 2025?")],
});

console.log("Question: What are the top 3 programming languages in 2025?");
console.log("Answer: " + response.getTextContent());
