import { Letta } from "@letta-ai/letta-client";

const apiKey = process.env.LETTA_API_KEY;
if (!apiKey) {
  console.error("LETTA_API_KEY env var is not set");
  process.exit(1);
}

const client = new Letta({ apiKey });

const response = await client.agents.messages.create("agent-c52fd0b7-24a7-4c69-b9aa-0ef3457a5b77", {
  input: "What do you remember about me?",
});
console.log(response.messages);

console.log("Retrieving agent to see memory blocks...");
const agent = await client.agents.retrieve("agent-c52fd0b7-24a7-4c69-b9aa-0ef3457a5b77");
console.log(agent.memory.blocks);
