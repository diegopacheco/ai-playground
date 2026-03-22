import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:10531/v1",
  apiKey: "not-needed",
});

async function getFirstModel(): Promise<string> {
  const models = await client.models.list();
  const ids = models.data.map(m => m.id);
  console.log("=== Available Models ===");
  for (const id of ids) {
    console.log(`  - ${id}`);
  }
  console.log("");
  return ids[0];
}

async function chatCompletion(model: string) {
  console.log(`=== Chat Completion (${model}) ===`);
  const response = await client.chat.completions.create({
    model,
    messages: [
      { role: "user", content: "What is 2 + 2? Answer in one sentence." }
    ],
  });
  console.log("Response:", response.choices[0]?.message?.content);
  console.log("");
}

async function streamingChat(model: string) {
  console.log(`=== Streaming Chat (${model}) ===`);
  const stream = await client.chat.completions.create({
    model,
    messages: [
      { role: "user", content: "Tell me a short joke about programming." }
    ],
    stream: true,
  });

  process.stdout.write("Response: ");
  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) {
      process.stdout.write(content);
    }
  }
  console.log("\n");
}

async function main() {
  console.log("OpenAI OAuth Proxy PoC\n");
  const model = await getFirstModel();
  console.log(`Using model: ${model}\n`);
  await chatCompletion(model);
  await streamingChat(model);
  console.log("Done!");
}

main().catch(console.error);
