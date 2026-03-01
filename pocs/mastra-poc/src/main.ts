import { mastra } from "./mastra/index";

async function main() {
  const agent = mastra.getAgent("chefAgent");
  const result = await agent.generate(
    "I want to cook something with chicken and vegetables. What do you suggest?"
  );
  console.log(result.text);
}

main();
