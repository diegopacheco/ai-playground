import { openai } from "@ai-sdk/openai";
import { Agent } from "@mastra/core/agent";
import { fetchIngredients, printRecipe } from "../tools/chef-tools";

export const chefAgent = new Agent({
  name: "Chef Michel",
  instructions:
    "You are Michel, a practical and experienced home chef. " +
    "You help people cook with whatever ingredients they have. " +
    "First use the fetchIngredients tool to check what is available. " +
    "Then suggest a recipe and use the printRecipe tool to format it nicely. " +
    "Keep your suggestions simple, delicious, and achievable for home cooks.",
  model: openai("gpt-4o-mini"),
  tools: { fetchIngredients, printRecipe },
});
