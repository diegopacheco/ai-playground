import { createTool } from "@mastra/core/tools";
import { z } from "zod";

export const fetchIngredients = createTool({
  id: "fetch-ingredients",
  description: "Fetches a list of available ingredients from the pantry",
  inputSchema: z.object({
    category: z
      .string()
      .optional()
      .describe("Category of ingredients to fetch (e.g., vegetables, proteins, spices)"),
  }),
  outputSchema: z.object({
    ingredients: z.array(z.string()),
  }),
  execute: async (inputData) => {
    const pantry: Record<string, string[]> = {
      vegetables: ["tomato", "onion", "garlic", "bell pepper", "carrot", "spinach", "potato"],
      proteins: ["chicken breast", "ground beef", "eggs", "tofu", "salmon", "shrimp"],
      spices: ["salt", "pepper", "cumin", "paprika", "oregano", "basil", "chili flakes"],
      dairy: ["butter", "milk", "cheddar cheese", "cream", "parmesan"],
      pantry: ["olive oil", "rice", "pasta", "flour", "canned tomatoes", "soy sauce", "lemon"],
    };

    const category = inputData.category;
    if (category && pantry[category.toLowerCase()]) {
      return { ingredients: pantry[category.toLowerCase()] };
    }

    const all = Object.values(pantry).flat();
    return { ingredients: all };
  },
});

export const printRecipe = createTool({
  id: "print-recipe",
  description: "Formats and prints a recipe with title, ingredients, and steps",
  inputSchema: z.object({
    title: z.string().describe("Name of the recipe"),
    ingredients: z.array(z.string()).describe("List of ingredients with quantities"),
    steps: z.array(z.string()).describe("Step-by-step cooking instructions"),
    cookingTime: z.string().describe("Total cooking time"),
    servings: z.number().describe("Number of servings"),
  }),
  outputSchema: z.object({
    formatted: z.string(),
  }),
  execute: async (inputData) => {
    const lines: string[] = [];
    lines.push(`=== ${inputData.title.toUpperCase()} ===`);
    lines.push(`Cooking Time: ${inputData.cookingTime} | Servings: ${inputData.servings}`);
    lines.push("");
    lines.push("INGREDIENTS:");
    inputData.ingredients.forEach((ing: string) => lines.push(`  - ${ing}`));
    lines.push("");
    lines.push("STEPS:");
    inputData.steps.forEach((step: string, i: number) => lines.push(`  ${i + 1}. ${step}`));
    lines.push("===========================");

    const formatted = lines.join("\n");
    console.log(formatted);
    return { formatted };
  },
});
