import type { DishId, DishSpec, IngredientId, IngredientSpec } from "./types";

export const INGREDIENTS: readonly IngredientSpec[] = [
  { id: "berry", name: "Berry", emoji: "🍓", color: "#ff5d8f" },
  { id: "banana", name: "Banana", emoji: "🍌", color: "#ffe156" },
  { id: "egg", name: "Egg", emoji: "🥚", color: "#fff3d1" },
  { id: "milk", name: "Milk", emoji: "🥛", color: "#eef8ff" },
  { id: "cocoa", name: "Cocoa", emoji: "🍫", color: "#9c6644" },
  { id: "cheese", name: "Cheese", emoji: "🧀", color: "#ffbe3b" },
  { id: "carrot", name: "Carrot", emoji: "🥕", color: "#ff8243" },
  { id: "honey", name: "Honey", emoji: "🍯", color: "#ffb703" },
] as const;

export const DISHES: readonly DishSpec[] = [
  { id: "cupcake", name: "Berry Cupcake", emoji: "🧁", mainColor: "#ffb5c8", toppingColor: "#ff4d79" },
  { id: "pancakes", name: "Golden Pancakes", emoji: "🥞", mainColor: "#f4b961", toppingColor: "#ffbe0b" },
  { id: "smoothie", name: "Berry Smoothie", emoji: "🥤", mainColor: "#ff8fab", toppingColor: "#fff1f5" },
  { id: "pizza", name: "Cheese Pizza", emoji: "🍕", mainColor: "#ffd166", toppingColor: "#ef476f" },
  { id: "soup", name: "Carrot Soup", emoji: "🥣", mainColor: "#ff9f43", toppingColor: "#8ac926" },
  { id: "honeyCake", name: "Honey Cake", emoji: "🍰", mainColor: "#ffe082", toppingColor: "#ff9f1c" },
  { id: "cookie", name: "Choco Cookie", emoji: "🍪", mainColor: "#d49a6a", toppingColor: "#5c3d2e" },
] as const;

export function getDishSpec(id: DishId): DishSpec {
  const found = DISHES.find((d) => d.id === id);
  if (!found) {
    throw new Error(`Unknown dish: ${id}`);
  }
  return found;
}

export function getIngredientSpec(id: IngredientId): IngredientSpec {
  const found = INGREDIENTS.find((i) => i.id === id);
  if (!found) {
    throw new Error(`Unknown ingredient: ${id}`);
  }
  return found;
}

export function cookRecipe(picked: readonly IngredientId[]): DishSpec {
  const has = (id: IngredientId) => picked.includes(id);
  if (has("cheese") && (has("carrot") || has("egg") || has("milk"))) {
    return getDishSpec("pizza");
  }
  if (has("carrot")) {
    return getDishSpec("soup");
  }
  if (has("berry") && has("milk")) {
    return getDishSpec("smoothie");
  }
  if (has("honey") && (has("egg") || has("banana"))) {
    return getDishSpec("honeyCake");
  }
  if (has("cocoa")) {
    return getDishSpec("cookie");
  }
  if (has("banana") || has("egg")) {
    return getDishSpec("pancakes");
  }
  if (has("berry")) {
    return getDishSpec("cupcake");
  }
  if (has("cheese")) {
    return getDishSpec("pizza");
  }
  return getDishSpec("cupcake");
}
