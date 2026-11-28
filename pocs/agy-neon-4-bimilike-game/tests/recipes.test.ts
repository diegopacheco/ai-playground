import { describe, expect, test } from "bun:test";
import { cookRecipe, INGREDIENTS } from "../src/features/kitchen/recipes";

describe("kitchen recipe book", () => {
  test("creates specific dishes from kid-friendly ingredient combinations", () => {
    expect(cookRecipe(["berry", "milk"]).id).toBe("smoothie");
    expect(cookRecipe(["cheese", "egg"]).id).toBe("pizza");
    expect(cookRecipe(["carrot", "milk"]).id).toBe("soup");
    expect(cookRecipe(["honey", "egg"]).id).toBe("honeyCake");
    expect(cookRecipe(["cocoa", "milk"]).id).toBe("cookie");
    expect(cookRecipe(["banana", "milk"]).id).toBe("pancakes");
    expect(cookRecipe(["berry"]).id).toBe("cupcake");
  });

  test("always produces a delicious dish for every single ingredient", () => {
    for (const ing of INGREDIENTS) {
      const dish = cookRecipe([ing.id]);
      expect(dish.name.length).toBeGreaterThan(0);
      expect(dish.emoji.length).toBeGreaterThan(0);
    }
  });
});
