export const INGREDIENT_IDS = [
  "berry",
  "banana",
  "egg",
  "milk",
  "cocoa",
  "cheese",
  "carrot",
  "honey",
] as const;
export type IngredientId = (typeof INGREDIENT_IDS)[number];

export const DISH_IDS = [
  "cupcake",
  "pancakes",
  "smoothie",
  "pizza",
  "soup",
  "honeyCake",
  "cookie",
] as const;
export type DishId = (typeof DISH_IDS)[number];

export type IngredientSpec = {
  readonly id: IngredientId;
  readonly name: string;
  readonly emoji: string;
  readonly color: string;
};

export type DishSpec = {
  readonly id: DishId;
  readonly name: string;
  readonly emoji: string;
  readonly mainColor: string;
  readonly toppingColor: string;
};

export type CookedDish = {
  readonly uid: string;
  readonly dishId: DishId;
  readonly slotIndex: number;
};
