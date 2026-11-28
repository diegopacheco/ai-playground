import type { ActivityStats, CharacterId, CharacterPlacement } from "../characters/types";
import type { CookedDish, DishId } from "../kitchen/types";

export const ROOM_IDS = ["all", "kitchen", "living", "bedroom"] as const;
export type RoomFocus = (typeof ROOM_IDS)[number];

export const VIEW_MODES = ["3d", "2d"] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

export const SURFACE_IDS = [
  "kitchenWall",
  "livingWall",
  "bedroomWall",
  "floor",
  "roof",
] as const;
export type SurfaceId = (typeof SURFACE_IDS)[number];

export const FURNITURE_KINDS = [
  "fridge",
  "stove",
  "table",
  "sofa",
  "rug",
  "phoneTable",
  "lamp",
  "bed",
  "toyChest",
] as const;
export type FurnitureKind = (typeof FURNITURE_KINDS)[number];

export type FurnitureItem = {
  readonly id: string;
  readonly kind: FurnitureKind;
  readonly label: string;
  readonly x: number;
  readonly z: number;
  readonly color: string;
  readonly accentColor: string;
  readonly styleVariant: 0 | 1 | 2;
  readonly active: boolean;
};

export type SelectionTarget =
  | { readonly kind: "furniture"; readonly id: string }
  | { readonly kind: "surface"; readonly id: SurfaceId }
  | { readonly kind: "character"; readonly id: CharacterId };

export type HouseState = {
  readonly surfaces: Readonly<Record<SurfaceId, string>>;
  readonly furniture: readonly FurnitureItem[];
  readonly unlockedCharacters: readonly CharacterId[];
  readonly characters: readonly CharacterPlacement[];
  readonly dishes: readonly CookedDish[];
  readonly stats: ActivityStats;
};

export type HouseAction =
  | { readonly type: "moveFurniture"; readonly id: string; readonly x: number; readonly z: number }
  | { readonly type: "moveCharacter"; readonly id: CharacterId; readonly x: number; readonly z: number }
  | { readonly type: "paintFurniture"; readonly id: string; readonly color: string }
  | { readonly type: "cycleFurnitureStyle"; readonly id: string }
  | { readonly type: "toggleFurnitureActive"; readonly id: string }
  | { readonly type: "paintSurface"; readonly id: SurfaceId; readonly color: string }
  | { readonly type: "cookDish"; readonly dishId: DishId; readonly uid: string }
  | { readonly type: "feedCharacter"; readonly dishUid: string; readonly characterId: CharacterId; readonly now: number }
  | { readonly type: "recordCall"; readonly characterId: CharacterId; readonly mood: CharacterPlacement["mood"]; readonly now: number }
  | { readonly type: "reset" };

export type HouseUpdateResult = {
  readonly state: HouseState;
  readonly newlyUnlocked: readonly CharacterId[];
};
