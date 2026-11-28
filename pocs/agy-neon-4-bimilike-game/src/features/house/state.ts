import { config } from "../../lib/config";
import type { StorageAdapter } from "../../lib/storage";
import { CHARACTERS, getCharacterSpec } from "../characters/catalog";
import { CHARACTER_IDS, type CharacterId, type CharacterPlacement } from "../characters/types";
import { evaluateUnlocks } from "../characters/unlocks";
import { DISH_IDS, type CookedDish, type DishId } from "../kitchen/types";
import { clampHousePosition, DEFAULT_FURNITURE, DEFAULT_SURFACES } from "./catalog";
import {
  FURNITURE_KINDS,
  SURFACE_IDS,
  type FurnitureItem,
  type FurnitureKind,
  type HouseAction,
  type HouseState,
  type HouseUpdateResult,
  type SurfaceId,
} from "./types";

const MAX_DISHES = 5;

export function createInitialHouseState(): HouseState {
  const starters = CHARACTERS.filter((c) => c.unlockRule.kind === "starter");
  return {
    surfaces: { ...DEFAULT_SURFACES },
    furniture: DEFAULT_FURNITURE.map((f) => ({ ...f })),
    unlockedCharacters: starters.map((c) => c.id),
    characters: starters.map((c) => ({
      id: c.id,
      x: c.homeX,
      z: c.homeZ,
      mood: "happy",
      moodUntil: 0,
    })),
    dishes: [{ uid: "starter-cupcake", dishId: "cupcake", slotIndex: 0 }],
    stats: {
      cookedCount: 0,
      paintedCount: 0,
      movedCount: 0,
      calledCount: 0,
      fedCount: 0,
    },
  };
}

function syncUnlockedPlacements(
  current: readonly CharacterPlacement[],
  unlocked: readonly CharacterId[],
): readonly CharacterPlacement[] {
  const map = new Map<CharacterId, CharacterPlacement>();
  for (const placement of current) {
    map.set(placement.id, placement);
  }
  return unlocked.map((id) => {
    const existing = map.get(id);
    if (existing) return existing;
    const spec = getCharacterSpec(id);
    return {
      id,
      x: spec.homeX,
      z: spec.homeZ,
      mood: "dancing",
      moodUntil: Date.now() + 4000,
    };
  });
}

function finalizeState(next: HouseState): HouseUpdateResult {
  const { unlocked, newlyUnlocked } = evaluateUnlocks(next.stats, next.unlockedCharacters);
  if (newlyUnlocked.length === 0) {
    return { state: next, newlyUnlocked: [] };
  }
  return {
    state: {
      ...next,
      unlockedCharacters: unlocked,
      characters: syncUnlockedPlacements(next.characters, unlocked),
    },
    newlyUnlocked,
  };
}

export function reduceHouseState(state: HouseState, action: HouseAction): HouseUpdateResult {
  switch (action.type) {
    case "moveFurniture": {
      const clamped = clampHousePosition(action.x, action.z);
      let changed = false;
      const furniture = state.furniture.map((item) => {
        if (item.id !== action.id) return item;
        const dist = Math.hypot(item.x - clamped.x, item.z - clamped.z);
        if (dist > 0.15) {
          changed = true;
        }
        return { ...item, x: clamped.x, z: clamped.z };
      });
      return finalizeState({
        ...state,
        furniture,
        stats: changed
          ? { ...state.stats, movedCount: state.stats.movedCount + 1 }
          : state.stats,
      });
    }
    case "moveCharacter": {
      const clamped = clampHousePosition(action.x, action.z);
      const characters = state.characters.map((c) =>
        c.id === action.id ? { ...c, x: clamped.x, z: clamped.z } : c,
      );
      return { state: { ...state, characters }, newlyUnlocked: [] };
    }
    case "paintFurniture": {
      const furniture = state.furniture.map((item) =>
        item.id === action.id ? { ...item, color: action.color } : item,
      );
      return finalizeState({
        ...state,
        furniture,
        stats: { ...state.stats, paintedCount: state.stats.paintedCount + 1 },
      });
    }
    case "cycleFurnitureStyle": {
      const furniture = state.furniture.map((item) => {
        if (item.id !== action.id) return item;
        const nextVariant: 0 | 1 | 2 = item.styleVariant === 0 ? 1 : item.styleVariant === 1 ? 2 : 0;
        return { ...item, styleVariant: nextVariant };
      });
      return finalizeState({
        ...state,
        furniture,
        stats: { ...state.stats, paintedCount: state.stats.paintedCount + 1 },
      });
    }
    case "toggleFurnitureActive": {
      const furniture = state.furniture.map((item) =>
        item.id === action.id ? { ...item, active: !item.active } : item,
      );
      return { state: { ...state, furniture }, newlyUnlocked: [] };
    }
    case "paintSurface": {
      return finalizeState({
        ...state,
        surfaces: { ...state.surfaces, [action.id]: action.color },
        stats: { ...state.stats, paintedCount: state.stats.paintedCount + 1 },
      });
    }
    case "cookDish": {
      const usedSlots = new Set(state.dishes.map((d) => d.slotIndex));
      let nextSlot = 0;
      for (let i = 0; i < MAX_DISHES; i += 1) {
        if (!usedSlots.has(i)) {
          nextSlot = i;
          break;
        }
      }
      const trimmed = state.dishes.length >= MAX_DISHES ? state.dishes.slice(1) : state.dishes;
      const newDish: CookedDish = {
        uid: action.uid,
        dishId: action.dishId,
        slotIndex: nextSlot,
      };
      return finalizeState({
        ...state,
        dishes: [...trimmed, newDish],
        stats: { ...state.stats, cookedCount: state.stats.cookedCount + 1 },
      });
    }
    case "feedCharacter": {
      const exists = state.dishes.some((d) => d.uid === action.dishUid);
      if (!exists) return { state, newlyUnlocked: [] };
      const dishes = state.dishes.filter((d) => d.uid !== action.dishUid);
      const characters = state.characters.map((c) =>
        c.id === action.characterId
          ? { ...c, mood: "yummy" as const, moodUntil: action.now + 3200 }
          : c,
      );
      return finalizeState({
        ...state,
        dishes,
        characters,
        stats: { ...state.stats, fedCount: state.stats.fedCount + 1 },
      });
    }
    case "recordCall": {
      const characters = state.characters.map((c) =>
        c.id === action.characterId
          ? { ...c, mood: action.mood, moodUntil: action.now + 3500 }
          : c,
      );
      return finalizeState({
        ...state,
        characters,
        stats: { ...state.stats, calledCount: state.stats.calledCount + 1 },
      });
    }
    case "reset":
      return { state: createInitialHouseState(), newlyUnlocked: [] };
    default: {
      const unreachable: never = action;
      return unreachable;
    }
  }
}

function isRecord(val: unknown): val is Record<string, unknown> {
  return typeof val === "object" && val !== null && !Array.isArray(val);
}

function isCharacterId(val: unknown): val is CharacterId {
  return typeof val === "string" && (CHARACTER_IDS as readonly string[]).includes(val);
}

function isDishId(val: unknown): val is DishId {
  return typeof val === "string" && (DISH_IDS as readonly string[]).includes(val);
}

function isFurnitureKind(val: unknown): val is FurnitureKind {
  return typeof val === "string" && (FURNITURE_KINDS as readonly string[]).includes(val);
}

function isMood(val: unknown): val is CharacterPlacement["mood"] {
  return val === "happy" || val === "yummy" || val === "dancing" || val === "waving";
}

function readNum(obj: Record<string, unknown>, key: string, fallback: number): number {
  const v = obj[key];
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

function readStr(obj: Record<string, unknown>, key: string, fallback: string): string {
  const v = obj[key];
  return typeof v === "string" && v.length > 0 ? v : fallback;
}

export function parseHouseState(raw: unknown): HouseState {
  const initial = createInitialHouseState();
  if (!isRecord(raw)) return initial;

  const surfacesOut: Record<SurfaceId, string> = { ...initial.surfaces };
  if (isRecord(raw["surfaces"])) {
    for (const sid of SURFACE_IDS) {
      surfacesOut[sid] = readStr(raw["surfaces"], sid, initial.surfaces[sid]);
    }
  }

  let furnitureOut: readonly FurnitureItem[] = initial.furniture;
  if (Array.isArray(raw["furniture"]) && raw["furniture"].length > 0) {
    const parsed: FurnitureItem[] = [];
    for (const entry of raw["furniture"]) {
      if (!isRecord(entry) || !isFurnitureKind(entry["kind"])) continue;
      const fallback = DEFAULT_FURNITURE.find((f) => f.id === entry["id"]) ?? DEFAULT_FURNITURE[0];
      if (!fallback) continue;
      const clamped = clampHousePosition(
        readNum(entry, "x", fallback.x),
        readNum(entry, "z", fallback.z),
      );
      const sv = entry["styleVariant"];
      const styleVariant: 0 | 1 | 2 = sv === 1 ? 1 : sv === 2 ? 2 : 0;
      parsed.push({
        id: readStr(entry, "id", fallback.id),
        kind: entry["kind"],
        label: readStr(entry, "label", fallback.label),
        x: clamped.x,
        z: clamped.z,
        color: readStr(entry, "color", fallback.color),
        accentColor: readStr(entry, "accentColor", fallback.accentColor),
        styleVariant,
        active: typeof entry["active"] === "boolean" ? entry["active"] : fallback.active,
      });
    }
    if (parsed.length === DEFAULT_FURNITURE.length) {
      furnitureOut = parsed;
    }
  }

  const statsRaw = isRecord(raw["stats"]) ? raw["stats"] : {};
  const stats = {
    cookedCount: Math.max(0, Math.floor(readNum(statsRaw, "cookedCount", 0))),
    paintedCount: Math.max(0, Math.floor(readNum(statsRaw, "paintedCount", 0))),
    movedCount: Math.max(0, Math.floor(readNum(statsRaw, "movedCount", 0))),
    calledCount: Math.max(0, Math.floor(readNum(statsRaw, "calledCount", 0))),
    fedCount: Math.max(0, Math.floor(readNum(statsRaw, "fedCount", 0))),
  };

  const unlockedRaw: CharacterId[] = ["momo", "pip"];
  if (Array.isArray(raw["unlockedCharacters"])) {
    for (const item of raw["unlockedCharacters"]) {
      if (isCharacterId(item) && !unlockedRaw.includes(item)) {
        unlockedRaw.push(item);
      }
    }
  }
  const { unlocked } = evaluateUnlocks(stats, unlockedRaw);

  const placements: CharacterPlacement[] = [];
  if (Array.isArray(raw["characters"])) {
    for (const entry of raw["characters"]) {
      if (!isRecord(entry) || !isCharacterId(entry["id"])) continue;
      const spec = getCharacterSpec(entry["id"]);
      const clamped = clampHousePosition(
        readNum(entry, "x", spec.homeX),
        readNum(entry, "z", spec.homeZ),
      );
      placements.push({
        id: entry["id"],
        x: clamped.x,
        z: clamped.z,
        mood: isMood(entry["mood"]) ? entry["mood"] : "happy",
        moodUntil: 0,
      });
    }
  }

  const dishes: CookedDish[] = [];
  if (Array.isArray(raw["dishes"])) {
    for (const entry of raw["dishes"]) {
      if (!isRecord(entry) || !isDishId(entry["dishId"])) continue;
      const uid = readStr(entry, "uid", `dish-${dishes.length}`);
      const slotIndex = Math.max(0, Math.min(MAX_DISHES - 1, Math.floor(readNum(entry, "slotIndex", dishes.length))));
      dishes.push({ uid, dishId: entry["dishId"], slotIndex });
    }
  }

  return {
    surfaces: surfacesOut,
    furniture: furnitureOut,
    unlockedCharacters: unlocked,
    characters: syncUnlockedPlacements(placements, unlocked),
    dishes,
    stats,
  };
}

export function loadHouseState(storage: StorageAdapter): HouseState {
  const raw = storage.read(config.storageKey);
  if (!raw) return createInitialHouseState();
  try {
    return parseHouseState(JSON.parse(raw));
  } catch {
    return createInitialHouseState();
  }
}

export function saveHouseState(storage: StorageAdapter, state: HouseState): void {
  storage.write(config.storageKey, JSON.stringify(state));
}
