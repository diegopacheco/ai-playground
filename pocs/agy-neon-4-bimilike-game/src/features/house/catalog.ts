import type { FurnitureItem, SurfaceId } from "./types";

export type PaintSwatch = {
  readonly name: string;
  readonly hex: string;
};

export const PAINT_SWATCHES: readonly PaintSwatch[] = [
  { name: "Strawberry", hex: "#ff8fab" },
  { name: "Buttercup", hex: "#ffe066" },
  { name: "Mint", hex: "#8ce99a" },
  { name: "Sky", hex: "#74c0fc" },
  { name: "Lavender", hex: "#cbb2fe" },
  { name: "Peach", hex: "#ffb085" },
  { name: "Berry", hex: "#ff6b8b" },
  { name: "Aqua", hex: "#63e6be" },
  { name: "Honey Wood", hex: "#e2b17f" },
  { name: "Cream", hex: "#fff6e5" },
] as const;

export const HOUSE_BOUNDS = {
  minX: -6.0,
  maxX: 6.0,
  minZ: -1.55,
  maxZ: 1.45,
} as const;

export const SURFACE_LABELS: Readonly<Record<SurfaceId, string>> = {
  kitchenWall: "Kitchen Wall",
  livingWall: "Living Wall",
  bedroomWall: "Bedroom Wall",
  floor: "House Floor",
  roof: "House Roof",
};

export const DEFAULT_SURFACES: Readonly<Record<SurfaceId, string>> = {
  kitchenWall: "#fff1c9",
  livingWall: "#dff5ff",
  bedroomWall: "#f5e3ff",
  floor: "#f2cfa1",
  roof: "#ff7d9c",
};

export const DEFAULT_FURNITURE: readonly FurnitureItem[] = [
  {
    id: "fridge",
    kind: "fridge",
    label: "Fridge",
    x: -5.55,
    z: -1.25,
    color: "#99e9f2",
    accentColor: "#ffffff",
    styleVariant: 0,
    active: false,
  },
  {
    id: "stove",
    kind: "stove",
    label: "Magic Stove",
    x: -4.15,
    z: -1.25,
    color: "#ff9ebb",
    accentColor: "#ffe066",
    styleVariant: 0,
    active: true,
  },
  {
    id: "table",
    kind: "table",
    label: "Kitchen Table",
    x: -3.45,
    z: 0.25,
    color: "#f7c588",
    accentColor: "#fff8eb",
    styleVariant: 0,
    active: false,
  },
  {
    id: "rug",
    kind: "rug",
    label: "Cozy Rug",
    x: 0.0,
    z: 0.45,
    color: "#ffe066",
    accentColor: "#ff8fab",
    styleVariant: 0,
    active: false,
  },
  {
    id: "sofa",
    kind: "sofa",
    label: "Squishy Sofa",
    x: -0.35,
    z: -1.15,
    color: "#74c0fc",
    accentColor: "#ffe066",
    styleVariant: 0,
    active: false,
  },
  {
    id: "phoneTable",
    kind: "phoneTable",
    label: "Toy Phone Table",
    x: 1.45,
    z: -1.1,
    color: "#ffb085",
    accentColor: "#ff6b8b",
    styleVariant: 0,
    active: false,
  },
  {
    id: "lamp",
    kind: "lamp",
    label: "Sunny Lamp",
    x: -1.75,
    z: -1.25,
    color: "#ffe066",
    accentColor: "#fff4cc",
    styleVariant: 0,
    active: true,
  },
  {
    id: "bed",
    kind: "bed",
    label: "Cloud Bed",
    x: 4.95,
    z: -1.0,
    color: "#cbb2fe",
    accentColor: "#ff8fab",
    styleVariant: 0,
    active: false,
  },
  {
    id: "toyChest",
    kind: "toyChest",
    label: "Toy Chest",
    x: 3.15,
    z: -1.2,
    color: "#8ce99a",
    accentColor: "#ffe066",
    styleVariant: 0,
    active: false,
  },
] as const;

export function clampHousePosition(x: number, z: number): { readonly x: number; readonly z: number } {
  return {
    x: Math.max(HOUSE_BOUNDS.minX, Math.min(HOUSE_BOUNDS.maxX, Math.round(x * 100) / 100)),
    z: Math.max(HOUSE_BOUNDS.minZ, Math.min(HOUSE_BOUNDS.maxZ, Math.round(z * 100) / 100)),
  };
}
