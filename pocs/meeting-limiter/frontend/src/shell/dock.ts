export type Side = "left" | "right";

export type HiddenPanel = { readonly id: string; readonly title: string; readonly side: Side };

export function dockSide(rect: { readonly left: number; readonly width: number }, windowWidth: number): Side {
  return rect.left + rect.width / 2 < windowWidth / 2 ? "left" : "right";
}

export function hidePanel(list: readonly HiddenPanel[], panel: HiddenPanel): readonly HiddenPanel[] {
  return [...list.filter((p) => p.id !== panel.id), panel];
}

export function restorePanel(list: readonly HiddenPanel[], id: string): readonly HiddenPanel[] {
  return list.filter((p) => p.id !== id);
}

export function parseHidden(raw: unknown): readonly HiddenPanel[] | undefined {
  if (!Array.isArray(raw)) return undefined;
  const out: HiddenPanel[] = [];
  for (const item of raw) {
    if (typeof item !== "object" || item === null) continue;
    if (!("id" in item) || !("title" in item) || !("side" in item)) continue;
    const { id, title, side } = item;
    if (typeof id === "string" && typeof title === "string" && (side === "left" || side === "right")) {
      out.push({ id, title, side });
    }
  }
  return out;
}
