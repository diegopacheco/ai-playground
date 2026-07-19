export type GridKey = "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight";

export function isGridKey(key: string): key is GridKey {
  return key === "ArrowUp" || key === "ArrowDown" || key === "ArrowLeft" || key === "ArrowRight";
}

export function nextIndex(key: GridKey, current: number, count: number, columns: number): number {
  if (count === 0) {
    return 0;
  }
  const perRow = Math.max(columns, 1);
  if (key === "ArrowLeft") {
    return (current - 1 + count) % count;
  }
  if (key === "ArrowRight") {
    return (current + 1) % count;
  }
  if (key === "ArrowDown") {
    const candidate = current + perRow;
    return candidate < count ? candidate : current % perRow;
  }
  const candidate = current - perRow;
  if (candidate >= 0) {
    return candidate;
  }
  const column = current % perRow;
  for (let row = Math.floor((count - 1) / perRow); row >= 0; row--) {
    const bottom = row * perRow + column;
    if (bottom < count) {
      return bottom;
    }
  }
  return current;
}

export function measureColumns(container: HTMLElement | null, itemSelector: string): number {
  if (!container) {
    return 1;
  }
  const items = Array.from(container.querySelectorAll<HTMLElement>(itemSelector));
  if (items.length === 0) {
    return 1;
  }
  const firstTop = items[0].offsetTop;
  const inFirstRow = items.filter((item) => item.offsetTop === firstTop).length;
  return Math.max(inFirstRow, 1);
}
