export function clampPage(page: number, total: number): number {
  if (!Number.isFinite(page)) return 1;
  return Math.min(Math.max(Math.round(page), 1), total);
}

export function parsePageParam(raw: string, total: number): number {
  const page = Number.parseInt(raw, 10);
  return Number.isNaN(page) ? 1 : clampPage(page, total);
}

export function nextPage(page: number, total: number): number {
  return clampPage(page + 1, total);
}

export function prevPage(page: number, total: number): number {
  return clampPage(page - 1, total);
}
