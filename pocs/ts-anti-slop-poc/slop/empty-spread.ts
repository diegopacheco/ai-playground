export function query(page: number, filter: string | undefined) {
  return { page, ...(filter ? { filter } : {}) };
}
