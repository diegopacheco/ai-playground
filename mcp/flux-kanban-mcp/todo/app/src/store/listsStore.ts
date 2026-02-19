import type { GroceryList } from '../types'

const KEY = 'grocery_lists'

export function getLists(): GroceryList[] {
  const raw = localStorage.getItem(KEY)
  if (!raw) return []
  try {
    return JSON.parse(raw)
  } catch {
    return []
  }
}

export function saveLists(lists: GroceryList[]): void {
  localStorage.setItem(KEY, JSON.stringify(lists))
}
