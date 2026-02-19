import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getLists, saveLists } from '../store/listsStore'
import type { GroceryList, GroceryItem } from '../types'

const QUERY_KEY = ['lists']

function genId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36)
}

export function useLists() {
  return useQuery({
    queryKey: QUERY_KEY,
    queryFn: getLists,
    staleTime: Infinity,
  })
}

export function useCreateList() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (name: string) => {
      const lists = getLists()
      const newList: GroceryList = {
        id: genId(),
        name,
        items: [],
        createdAt: Date.now(),
      }
      saveLists([...lists, newList])
      return newList
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  })
}

export function useDeleteList() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (listId: string) => {
      saveLists(getLists().filter(l => l.id !== listId))
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  })
}

export function useRenameList() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ listId, name }: { listId: string; name: string }) => {
      saveLists(getLists().map(l => l.id === listId ? { ...l, name } : l))
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  })
}

export function useAddItem() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ listId, name }: { listId: string; name: string }) => {
      const item: GroceryItem = { id: genId(), name, done: false, createdAt: Date.now() }
      saveLists(getLists().map(l =>
        l.id === listId ? { ...l, items: [...l.items, item] } : l
      ))
      return item
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  })
}

export function useAddItems() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ listId, names }: { listId: string; names: string[] }) => {
      const newItems: GroceryItem[] = names.map(name => ({
        id: genId(),
        name,
        done: false,
        createdAt: Date.now(),
      }))
      saveLists(getLists().map(l =>
        l.id === listId ? { ...l, items: [...l.items, ...newItems] } : l
      ))
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  })
}

export function useToggleItem() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ listId, itemId }: { listId: string; itemId: string }) => {
      saveLists(getLists().map(l =>
        l.id === listId
          ? { ...l, items: l.items.map(i => i.id === itemId ? { ...i, done: !i.done } : i) }
          : l
      ))
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  })
}

export function useDeleteItem() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ listId, itemId }: { listId: string; itemId: string }) => {
      saveLists(getLists().map(l =>
        l.id === listId ? { ...l, items: l.items.filter(i => i.id !== itemId) } : l
      ))
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  })
}
