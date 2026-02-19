export interface GroceryItem {
  id: string
  name: string
  done: boolean
  createdAt: number
}

export interface GroceryList {
  id: string
  name: string
  items: GroceryItem[]
  createdAt: number
}
