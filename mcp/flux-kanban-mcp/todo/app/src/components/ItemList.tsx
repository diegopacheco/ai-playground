import { useToggleItem, useDeleteItem } from '../hooks/useLists'
import type { GroceryItem } from '../types'

interface Props {
  listId: string
  items: GroceryItem[]
}

export function ItemList({ listId, items }: Props) {
  const toggleItem = useToggleItem()
  const deleteItem = useDeleteItem()

  if (items.length === 0) {
    return <p className="empty-state">No items. Add one above.</p>
  }

  return (
    <ul className="item-list">
      {items.map(item => (
        <li key={item.id} className={`item ${item.done ? 'item-done' : ''}`}>
          <label className="item-label">
            <input
              type="checkbox"
              checked={item.done}
              onChange={() => toggleItem.mutate({ listId, itemId: item.id })}
            />
            <span className="item-name">{item.name}</span>
          </label>
          <button
            className="btn-icon btn-danger"
            onClick={() => deleteItem.mutate({ listId, itemId: item.id })}
            title="Delete item"
          >
            ✕
          </button>
        </li>
      ))}
    </ul>
  )
}
