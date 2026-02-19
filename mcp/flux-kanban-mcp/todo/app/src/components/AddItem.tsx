import { useState } from 'react'
import { useAddItem } from '../hooks/useLists'

interface Props {
  listId: string
}

export function AddItem({ listId }: Props) {
  const [name, setName] = useState('')
  const addItem = useAddItem()

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) return
    addItem.mutate({ listId, name: trimmed })
    setName('')
  }

  return (
    <form onSubmit={handleSubmit} className="add-item-form">
      <input
        className="input"
        placeholder="Add grocery item..."
        value={name}
        onChange={e => setName(e.target.value)}
      />
      <button className="btn btn-primary" type="submit">Add Item</button>
    </form>
  )
}
