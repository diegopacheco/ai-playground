import { useState } from 'react'
import { Link } from '@tanstack/react-router'
import { useLists, useCreateList, useDeleteList } from '../hooks/useLists'

export function ListSidebar() {
  const { data: lists = [] } = useLists()
  const createList = useCreateList()
  const deleteList = useDeleteList()
  const [newListName, setNewListName] = useState('')

  function handleCreate(e: React.FormEvent) {
    e.preventDefault()
    const name = newListName.trim()
    if (!name) return
    createList.mutate(name)
    setNewListName('')
  }

  return (
    <aside className="sidebar">
      <h1 className="sidebar-title">Grocery Lists</h1>
      <form onSubmit={handleCreate} className="new-list-form">
        <input
          className="input"
          placeholder="New list name..."
          value={newListName}
          onChange={e => setNewListName(e.target.value)}
        />
        <button className="btn btn-primary" type="submit">Add</button>
      </form>
      <nav className="list-nav">
        {lists.map(list => (
          <div key={list.id} className="list-nav-item">
            <Link
              to="/list/$listId"
              params={{ listId: list.id }}
              className="list-nav-link"
              activeProps={{ className: 'list-nav-link active' }}
            >
              <span>{list.name}</span>
              <span className="list-count">{list.items.length}</span>
            </Link>
            <button
              className="btn-icon btn-danger"
              onClick={() => deleteList.mutate(list.id)}
              title="Delete list"
            >
              ✕
            </button>
          </div>
        ))}
        {lists.length === 0 && (
          <p className="empty-state">No lists yet.</p>
        )}
      </nav>
    </aside>
  )
}
