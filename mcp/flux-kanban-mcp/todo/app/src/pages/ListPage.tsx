import { useState } from 'react'
import { useParams } from '@tanstack/react-router'
import { jsPDF } from 'jspdf'
import { useLists, useRenameList } from '../hooks/useLists'
import { AddItem } from '../components/AddItem'
import { ItemList } from '../components/ItemList'
import { SearchBar } from '../components/SearchBar'
import { ImageDrop } from '../components/ImageDrop'

export function ListPage() {
  const { listId } = useParams({ from: '/list/$listId' })
  const { data: lists = [] } = useLists()
  const renameList = useRenameList()
  const [search, setSearch] = useState('')
  const [editingName, setEditingName] = useState(false)
  const [nameInput, setNameInput] = useState('')

  const list = lists.find(l => l.id === listId)

  if (!list) {
    return <div className="home-page"><p>List not found.</p></div>
  }

  const filteredItems = list.items.filter(item =>
    item.name.toLowerCase().includes(search.toLowerCase())
  )

  function startRename() {
    setNameInput(list!.name)
    setEditingName(true)
  }

  function submitRename(e: React.FormEvent) {
    e.preventDefault()
    const name = nameInput.trim()
    if (name) renameList.mutate({ listId, name })
    setEditingName(false)
  }

  function exportToPdf() {
    const doc = new jsPDF()
    doc.setFontSize(18)
    doc.text(list!.name, 14, 20)
    doc.setFontSize(12)
    let y = 35
    list!.items.forEach((item, i) => {
      doc.text(`${i + 1}. [${item.done ? 'x' : ' '}] ${item.name}`, 14, y)
      y += 8
      if (y > 270) {
        doc.addPage()
        y = 20
      }
    })
    doc.save(`${list!.name}.pdf`)
  }

  return (
    <div className="list-page">
      <div className="list-header">
        {editingName ? (
          <form onSubmit={submitRename} className="rename-form">
            <input
              className="input"
              value={nameInput}
              onChange={e => setNameInput(e.target.value)}
              autoFocus
            />
            <button className="btn btn-primary" type="submit">Save</button>
            <button className="btn" type="button" onClick={() => setEditingName(false)}>Cancel</button>
          </form>
        ) : (
          <>
            <h2 className="list-title" onClick={startRename} title="Click to rename">{list.name}</h2>
            <button className="btn" onClick={exportToPdf}>Export PDF</button>
          </>
        )}
      </div>
      <AddItem listId={listId} />
      <SearchBar value={search} onChange={setSearch} />
      <ItemList listId={listId} items={filteredItems} />
      <ImageDrop listId={listId} />
    </div>
  )
}
