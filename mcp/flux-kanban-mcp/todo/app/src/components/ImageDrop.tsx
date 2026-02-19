import { useState, useCallback } from 'react'
import { useAddItems } from '../hooks/useLists'

interface Props {
  listId: string
}

export function ImageDrop({ listId }: Props) {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [added, setAdded] = useState<string | null>(null)
  const addItems = useAddItems()

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    setError(null)
    setAdded(null)
    const file = e.dataTransfer.files[0]
    if (!file || !file.type.startsWith('image/')) {
      setError('Drop an image file (jpg, png, webp, gif)')
      return
    }
    setLoading(true)
    try {
      const formData = new FormData()
      formData.append('image', file)
      const res = await fetch('http://localhost:3001/extract-items', {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) throw new Error('Server error')
      const { items } = await res.json() as { items: string[] }
      if (items.length > 0) {
        addItems.mutate({ listId, names: items })
        setAdded(`Added ${items.length} item(s): ${items.join(', ')}`)
      } else {
        setError('No grocery items found in the image.')
      }
    } catch {
      setError('Failed to extract items. Is the AI server running? (bun run start in server/)')
    } finally {
      setLoading(false)
    }
  }, [listId, addItems])

  return (
    <div
      className={`image-drop ${dragging ? 'dragging' : ''}`}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      {loading
        ? <span>Extracting grocery items from image via AI...</span>
        : <span>Drop an image here — AI will extract grocery items automatically</span>
      }
      {error && <span className="error-text">{error}</span>}
      {added && <span className="success-text">{added}</span>}
    </div>
  )
}
