import { useMemo, useState } from 'react'

function App() {
  const [items, setItems] = useState([
    { id: 1, text: 'Milk', done: false },
    { id: 2, text: 'Eggs', done: false },
    { id: 3, text: 'Bread', done: true }
  ])
  const [input, setInput] = useState('')

  const remaining = useMemo(() => items.filter((item) => !item.done).length, [items])

  const addItem = (event) => {
    event.preventDefault()
    const text = input.trim()
    if (!text) return
    setItems((current) => [...current, { id: Date.now(), text, done: false }])
    setInput('')
  }

  const toggleItem = (id) => {
    setItems((current) =>
      current.map((item) =>
        item.id === id ? { ...item, done: !item.done } : item
      )
    )
  }

  const removeItem = (id) => {
    setItems((current) => current.filter((item) => item.id !== id))
  }

  const clearDone = () => {
    setItems((current) => current.filter((item) => !item.done))
  }

  return (
    <main className="page">
      <section className="card">
        <h1>Grocery Todo</h1>
        <p>{remaining} pending</p>

        <form onSubmit={addItem} className="row">
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Add grocery item"
            aria-label="Add grocery item"
          />
          <button type="submit">Add</button>
        </form>

        <ul>
          {items.map((item) => (
            <li key={item.id} className={item.done ? 'done' : ''}>
              <label>
                <input
                  type="checkbox"
                  checked={item.done}
                  onChange={() => toggleItem(item.id)}
                />
                <span>{item.text}</span>
              </label>
              <button onClick={() => removeItem(item.id)} type="button">
                Remove
              </button>
            </li>
          ))}
        </ul>

        {items.some((item) => item.done) && (
          <button className="clear" onClick={clearDone} type="button">
            Clear completed
          </button>
        )}
      </section>
    </main>
  )
}

export default App
