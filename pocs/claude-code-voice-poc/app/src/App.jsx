import { useNavigate } from '@tanstack/react-router'
import { useForm } from '@tanstack/react-form'
import { setPlayerNames } from './store.js'
import './App.css'

function App() {
  const navigate = useNavigate()

  const form = useForm({
    defaultValues: { player1: '', player2: '' },
    onSubmit: ({ value }) => {
      setPlayerNames(value.player1, value.player2)
      navigate({ to: '/cards' })
    },
  })

  return (
    <div className="home">
      <h1 className="title">Pokemon Battle Arena</h1>
      <p className="subtitle">Enter player names to begin</p>
      <form
        onSubmit={(e) => { e.preventDefault(); form.handleSubmit() }}
        className="player-form"
      >
        <form.Field name="player1">
          {(field) => (
            <div className="field">
              <label>Player 1</label>
              <input
                value={field.state.value}
                onChange={(e) => field.handleChange(e.target.value)}
                placeholder="Enter name..."
                required
              />
            </div>
          )}
        </form.Field>
        <div className="vs-text">VS</div>
        <form.Field name="player2">
          {(field) => (
            <div className="field">
              <label>Player 2</label>
              <input
                value={field.state.value}
                onChange={(e) => field.handleChange(e.target.value)}
                placeholder="Enter name..."
                required
              />
            </div>
          )}
        </form.Field>
        <button type="submit" className="start-btn">Start Battle!</button>
      </form>
    </div>
  )
}

export default App
