import { useState } from 'react'
import Board from './components/Board'
import Leaderboard from './components/Leaderboard'
import './App.css'

type View = 'menu' | 'game' | 'leaderboard'

export default function App() {
  const [view, setView] = useState<View>('menu')

  return (
    <div className="app">
      <h1 className="title">Memory Game</h1>
      <nav className="nav">
        <button onClick={() => setView('game')}>Play</button>
        <button onClick={() => setView('leaderboard')}>Leaderboard</button>
      </nav>
      {view === 'menu' && (
        <div className="menu">
          <p>Match all pairs before time runs out!</p>
          <button className="btn-play" onClick={() => setView('game')}>Start Game</button>
        </div>
      )}
      {view === 'game' && <Board onGameEnd={() => setView('leaderboard')} />}
      {view === 'leaderboard' && <Leaderboard />}
    </div>
  )
}
