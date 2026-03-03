import { useState, useEffect, useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import Card from './Card'
import Timer from './Timer'
import { submitScore } from '../api'

const SYMBOLS = ['🐶', '🐱', '🦊', '🐸', '🦁', '🐼', '🐨', '🦋']

interface CardData {
  id: number
  value: string
  flipped: boolean
  matched: boolean
}

function buildDeck(): CardData[] {
  const pairs = [...SYMBOLS, ...SYMBOLS]
  for (let i = pairs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pairs[i], pairs[j]] = [pairs[j], pairs[i]]
  }
  return pairs.map((value, id) => ({ id, value, flipped: false, matched: false }))
}

interface BoardProps {
  onGameEnd: () => void
}

export default function Board({ onGameEnd }: BoardProps) {
  const [cards, setCards] = useState<CardData[]>(buildDeck)
  const [selected, setSelected] = useState<number[]>([])
  const [moves, setMoves] = useState(0)
  const [timeLeft, setTimeLeft] = useState(120)
  const [locked, setLocked] = useState(false)
  const [finished, setFinished] = useState(false)
  const [won, setWon] = useState(false)
  const [playerName, setPlayerName] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const queryClient = useQueryClient()

  useEffect(() => {
    if (finished) return
    if (timeLeft <= 0) {
      setFinished(true)
      return
    }
    const t = setTimeout(() => setTimeLeft(s => s - 1), 1000)
    return () => clearTimeout(t)
  }, [timeLeft, finished])

  const checkWin = useCallback((updated: CardData[]) => {
    if (updated.every(c => c.matched)) {
      setFinished(true)
      setWon(true)
    }
  }, [])

  const handleClick = (id: number) => {
    if (locked || finished) return
    const card = cards[id]
    if (card.flipped || card.matched) return
    if (selected.length === 1 && selected[0] === id) return

    const next = cards.map(c => c.id === id ? { ...c, flipped: true } : c)
    setCards(next)

    if (selected.length === 0) {
      setSelected([id])
      return
    }

    const firstId = selected[0]
    setMoves(m => m + 1)
    setSelected([])

    if (next[firstId].value === next[id].value) {
      const matched = next.map(c => (c.id === firstId || c.id === id) ? { ...c, matched: true } : c)
      setCards(matched)
      checkWin(matched)
    } else {
      setLocked(true)
      setTimeout(() => {
        setCards(c => c.map(x => (x.id === firstId || x.id === id) ? { ...x, flipped: false } : x))
        setLocked(false)
      }, 900)
    }
  }

  const handleSubmit = async () => {
    if (!playerName.trim()) return
    await submitScore(playerName.trim(), moves, 120 - timeLeft)
    queryClient.invalidateQueries({ queryKey: ['leaderboard'] })
    setSubmitted(true)
  }

  const handleRestart = () => {
    setCards(buildDeck())
    setSelected([])
    setMoves(0)
    setTimeLeft(120)
    setLocked(false)
    setFinished(false)
    setWon(false)
    setPlayerName('')
    setSubmitted(false)
  }

  return (
    <div className="board-wrapper">
      <div className="hud">
        <Timer seconds={timeLeft} />
        <div className="moves">Moves: {moves}</div>
      </div>

      <div className="board">
        {cards.map(card => (
          <Card
            key={card.id}
            value={card.value}
            flipped={card.flipped}
            matched={card.matched}
            onClick={() => handleClick(card.id)}
          />
        ))}
      </div>

      {finished && (
        <div className="overlay">
          <div className="modal">
            {won ? (
              <>
                <h2>You Won!</h2>
                <p>Moves: {moves} | Time: {120 - timeLeft}s</p>
                {!submitted ? (
                  <>
                    <input
                      placeholder="Your name"
                      value={playerName}
                      onChange={e => setPlayerName(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                    />
                    <button onClick={handleSubmit}>Save Score</button>
                  </>
                ) : (
                  <p>Score saved!</p>
                )}
              </>
            ) : (
              <>
                <h2>Time's Up!</h2>
                <p>Better luck next time.</p>
              </>
            )}
            <button onClick={handleRestart}>Play Again</button>
            <button onClick={onGameEnd}>Leaderboard</button>
          </div>
        </div>
      )}
    </div>
  )
}
