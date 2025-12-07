import { useState, useEffect } from 'react'
import './App.css'
import PokemonCard from './components/PokemonCard'
import Collection from './components/Collection'

const TOTAL_POKEMON = 125
const CARDS_PER_DRAW = 5
const STORAGE_KEY = 'pokemon-collection'

function App() {
  const [collectedCards, setCollectedCards] = useState<number[]>(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved ? JSON.parse(saved) : []
  })
  const [newCards, setNewCards] = useState<number[]>([])
  const [showCollection, setShowCollection] = useState(false)
  const [isDrawing, setIsDrawing] = useState(false)
  const [showVictory, setShowVictory] = useState(false)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(collectedCards))
    if (collectedCards.length === TOTAL_POKEMON) {
      setShowVictory(true)
    }
  }, [collectedCards])

  const getRandomCards = () => {
    if (isDrawing || collectedCards.length === TOTAL_POKEMON) return

    setIsDrawing(true)
    setNewCards([])

    const availableCards = Array.from({ length: TOTAL_POKEMON }, (_, i) => i + 1)
      .filter(id => !collectedCards.includes(id))

    const cardsToAdd: number[] = []
    const shuffled = [...availableCards].sort(() => Math.random() - 0.5)

    for (let i = 0; i < Math.min(CARDS_PER_DRAW, shuffled.length); i++) {
      cardsToAdd.push(shuffled[i])
    }

    setTimeout(() => {
      setNewCards(cardsToAdd)
      setCollectedCards(prev => [...prev, ...cardsToAdd].sort((a, b) => a - b))
      setIsDrawing(false)
    }, 500)
  }

  const resetGame = () => {
    setCollectedCards([])
    setNewCards([])
    setShowVictory(false)
    localStorage.removeItem(STORAGE_KEY)
  }

  const isGameComplete = collectedCards.length === TOTAL_POKEMON

  return (
    <div className="app">
      {showVictory && (
        <div className="victory-overlay" onClick={() => setShowVictory(false)}>
          <div className="victory-content">
            <h1 className="victory-title">Congratulations!</h1>
            <p className="victory-text">You collected all {TOTAL_POKEMON} Pokemon cards!</p>
            <div className="victory-sparkles">
              {Array.from({ length: 20 }).map((_, i) => (
                <div key={i} className="sparkle" style={{
                  left: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 2}s`
                }} />
              ))}
            </div>
          </div>
        </div>
      )}

      <header className="header">
        <h1 className="title">Pokemon Card Collector</h1>
        <div className="progress-container">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${(collectedCards.length / TOTAL_POKEMON) * 100}%` }}
            />
          </div>
          <span className="progress-text">
            {collectedCards.length} / {TOTAL_POKEMON} Cards
          </span>
        </div>
      </header>

      <div className="button-container">
        <button
          className={`action-button draw-button ${isDrawing ? 'drawing' : ''}`}
          onClick={getRandomCards}
          disabled={isDrawing || isGameComplete}
        >
          {isDrawing ? 'Drawing...' : isGameComplete ? 'Collection Complete!' : 'Draw 5 Cards'}
        </button>
        <button
          className="action-button collection-button"
          onClick={() => setShowCollection(!showCollection)}
        >
          {showCollection ? 'Hide Collection' : 'View Collection'}
        </button>
        <button
          className="action-button reset-button"
          onClick={resetGame}
        >
          Reset Game
        </button>
      </div>

      {!showCollection && newCards.length > 0 && (
        <div className="new-cards-section">
          <h2 className="section-title">New Cards!</h2>
          <div className="cards-grid">
            {newCards.map((id, index) => (
              <PokemonCard key={id} pokemonId={id} delay={index * 0.15} isNew={true} />
            ))}
          </div>
        </div>
      )}

      {showCollection && (
        <Collection collectedCards={collectedCards} totalCards={TOTAL_POKEMON} />
      )}
    </div>
  )
}

export default App
