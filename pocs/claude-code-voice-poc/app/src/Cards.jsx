import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { useStore } from './useStore.js'
import { setPlayerCards } from './store.js'
import './Cards.css'

function shuffle(arr) {
  const a = [...arr]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

async function fetchPokemonBatch() {
  const ids = shuffle(Array.from({ length: 151 }, (_, i) => i + 1)).slice(0, 6)
  const results = await Promise.all(
    ids.map((id) =>
      fetch(`https://pokeapi.co/api/v2/pokemon/${id}`).then((r) => r.json())
    )
  )
  return results.map((p) => ({
    id: p.id,
    name: p.name,
    sprite: p.sprites.front_default,
    hp: p.stats[0].base_stat,
    attack: p.stats[1].base_stat,
    defense: p.stats[2].base_stat,
    speed: p.stats[5].base_stat,
    types: p.types.map((t) => t.type.name),
    power: p.stats[0].base_stat + p.stats[1].base_stat + p.stats[2].base_stat + p.stats[5].base_stat,
  }))
}

function PokemonCard({ pokemon, flipped, onClick, index }) {
  return (
    <div
      className={`card-container ${flipped ? 'flipped' : ''}`}
      onClick={onClick}
      style={{ animationDelay: `${index * 0.1}s` }}
    >
      <div className="card-inner">
        <div className="card-back">
          <div className="pokeball-icon">?</div>
        </div>
        <div className="card-front">
          {pokemon && (
            <>
              <img src={pokemon.sprite} alt={pokemon.name} />
              <h3>{pokemon.name}</h3>
              <div className="card-stats">
                <span>HP: {pokemon.hp}</span>
                <span>ATK: {pokemon.attack}</span>
                <span>DEF: {pokemon.defense}</span>
                <span>SPD: {pokemon.speed}</span>
              </div>
              <div className="card-types">
                {pokemon.types.map((t) => (
                  <span key={t} className={`type-badge ${t}`}>{t}</span>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function Cards() {
  const navigate = useNavigate()
  const state = useStore()
  const [flipped, setFlipped] = useState({})

  const { data: pokemon, isLoading, refetch } = useQuery({
    queryKey: ['battle-cards'],
    queryFn: fetchPokemonBatch,
    staleTime: Infinity,
  })

  const handleFlip = (idx) => {
    setFlipped((prev) => ({ ...prev, [idx]: !prev[idx] }))
  }

  const handleShuffle = () => {
    setFlipped({})
    refetch()
  }

  useEffect(() => {
    if (pokemon && Object.keys(flipped).filter((k) => flipped[k]).length === 6) {
      setPlayerCards(1, pokemon.slice(0, 3))
      setPlayerCards(2, pokemon.slice(3, 6))
    }
  }, [flipped, pokemon])

  const allFlipped = Object.keys(flipped).filter((k) => flipped[k]).length === 6
  const p1Cards = pokemon ? pokemon.slice(0, 3) : []
  const p2Cards = pokemon ? pokemon.slice(3, 6) : []

  if (isLoading) return <div className="loading">Loading Pokemon cards...</div>

  return (
    <div className="cards-page">
      <h2>{state.player1.name}'s Cards</h2>
      <div className="card-row">
        {p1Cards.map((p, i) => (
          <PokemonCard key={p.id} pokemon={p} flipped={flipped[i]} onClick={() => handleFlip(i)} index={i} />
        ))}
      </div>

      <h2>{state.player2.name}'s Cards</h2>
      <div className="card-row">
        {p2Cards.map((p, i) => (
          <PokemonCard key={p.id} pokemon={p} flipped={flipped[i + 3]} onClick={() => handleFlip(i + 3)} index={i + 3} />
        ))}
      </div>

      <div className="cards-actions">
        <button className="action-btn shuffle-btn" onClick={handleShuffle}>Shuffle</button>
        {allFlipped && (
          <button className="action-btn battle-btn" onClick={() => navigate({ to: '/battle' })}>
            Go to Battle!
          </button>
        )}
      </div>
    </div>
  )
}

export default Cards
