import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useVirtualizer } from '@tanstack/react-virtual'
import { useRef } from 'react'
import './Pokedex.css'

const TYPES = ['all', 'fire', 'water', 'grass', 'electric', 'psychic', 'normal', 'fighting', 'poison', 'ground', 'rock', 'bug', 'ghost', 'dragon', 'ice', 'flying', 'fairy']

async function fetchAllPokemon() {
  const res = await fetch('https://pokeapi.co/api/v2/pokemon?limit=151')
  const data = await res.json()
  const details = await Promise.all(
    data.results.map((p) => fetch(p.url).then((r) => r.json()))
  )
  return details.map((p) => ({
    id: p.id,
    name: p.name,
    sprite: p.sprites.front_default,
    hp: p.stats[0].base_stat,
    attack: p.stats[1].base_stat,
    defense: p.stats[2].base_stat,
    speed: p.stats[5].base_stat,
    types: p.types.map((t) => t.type.name),
  }))
}

function Pokedex() {
  const [selectedType, setSelectedType] = useState('all')
  const parentRef = useRef(null)

  const { data: allPokemon, isLoading } = useQuery({
    queryKey: ['all-pokemon'],
    queryFn: fetchAllPokemon,
    staleTime: 1000 * 60 * 30,
  })

  const filtered = allPokemon
    ? selectedType === 'all'
      ? allPokemon
      : allPokemon.filter((p) => p.types.includes(selectedType))
    : []

  const virtualizer = useVirtualizer({
    count: Math.ceil(filtered.length / 4),
    getScrollElement: () => parentRef.current,
    estimateSize: () => 200,
    overscan: 3,
  })

  if (isLoading) return <div className="loading">Loading Pokedex...</div>

  return (
    <div className="pokedex-page">
      <h2>Pokedex</h2>
      <div className="type-filters">
        {TYPES.map((t) => (
          <button
            key={t}
            className={`type-filter-btn ${t} ${selectedType === t ? 'selected' : ''}`}
            onClick={() => setSelectedType(t)}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="pokemon-grid-wrapper" ref={parentRef}>
        <div style={{ height: virtualizer.getTotalSize(), width: '100%', position: 'relative' }}>
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const startIdx = virtualRow.index * 4
            const rowPokemon = filtered.slice(startIdx, startIdx + 4)
            return (
              <div
                key={virtualRow.key}
                className="pokemon-row"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: virtualRow.size,
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                {rowPokemon.map((p) => (
                  <div key={p.id} className="pokedex-card">
                    <span className="poke-number">#{String(p.id).padStart(3, '0')}</span>
                    <img src={p.sprite} alt={p.name} />
                    <h4>{p.name}</h4>
                    <div className="poke-types">
                      {p.types.map((t) => (
                        <span key={t} className={`type-badge ${t}`}>{t}</span>
                      ))}
                    </div>
                    <div className="poke-stats-mini">
                      <span>HP {p.hp}</span>
                      <span>ATK {p.attack}</span>
                      <span>DEF {p.defense}</span>
                      <span>SPD {p.speed}</span>
                    </div>
                  </div>
                ))}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export default Pokedex
