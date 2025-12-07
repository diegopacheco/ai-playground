import PokemonCard from './PokemonCard'
import './Collection.css'

interface CollectionProps {
  collectedCards: number[]
  totalCards: number
}

function Collection({ collectedCards, totalCards }: CollectionProps) {
  const allCards = Array.from({ length: totalCards }, (_, i) => i + 1)

  return (
    <div className="collection-container">
      <h2 className="collection-title">Your Collection</h2>
      <div className="collection-stats">
        <div className="stat">
          <span className="stat-value">{collectedCards.length}</span>
          <span className="stat-label">Collected</span>
        </div>
        <div className="stat">
          <span className="stat-value">{totalCards - collectedCards.length}</span>
          <span className="stat-label">Remaining</span>
        </div>
        <div className="stat">
          <span className="stat-value">{Math.round((collectedCards.length / totalCards) * 100)}%</span>
          <span className="stat-label">Complete</span>
        </div>
      </div>
      <div className="collection-grid">
        {allCards.map((id, index) => (
          <PokemonCard
            key={id}
            pokemonId={id}
            delay={index * 0.02}
            isCollected={collectedCards.includes(id)}
          />
        ))}
      </div>
    </div>
  )
}

export default Collection
