import { useState } from 'react'
import './PokemonCard.css'

interface PokemonCardProps {
  pokemonId: number
  delay?: number
  isNew?: boolean
  isCollected?: boolean
}

const POKEMON_NAMES: { [key: number]: string } = {
  1: 'Bulbasaur', 2: 'Ivysaur', 3: 'Venusaur', 4: 'Charmander', 5: 'Charmeleon',
  6: 'Charizard', 7: 'Squirtle', 8: 'Wartortle', 9: 'Blastoise', 10: 'Caterpie',
  11: 'Metapod', 12: 'Butterfree', 13: 'Weedle', 14: 'Kakuna', 15: 'Beedrill',
  16: 'Pidgey', 17: 'Pidgeotto', 18: 'Pidgeot', 19: 'Rattata', 20: 'Raticate',
  21: 'Spearow', 22: 'Fearow', 23: 'Ekans', 24: 'Arbok', 25: 'Pikachu',
  26: 'Raichu', 27: 'Sandshrew', 28: 'Sandslash', 29: 'Nidoran F', 30: 'Nidorina',
  31: 'Nidoqueen', 32: 'Nidoran M', 33: 'Nidorino', 34: 'Nidoking', 35: 'Clefairy',
  36: 'Clefable', 37: 'Vulpix', 38: 'Ninetales', 39: 'Jigglypuff', 40: 'Wigglytuff',
  41: 'Zubat', 42: 'Golbat', 43: 'Oddish', 44: 'Gloom', 45: 'Vileplume',
  46: 'Paras', 47: 'Parasect', 48: 'Venonat', 49: 'Venomoth', 50: 'Diglett',
  51: 'Dugtrio', 52: 'Meowth', 53: 'Persian', 54: 'Psyduck', 55: 'Golduck',
  56: 'Mankey', 57: 'Primeape', 58: 'Growlithe', 59: 'Arcanine', 60: 'Poliwag',
  61: 'Poliwhirl', 62: 'Poliwrath', 63: 'Abra', 64: 'Kadabra', 65: 'Alakazam',
  66: 'Machop', 67: 'Machoke', 68: 'Machamp', 69: 'Bellsprout', 70: 'Weepinbell',
  71: 'Victreebel', 72: 'Tentacool', 73: 'Tentacruel', 74: 'Geodude', 75: 'Graveler',
  76: 'Golem', 77: 'Ponyta', 78: 'Rapidash', 79: 'Slowpoke', 80: 'Slowbro',
  81: 'Magnemite', 82: 'Magneton', 83: 'Farfetchd', 84: 'Doduo', 85: 'Dodrio',
  86: 'Seel', 87: 'Dewgong', 88: 'Grimer', 89: 'Muk', 90: 'Shellder',
  91: 'Cloyster', 92: 'Gastly', 93: 'Haunter', 94: 'Gengar', 95: 'Onix',
  96: 'Drowzee', 97: 'Hypno', 98: 'Krabby', 99: 'Kingler', 100: 'Voltorb',
  101: 'Electrode', 102: 'Exeggcute', 103: 'Exeggutor', 104: 'Cubone', 105: 'Marowak',
  106: 'Hitmonlee', 107: 'Hitmonchan', 108: 'Lickitung', 109: 'Koffing', 110: 'Weezing',
  111: 'Rhyhorn', 112: 'Rhydon', 113: 'Chansey', 114: 'Tangela', 115: 'Kangaskhan',
  116: 'Horsea', 117: 'Seadra', 118: 'Goldeen', 119: 'Seaking', 120: 'Staryu',
  121: 'Starmie', 122: 'Mr. Mime', 123: 'Scyther', 124: 'Jynx', 125: 'Electabuzz'
}

function PokemonCard({ pokemonId, delay = 0, isNew = false, isCollected = true }: PokemonCardProps) {
  const [isFlipped, setIsFlipped] = useState(isNew)
  const [imageLoaded, setImageLoaded] = useState(false)

  const paddedId = String(pokemonId).padStart(3, '0')
  const imageUrl = `https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/images/${paddedId}.png`
  const pokemonName = POKEMON_NAMES[pokemonId] || `Pokemon #${pokemonId}`

  const handleClick = () => {
    if (isCollected) {
      setIsFlipped(!isFlipped)
    }
  }

  return (
    <div
      className={`pokemon-card ${isNew ? 'new-card' : ''} ${!isCollected ? 'not-collected' : ''}`}
      style={{ animationDelay: `${delay}s` }}
      onClick={handleClick}
    >
      <div className={`card-inner ${isFlipped ? 'flipped' : ''}`}>
        <div className="card-front">
          <div className="card-shine" />
          <div className="card-number">#{paddedId}</div>
          <div className="card-image-container">
            {!imageLoaded && <div className="image-placeholder" />}
            <img
              src={imageUrl}
              alt={pokemonName}
              className={`card-image ${imageLoaded ? 'loaded' : ''}`}
              onLoad={() => setImageLoaded(true)}
              loading="lazy"
            />
          </div>
          <div className="card-name">{pokemonName}</div>
          {isNew && <div className="new-badge">NEW!</div>}
        </div>
        <div className="card-back">
          <div className="card-back-pattern" />
          <div className="pokeball-logo" />
        </div>
      </div>
    </div>
  )
}

export default PokemonCard
