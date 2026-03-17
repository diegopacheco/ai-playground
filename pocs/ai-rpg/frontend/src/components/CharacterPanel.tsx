interface Character {
  hp: number
  max_hp: number
  level: number
  xp: number
  gold: number
  inventory: string[]
  location: string
}

interface Props {
  character: Character
}

export default function CharacterPanel({ character }: Props) {
  const hpPercent = Math.max(0, Math.min(100, (character.hp / character.max_hp) * 100))
  const hpColor = hpPercent > 50 ? 'bg-emerald-500' : hpPercent > 25 ? 'bg-yellow-500' : 'bg-red-500'

  return (
    <div className="w-64 bg-[#0d0d14] border border-amber-900/30 rounded-lg p-4 h-fit">
      <h3 className="text-amber-400 font-bold text-lg mb-4">Character</h3>

      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-400">HP</span>
            <span className="text-white">{character.hp}/{character.max_hp}</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2">
            <div className={`${hpColor} h-2 rounded-full transition-all`} style={{ width: `${hpPercent}%` }} />
          </div>
        </div>

        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Level</span>
          <span className="text-white">{character.level}</span>
        </div>

        <div className="flex justify-between text-sm">
          <span className="text-gray-400">XP</span>
          <span className="text-white">{character.xp}</span>
        </div>

        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Gold</span>
          <span className="text-amber-300">{character.gold}</span>
        </div>

        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Location</span>
          <span className="text-cyan-300 text-right text-xs max-w-[120px]">{character.location}</span>
        </div>

        <div className="border-t border-amber-900/30 pt-3">
          <h4 className="text-amber-400 text-sm font-bold mb-2">Inventory</h4>
          {character.inventory.length === 0 ? (
            <p className="text-gray-500 text-sm">Empty</p>
          ) : (
            <ul className="space-y-1">
              {character.inventory.map((item, i) => (
                <li key={i} className="text-gray-300 text-sm">- {item}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}
