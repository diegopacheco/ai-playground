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
  const hpColor = hpPercent > 50 ? 'from-emerald-500 to-emerald-400' : hpPercent > 25 ? 'from-yellow-500 to-yellow-400' : 'from-red-500 to-red-400'
  const xpToNext = character.level * 100
  const xpPercent = Math.min(100, (character.xp / xpToNext) * 100)

  return (
    <div className="w-72 bg-[#1e1528]/80 backdrop-blur border border-white/8 rounded-2xl p-5 h-fit shadow-xl shadow-black/20 fade-in">
      <h3
        className="text-amber-300/90 font-bold text-base mb-5"
        style={{ fontFamily: 'Cinzel, serif' }}
      >
        Character
      </h3>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-xs mb-1.5">
            <span className="text-white/40 font-medium uppercase tracking-wider">HP</span>
            <span className="text-white/70 font-medium">{character.hp}/{character.max_hp}</span>
          </div>
          <div className="w-full bg-white/5 rounded-full h-2.5 overflow-hidden">
            <div className={`bg-gradient-to-r ${hpColor} h-2.5 rounded-full transition-all duration-500`} style={{ width: `${hpPercent}%` }} />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1.5">
            <span className="text-white/40 font-medium uppercase tracking-wider">XP</span>
            <span className="text-white/70 font-medium">{character.xp}/{xpToNext}</span>
          </div>
          <div className="w-full bg-white/5 rounded-full h-2.5 overflow-hidden">
            <div className="bg-gradient-to-r from-purple-500 to-purple-400 h-2.5 rounded-full transition-all duration-500" style={{ width: `${xpPercent}%` }} />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 pt-1">
          <div className="bg-white/5 rounded-xl p-3 text-center">
            <div className="text-white/30 text-[10px] font-semibold uppercase tracking-wider mb-1">Level</div>
            <div className="text-amber-200 text-lg font-bold">{character.level}</div>
          </div>
          <div className="bg-white/5 rounded-xl p-3 text-center">
            <div className="text-white/30 text-[10px] font-semibold uppercase tracking-wider mb-1">Gold</div>
            <div className="text-amber-300 text-lg font-bold">{character.gold}</div>
          </div>
        </div>

        <div className="bg-white/5 rounded-xl p-3">
          <div className="text-white/30 text-[10px] font-semibold uppercase tracking-wider mb-1">Location</div>
          <div className="text-cyan-300/80 text-sm font-medium">{character.location}</div>
        </div>

        <div className="pt-1">
          <div className="text-white/30 text-[10px] font-semibold uppercase tracking-wider mb-2">Inventory</div>
          {character.inventory.length === 0 ? (
            <p className="text-white/20 text-xs">Empty</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {character.inventory.map((item, i) => (
                <span key={i} className="text-xs px-2.5 py-1 rounded-lg bg-white/5 border border-white/8 text-amber-200/70">
                  {item}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
