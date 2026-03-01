import type { CardData } from "../types"

const CARD_COLORS: Record<number, string> = {
  1: "bg-red-500",
  2: "bg-blue-500",
  3: "bg-green-500",
  4: "bg-yellow-500",
  5: "bg-purple-500",
  6: "bg-pink-500",
  7: "bg-orange-500",
  8: "bg-teal-500",
}

interface CardProps {
  card: CardData
  index: number
  onClick: () => void
  disabled: boolean
}

export default function Card({ card, index, onClick, disabled }: CardProps) {
  const isRevealed = card.flipped || card.matched
  const colorClass = (card.value !== null && CARD_COLORS[card.value]) || "bg-gray-500"

  return (
    <div className="perspective w-full aspect-square" data-testid={`card-${index}`} onClick={() => !disabled && !isRevealed && onClick()}>
      <div
        className={`relative w-full h-full preserve-3d transition-transform duration-500 cursor-pointer ${
          isRevealed ? "rotate-y-180" : ""
        }`}
      >
        <div className="absolute inset-0 backface-hidden rounded-xl bg-gradient-to-br from-indigo-600 to-indigo-800 border-2 border-indigo-400 flex items-center justify-center shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-200">
          <span className="text-3xl font-bold text-indigo-200">?</span>
        </div>
        <div
          className={`absolute inset-0 backface-hidden rotate-y-180 rounded-xl ${colorClass} border-2 border-white/30 flex items-center justify-center shadow-lg ${
            card.matched ? "opacity-70 ring-2 ring-green-400" : ""
          }`}
        >
          <span className="text-4xl font-bold text-white drop-shadow-md">{card.value}</span>
        </div>
      </div>
    </div>
  )
}
