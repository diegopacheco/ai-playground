interface GameCellProps {
  value: string
  isLastMove: boolean
}

export function GameCell({ value, isLastMove }: GameCellProps) {
  let bgColor = 'bg-white'
  if (value === 'X') {
    bgColor = 'bg-red-500'
  } else if (value === 'O') {
    bgColor = 'bg-yellow-400'
  }

  return (
    <div className="w-14 h-14 p-1">
      <div
        className={`w-full h-full rounded-full ${bgColor} ${isLastMove ? 'ring-4 ring-green-400' : ''} transition-all duration-300`}
      />
    </div>
  )
}
