export interface Player {
  id: number
  name: string
}

export interface CardData {
  position: number
  value: number | null
  flipped: boolean
  matched: boolean
}

export interface Game {
  id: number
  player_id: number
  board: CardData[]
  moves: number
  matches_found: number
  total_pairs: number
  status: string
  score: number
}

export interface FlipResponse {
  game: Game
  matched: boolean | null
}

export interface LeaderboardEntry {
  player_name: string
  score: number
  moves: number
}
