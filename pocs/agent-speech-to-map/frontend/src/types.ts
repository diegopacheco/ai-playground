export interface Coord {
  lat: number
  lon: number
}

export interface Place {
  name: string
  lat: number
  lon: number
  address: string
  opening_hours: string
  distance_m: number
}

export interface Route {
  to: Coord
  mode: string
  distance_m: number
  duration_s: number
  geometry: [number, number][]
}

export interface QueryRequest {
  text: string
  lat: number
  lon: number
  now: string
}

export interface QueryResponse {
  answer: string
  center: Coord
  places: Place[]
  route?: Route
}
