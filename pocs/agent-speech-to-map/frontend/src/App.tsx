import { useState } from 'react'
import MapView from './MapView'
import VoiceControl from './VoiceControl'
import type { Coord, Place, Route, QueryResponse } from './types'

const DEFAULT_CENTER: Coord = { lat: 40.758, lon: -73.9855 }

export default function App() {
  const [center, setCenter] = useState<Coord>(DEFAULT_CENTER)
  const [places, setPlaces] = useState<Place[]>([])
  const [route, setRoute] = useState<Route | null>(null)

  function handleResult(resp: QueryResponse) {
    setCenter(resp.center)
    setPlaces(resp.places)
    setRoute(resp.route ?? null)
  }

  return (
    <div className="app">
      <MapView center={center} places={places} route={route} />
      <VoiceControl onResult={handleResult} />
    </div>
  )
}
