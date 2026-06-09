import { useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Popup, Polyline, CircleMarker, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'
import type { Coord, Place, Route } from './types'

L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
})

interface Props {
  center: Coord
  places: Place[]
  route: Route | null
}

function walkTime(seconds: number): string {
  return `${Math.max(1, Math.round(seconds / 60))} min`
}

function Fitter({ center, places, route }: Props) {
  const map = useMap()
  useEffect(() => {
    const pts: [number, number][] = [[center.lat, center.lon]]
    places.forEach((p) => pts.push([p.lat, p.lon]))
    route?.geometry.forEach((g) => pts.push([g[0], g[1]]))
    if (pts.length === 1) {
      map.setView(pts[0], 15)
    } else {
      map.fitBounds(pts, { padding: [60, 60] })
    }
  }, [center, places, route, map])
  return null
}

export default function MapView({ center, places, route }: Props) {
  return (
    <MapContainer center={[center.lat, center.lon]} zoom={14} className="map">
      <TileLayer
        attribution="&copy; OpenStreetMap contributors"
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <CircleMarker
        center={[center.lat, center.lon]}
        radius={8}
        pathOptions={{ color: '#2563eb', fillColor: '#3b82f6', fillOpacity: 0.9 }}
      >
        <Popup>You are here</Popup>
      </CircleMarker>
      {places.map((p, i) => (
        <Marker key={`${p.lat},${p.lon},${i}`} position={[p.lat, p.lon]}>
          <Popup>
            <strong>{p.name}</strong>
            {p.address && <div>{p.address}</div>}
            {p.opening_hours && <div>Hours: {p.opening_hours}</div>}
            <div>{p.distance_m} m away</div>
            {route && route.to.lat === p.lat && route.to.lon === p.lon && (
              <div>Walk: {walkTime(route.duration_s)}</div>
            )}
          </Popup>
        </Marker>
      ))}
      {route && (
        <Polyline positions={route.geometry} pathOptions={{ color: '#ef4444', weight: 4 }} />
      )}
      <Fitter center={center} places={places} route={route} />
    </MapContainer>
  )
}
