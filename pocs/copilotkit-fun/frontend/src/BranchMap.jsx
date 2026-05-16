import React, { useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, CircleMarker, Polyline } from 'react-leaflet';
import L from 'leaflet';

function pinSvg(fill) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 42" width="30" height="42">
    <path d="M15 0C6.7 0 0 6.7 0 15c0 11.2 15 27 15 27s15-15.8 15-27C30 6.7 23.3 0 15 0z" fill="${fill}" stroke="white" stroke-width="2"/>
    <circle cx="15" cy="15" r="5" fill="white"/>
  </svg>`;
}

const markerIcon = L.divIcon({
  className: 'branch-marker',
  html: pinSvg('#3b82f6'),
  iconSize: [30, 42],
  iconAnchor: [15, 42],
  popupAnchor: [0, -36]
});

const nearestIcon = L.divIcon({
  className: 'branch-marker nearest',
  html: pinSvg('#22c55e'),
  iconSize: [34, 48],
  iconAnchor: [17, 48],
  popupAnchor: [0, -42]
});

function haversine(a, b) {
  const R = 6371;
  const toRad = d => d * Math.PI / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const x = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
}

export default function BranchMap({ branches, userLocation, symbol }) {
  const nearest = useMemo(() => {
    if (!userLocation || branches.length === 0) return null;
    return branches.reduce((min, b) => {
      const d = haversine(userLocation, b);
      return !min || d < min.d ? { ...b, d } : min;
    }, null);
  }, [branches, userLocation]);

  const center = nearest
    ? [(nearest.lat + userLocation.lat) / 2, (nearest.lng + userLocation.lng) / 2]
    : userLocation
      ? [userLocation.lat, userLocation.lng]
      : [20, 0];

  return (
    <div className="widget map-widget">
      <div className="map-header">
        <h3>{symbol} Branches</h3>
        {nearest && (
          <div className="nearest-info">
            Nearest: <strong>{nearest.name}</strong> — {nearest.d.toFixed(0)} km
          </div>
        )}
      </div>
      <MapContainer center={center} zoom={3} style={{ height: '480px', width: '100%', borderRadius: '8px' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        {userLocation && (
          <CircleMarker
            center={[userLocation.lat, userLocation.lng]}
            radius={10}
            pathOptions={{ color: '#3b82f6', fillColor: '#3b82f6', fillOpacity: 0.6 }}
          >
            <Popup>You are here</Popup>
          </CircleMarker>
        )}
        {branches.map((b, i) => {
          const isNearest = nearest && b.name === nearest.name;
          return (
            <Marker key={i} position={[b.lat, b.lng]} icon={isNearest ? nearestIcon : markerIcon}>
              <Popup>
                <strong>{b.name}</strong>
                <br />
                {b.address}
                {isNearest && <div style={{ marginTop: 4, color: '#16a34a' }}>★ Nearest branch ({nearest.d.toFixed(0)} km)</div>}
              </Popup>
            </Marker>
          );
        })}
        {nearest && userLocation && (
          <Polyline
            positions={[[userLocation.lat, userLocation.lng], [nearest.lat, nearest.lng]]}
            pathOptions={{ color: '#22c55e', dashArray: '6, 8' }}
          />
        )}
      </MapContainer>
    </div>
  );
}
