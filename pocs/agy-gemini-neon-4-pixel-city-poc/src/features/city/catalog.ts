import type {
  CameraPreset,
  LandmarkId,
  LandmarkInfo,
  RoadWaypoint,
  VehicleRoute,
} from "./types";
import { PALETTE } from "./palette";

export const LANDMARKS: readonly LandmarkInfo[] = [
  {
    id: "golden-gate",
    name: "Golden Gate Bridge",
    district: "Golden Gate Strait",
    yearBuilt: 1937,
    description:
      "Art Deco suspension bridge painted in International Orange spanning the bay inlet with twin stepped towers and parabolic cables.",
    position: [-6.8, 3.4, -4.1],
    cameraTarget: [-6.5, 2.0, -4.2],
    zoom: 1.85,
    pinKind: "pin",
  },
  {
    id: "painted-ladies",
    name: "Painted Ladies",
    district: "Alamo Square",
    yearBuilt: 1896,
    description:
      "Row of six pastel Victorian and Edwardian townhouses featuring gabled roofs, ornate bay windows, and front stoops.",
    position: [-9.4, 2.2, 6.0],
    cameraTarget: [-9.0, 1.2, 6.0],
    zoom: 1.9,
    pinKind: "pin",
  },
  {
    id: "palace-of-fine-arts",
    name: "Palace of Fine Arts",
    district: "Marina District",
    yearBuilt: 1915,
    description:
      "Classical Greco-Roman rotunda and curved colonnade reflecting across a serene blue lagoon ringed by trees.",
    position: [-5.1, 1.8, 8.8],
    cameraTarget: [-5.1, 1.0, 8.8],
    zoom: 2.1,
    pinKind: "leaf",
  },
  {
    id: "ferry-building",
    name: "Ferry Building",
    district: "The Embarcadero",
    yearBuilt: 1898,
    description:
      "Beaux-Arts waterfront terminal with a prominent clock tower, arched colonnade, and timber bay pier.",
    position: [-0.2, 3.8, -5.6],
    cameraTarget: [-0.2, 1.8, -5.6],
    zoom: 2.0,
    pinKind: "cart",
  },
  {
    id: "coit-tower",
    name: "Coit Tower",
    district: "Telegraph Hill",
    yearBuilt: 1933,
    description:
      "Fluted cream Art Deco concrete tower overlooking Washington Square park canopy and the bay waterfront.",
    position: [-2.4, 3.2, -1.3],
    cameraTarget: [-2.4, 1.6, -1.3],
    zoom: 2.1,
    pinKind: "leaf",
  },
  {
    id: "transamerica-pyramid",
    name: "Transamerica Pyramid",
    district: "Financial District",
    yearBuilt: 1972,
    description:
      "Iconic 48-story tapered pyramid skyscraper with twin elevator wing fins and a gleaming spire.",
    position: [2.6, 7.2, -8.3],
    cameraTarget: [2.6, 4.2, -8.3],
    zoom: 1.7,
    pinKind: "pin",
  },
  {
    id: "salesforce-tower",
    name: "Salesforce Tower",
    district: "South of Market",
    yearBuilt: 2018,
    description:
      "Tapered blue-glass obelisk skyscraper grounded by a crisp white diagonal lattice podium.",
    position: [6.8, 6.2, -0.3],
    cameraTarget: [6.8, 3.8, -0.3],
    zoom: 1.75,
    pinKind: "leaf",
  },
  {
    id: "cable-car",
    name: "Powell-Hyde Cable Car",
    district: "Nob Hill Corridor",
    yearBuilt: 1873,
    description:
      "Historic crimson-and-ochre streetcar gliding along steel slot rails through central neighborhoods.",
    position: [0.4, 1.2, 4.9],
    cameraTarget: [0.4, 0.8, 4.9],
    zoom: 2.15,
    pinKind: "cart",
  },
] as const;

export const CAMERA_PRESETS: readonly CameraPreset[] = [
  {
    id: "overview",
    label: "Isometric",
    target: [0, 0.75, 0],
    zoom: 1.0,
    azimuth: Math.PI / 4,
    polar: 0.96,
  },
  {
    id: "bridge",
    label: "Golden Gate",
    target: [-6.4, 1.9, -4.0],
    zoom: 1.85,
    azimuth: Math.PI / 4 + 0.14,
    polar: 0.92,
  },
  {
    id: "downtown",
    label: "Skyline",
    target: [5.6, 3.8, -3.5],
    zoom: 1.65,
    azimuth: Math.PI / 4 - 0.12,
    polar: 0.94,
  },
  {
    id: "waterfront",
    label: "Waterfront",
    target: [-1.2, 1.5, -3.6],
    zoom: 1.9,
    azimuth: Math.PI / 4 + 0.08,
    polar: 0.9,
  },
  {
    id: "palace",
    label: "Marina & Shops",
    target: [-1.8, 1.0, 7.2],
    zoom: 1.85,
    azimuth: Math.PI / 4 - 0.08,
    polar: 0.95,
  },
] as const;

export const VEHICLE_ROUTES: readonly VehicleRoute[] = [
  {
    id: "bridge-southbound",
    color: PALETTE.carBlue,
    speed: 0.085,
    phase: 0.15,
    elevation: 0.24,
    points: [
      [-7.18, -10.5],
      [-7.18, 10.6],
    ],
  },
  {
    id: "bridge-northbound",
    color: PALETTE.carRed,
    speed: 0.075,
    phase: 0.62,
    elevation: 0.24,
    points: [
      [-6.42, 10.6],
      [-6.42, -10.5],
    ],
  },
  {
    id: "painted-ladies-sedan",
    color: PALETTE.carWhite,
    speed: 0.068,
    phase: 0.42,
    elevation: 0.24,
    points: [
      [-7.18, 1.4],
      [-7.18, 10.6],
    ],
  },
  {
    id: "market-eastbound",
    color: PALETTE.carCharcoal,
    speed: 0.072,
    phase: 0.1,
    elevation: 0.24,
    points: [
      [-6.4, 9.35],
      [9.15, 9.35],
      [9.15, -4.6],
    ],
  },
  {
    id: "market-westbound",
    color: PALETTE.carRed,
    speed: 0.064,
    phase: 0.55,
    elevation: 0.24,
    points: [
      [9.85, -4.6],
      [9.85, 10.05],
      [-1.5, 10.05],
    ],
  },
  {
    id: "embarcadero-cruiser",
    color: PALETTE.carYellow,
    speed: 0.08,
    phase: 0.33,
    elevation: 0.24,
    points: [
      [1.55, -9.2],
      [1.55, -2.35],
      [9.15, -2.35],
    ],
  },
  {
    id: "front-compact",
    color: PALETTE.carWhite,
    speed: 0.06,
    phase: 0.84,
    elevation: 0.24,
    points: [
      [-1.2, 9.35],
      [9.15, 9.35],
      [9.15, 5.25],
      [-6.4, 5.25],
    ],
  },
] as const;

export function findLandmarkById(id: LandmarkId): LandmarkInfo {
  const match = LANDMARKS.find((item) => item.id === id);
  if (!match) {
    throw new Error(`Unknown landmark id: ${id}`);
  }
  return match;
}

export type PathSample = {
  readonly x: number;
  readonly z: number;
  readonly angle: number;
};

export function samplePolyline(
  points: readonly RoadWaypoint[],
  progress: number,
): PathSample {
  if (points.length === 0) {
    return { x: 0, z: 0, angle: 0 };
  }
  if (points.length === 1) {
    const only = points[0];
    if (!only) {
      return { x: 0, z: 0, angle: 0 };
    }
    return { x: only[0], z: only[1], angle: 0 };
  }

  const lengths: number[] = [];
  let total = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    if (!a || !b) {
      continue;
    }
    const seg = Math.hypot(b[0] - a[0], b[1] - a[1]);
    lengths.push(seg);
    total += seg;
  }

  if (total === 0) {
    const first = points[0];
    return { x: first ? first[0] : 0, z: first ? first[1] : 0, angle: 0 };
  }

  const wrapped = ((progress % 1) + 1) % 1;
  let remaining = wrapped * total;

  for (let i = 0; i < lengths.length; i++) {
    const segLen = lengths[i] ?? 0;
    const a = points[i];
    const b = points[i + 1];
    if (!a || !b) {
      continue;
    }
    if (remaining <= segLen || i === lengths.length - 1) {
      const t = segLen > 0 ? Math.min(1, Math.max(0, remaining / segLen)) : 0;
      const x = a[0] + (b[0] - a[0]) * t;
      const z = a[1] + (b[1] - a[1]) * t;
      const angle = Math.atan2(b[0] - a[0], b[1] - a[1]);
      return { x, z, angle };
    }
    remaining -= segLen;
  }

  const last = points[points.length - 1];
  return { x: last ? last[0] : 0, z: last ? last[1] : 0, angle: 0 };
}
