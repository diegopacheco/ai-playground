export const LIGHTING_MODES = ["day", "golden", "night"] as const;
export type LightingMode = (typeof LIGHTING_MODES)[number];

export const PIN_KINDS = ["leaf", "pin", "cart"] as const;
export type PinKind = (typeof PIN_KINDS)[number];

export type Vec3 = readonly [number, number, number];

export type LandmarkId =
  | "golden-gate"
  | "painted-ladies"
  | "palace-of-fine-arts"
  | "ferry-building"
  | "coit-tower"
  | "transamerica-pyramid"
  | "salesforce-tower"
  | "cable-car";

export type LandmarkInfo = {
  readonly id: LandmarkId;
  readonly name: string;
  readonly district: string;
  readonly yearBuilt: number;
  readonly description: string;
  readonly position: Vec3;
  readonly cameraTarget: Vec3;
  readonly zoom: number;
  readonly pinKind: PinKind;
};

export type CameraPreset = {
  readonly id: "overview" | "bridge" | "downtown" | "waterfront" | "palace";
  readonly label: string;
  readonly target: Vec3;
  readonly zoom: number;
  readonly azimuth: number;
  readonly polar: number;
};

export type RoadWaypoint = readonly [number, number];

export type VehicleRoute = {
  readonly id: string;
  readonly color: number;
  readonly speed: number;
  readonly phase: number;
  readonly points: readonly RoadWaypoint[];
  readonly elevation: number;
};

export type CitySceneState = {
  readonly lighting: LightingMode;
  readonly selectedLandmark: LandmarkId | null;
  readonly paused: boolean;
  readonly showPins: boolean;
  readonly crispOutlines: boolean;
};

export type LightingPreset = {
  readonly background: number;
  readonly ambientColor: number;
  readonly ambientIntensity: number;
  readonly sunColor: number;
  readonly sunIntensity: number;
  readonly sunPosition: Vec3;
  readonly fillColor: number;
  readonly fillIntensity: number;
  readonly lampGlow: number;
  readonly windowGlow: number;
};
