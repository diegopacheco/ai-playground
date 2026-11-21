export const CABINET = {
  half: 0.6,
  wallHeight: 1.45,
  wallThickness: 0.05,
  floorThickness: 0.05,
  baseDepth: 0.75,
  chuteX: -0.22,
  chuteZ: 0.22,
  barrierHeight: 0.24,
  barrierThickness: 0.02,
  railY: 1.38,
  fallLine: -0.4,
} as const;

export const CLAW = {
  reach: 0.45,
  topY: 1.05,
  minY: 0.32,
  homeX: -0.41,
  homeZ: 0.41,
  hubRadius: 0.06,
  hubHalfHeight: 0.03,
  hingeRadius: 0.05,
  fingerRadius: 0.012,
  fingerCount: 3,
  openAngle: 0.7,
  touchDistance: 0.06,
} as const;

export type Box = {
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly hx: number;
  readonly hy: number;
  readonly hz: number;
};

function span(x0: number, x1: number, y0: number, y1: number, z0: number, z1: number): Box {
  return {
    cx: (x0 + x1) / 2,
    cy: (y0 + y1) / 2,
    cz: (z0 + z1) / 2,
    hx: (x1 - x0) / 2,
    hy: (y1 - y0) / 2,
    hz: (z1 - z0) / 2,
  };
}

const H = CABINET.half;
const W = CABINET.wallThickness;
const B = CABINET.barrierThickness / 2;
const DEPTH = -CABINET.baseDepth;

export const FLOOR_BOXES: readonly Box[] = [
  span(CABINET.chuteX, H, -CABINET.floorThickness, 0, -H, H),
  span(-H, CABINET.chuteX, -CABINET.floorThickness, 0, -H, CABINET.chuteZ),
];

export const BARRIER_BOXES: readonly Box[] = [
  span(CABINET.chuteX - B, CABINET.chuteX + B, 0, CABINET.barrierHeight, CABINET.chuteZ, H),
  span(-H, CABINET.chuteX, 0, CABINET.barrierHeight, CABINET.chuteZ - B, CABINET.chuteZ + B),
];

export const CHUTE_BOXES: readonly Box[] = [
  span(CABINET.chuteX - B, CABINET.chuteX + B, DEPTH, 0, CABINET.chuteZ, H),
  span(-H, CABINET.chuteX, DEPTH, 0, CABINET.chuteZ - B, CABINET.chuteZ + B),
];

export const WALL_BOXES: readonly Box[] = [
  span(-H - W, -H, DEPTH, CABINET.wallHeight, -H, H),
  span(H, H + W, DEPTH, CABINET.wallHeight, -H, H),
  span(-H, H, DEPTH, CABINET.wallHeight, -H - W, -H),
  span(-H, H, DEPTH, CABINET.wallHeight, H, H + W),
  span(-H, H, DEPTH - W, DEPTH, -H, H),
];

type Point = { readonly x: number; readonly y: number };

export type FingerSegment = {
  readonly midX: number;
  readonly midY: number;
  readonly halfLength: number;
  readonly angle: number;
};

const HINGE: Point = { x: 0, y: 0 };
const ELBOW: Point = { x: 0.11, y: -0.14 };
export const FINGER_TIP: Point = { x: -0.03, y: -0.27 };

function segment(a: Point, b: Point): FingerSegment {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return {
    midX: (a.x + b.x) / 2,
    midY: (a.y + b.y) / 2,
    halfLength: Math.hypot(dx, dy) / 2,
    angle: Math.atan2(-dx, dy),
  };
}

export const FINGER_SEGMENTS: readonly FingerSegment[] = [segment(HINGE, ELBOW), segment(ELBOW, FINGER_TIP)];

export function insideChute(x: number, z: number, margin: number): boolean {
  return x < CABINET.chuteX + margin && z > CABINET.chuteZ - margin;
}
