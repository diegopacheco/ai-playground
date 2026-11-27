import { GAITS } from './presets.js';

function wrap01(v) {
  return ((v % 1) + 1) % 1;
}

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function solveForeLeg(localPhase, duty, scale) {
  const isStance = localPhase < duty;
  let upper = 0;
  let knee = 0;
  let fetlock = 0;
  let hoof = 0;

  if (isStance) {
    const s = localPhase / duty;
    upper = (-0.56 + s * 1.12) * scale;
    knee = 0.05 + Math.sin(s * Math.PI) * 0.08;
    fetlock = -0.28 - Math.sin(s * Math.PI) * 0.52 * scale;
    hoof = -(upper + knee + fetlock) * 0.78;
  } else {
    const f = (localPhase - duty) / (1 - duty);
    const foldEnvelope = Math.sin(Math.pow(f, 0.84) * Math.PI);
    upper = Math.cos(f * Math.PI) * 0.56 * scale - 0.06 * foldEnvelope * scale;
    knee = 0.08 + 1.48 * foldEnvelope * (0.42 + 0.58 * scale);
    fetlock = 0.48 * Math.sin(f * Math.PI) * scale - 0.24 * smoothstep(0.72, 1.0, f);
    hoof = 0.28 * foldEnvelope;
  }

  return {
    upper,
    lower: knee,
    fetlock,
    hoof,
    inStance: isStance,
    groundImpact: isStance && localPhase < 0.065
  };
}

function solveHindLeg(localPhase, duty, scale) {
  const isStance = localPhase < duty;
  let upper = 0;
  let hock = 0;
  let fetlock = 0;
  let hoof = 0;

  if (isStance) {
    const s = localPhase / duty;
    upper = (-0.48 + s * 1.08) * scale;
    hock = -0.24 + s * 0.44 * scale - Math.sin(s * Math.PI) * 0.14;
    fetlock = -0.26 - Math.sin(s * Math.PI) * 0.48 * scale;
    hoof = -(upper + hock + fetlock) * 0.72;
  } else {
    const f = (localPhase - duty) / (1 - duty);
    const gather = Math.sin(Math.pow(f, 0.82) * Math.PI);
    upper = Math.cos(f * Math.PI) * 0.54 * scale + 0.04 * scale;
    hock = -0.14 - 1.05 * gather * (0.42 + 0.58 * scale);
    fetlock = 0.44 * Math.sin(f * Math.PI) * scale - 0.24 * smoothstep(0.74, 1.0, f);
    hoof = 0.22 * gather;
  }

  return {
    upper,
    lower: hock,
    fetlock,
    hoof,
    inStance: isStance,
    groundImpact: isStance && localPhase < 0.065
  };
}

export function evaluateGaitPose(cycle, gaitId = 'gallop') {
  const gait = GAITS[gaitId] || GAITS.gallop;
  const p = wrap01(cycle);
  const rad = p * Math.PI * 2;
  const scale = gait.strideScale;

  const legPhases = gait.phaseOffsets.map((offset) => wrap01(p - offset));
  const legs = [
    solveHindLeg(legPhases[0], gait.stanceDuty[0], scale),
    solveHindLeg(legPhases[1], gait.stanceDuty[1], scale),
    solveForeLeg(legPhases[2], gait.stanceDuty[2], scale),
    solveForeLeg(legPhases[3], gait.stanceDuty[3], scale)
  ];

  const isGallopLike = gaitId === 'gallop' || gaitId === 'canter';
  const bounce = isGallopLike
    ? Math.sin(rad - 0.85) * gait.bounceAmp + Math.sin(rad * 2 - 1.2) * gait.bounceAmp * 0.2
    : Math.abs(Math.sin(rad * 2)) * gait.bounceAmp - gait.bounceAmp * 0.42;

  const bodyPitch = isGallopLike
    ? -Math.cos(rad - 0.45) * gait.pitchAmp
    : Math.sin(rad * 2) * gait.pitchAmp;

  const bodyRoll = isGallopLike
    ? Math.sin(rad) * 0.032
    : Math.sin(rad) * 0.024;

  const spineFlex = Math.sin(rad - 0.9) * gait.spineFlexAmp;
  const neckBasePitch = 0.14 * scale + Math.sin(rad - 0.35) * gait.neckNodAmp - bodyPitch * 0.45;
  const neckMidPitch = -Math.sin(rad - 0.85) * gait.neckNodAmp * 0.52;
  const headPitch = 0.08 * scale + Math.sin(rad - 0.65) * gait.neckNodAmp * 0.68;
  const tailLift = 0.26 + 0.3 * scale + Math.sin(rad - 1.15) * 0.14;
  const tailWave = Math.sin(rad * 2 - 0.6) * (0.14 + 0.12 * scale);
  const ribBreath = 1 + Math.sin(rad) * (0.015 + 0.01 * scale);

  const anyStance = legs.some((l) => l.inStance);

  return {
    gaitId: gait.id,
    cycle: p,
    suspended: !anyStance,
    bodyY: bounce,
    bodyPitch,
    bodyRoll,
    spineFlex,
    neckBasePitch,
    neckMidPitch,
    headPitch,
    tailLift,
    tailWave,
    ribBreath,
    earLeftPitch: Math.sin(rad * 0.5 + 0.4) * 0.09 - 0.04 * scale,
    earRightPitch: Math.cos(rad * 0.5 + 1.1) * 0.09 - 0.04 * scale,
    legs
  };
}
