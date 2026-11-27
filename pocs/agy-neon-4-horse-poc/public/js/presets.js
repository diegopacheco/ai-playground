export const COATS = {
  bay: {
    id: 'bay',
    label: 'Chestnut Bay',
    bodyPrimary: '#7a3c1b',
    bodyShadow: '#49200b',
    bodyHighlight: '#ad5e2e',
    points: '#1f1510',
    mane: '#17100c',
    hoof: '#26211d',
    muzzle: '#2b221d',
    blaze: true,
    socks: [true, false, true, true],
    dapple: 0.08,
    sheen: 0.45
  },
  palomino: {
    id: 'palomino',
    label: 'Palomino Gold',
    bodyPrimary: '#d19b48',
    bodyShadow: '#9a6624',
    bodyHighlight: '#f0c878',
    points: '#a87835',
    mane: '#f5ecd8',
    hoof: '#4b4136',
    muzzle: '#6e5c4a',
    blaze: true,
    socks: [true, true, true, true],
    dapple: 0.14,
    sheen: 0.55
  },
  black: {
    id: 'black',
    label: 'Midnight Black',
    bodyPrimary: '#1e1d22',
    bodyShadow: '#0d0c0f',
    bodyHighlight: '#3d3c47',
    points: '#100f12',
    mane: '#0a0a0d',
    hoof: '#201e1d',
    muzzle: '#212026',
    blaze: false,
    socks: [false, false, true, false],
    dapple: 0.05,
    sheen: 0.62
  },
  dapple: {
    id: 'dapple',
    label: 'Dapple Grey',
    bodyPrimary: '#c0c3c8',
    bodyShadow: '#777c84',
    bodyHighlight: '#e6e9ee',
    points: '#52565d',
    mane: '#dbdde2',
    hoof: '#3c3a38',
    muzzle: '#4d5056',
    blaze: false,
    socks: [false, false, false, false],
    dapple: 0.48,
    sheen: 0.4
  }
};

export const GAITS = {
  gallop: {
    id: 'gallop',
    label: 'Gallop',
    strideHz: 2.35,
    speedMps: 14.5,
    speedMph: 32,
    phaseOffsets: [0.0, 0.2, 0.54, 0.72],
    stanceDuty: [0.27, 0.27, 0.25, 0.25],
    bounceAmp: 0.14,
    pitchAmp: 0.105,
    spineFlexAmp: 0.15,
    neckNodAmp: 0.14,
    strideScale: 1.0,
    dustIntensity: 1.0
  },
  canter: {
    id: 'canter',
    label: 'Canter',
    strideHz: 1.72,
    speedMps: 9.2,
    speedMph: 21,
    phaseOffsets: [0.0, 0.3, 0.34, 0.66],
    stanceDuty: [0.35, 0.34, 0.34, 0.34],
    bounceAmp: 0.105,
    pitchAmp: 0.08,
    spineFlexAmp: 0.1,
    neckNodAmp: 0.1,
    strideScale: 0.78,
    dustIntensity: 0.62
  },
  trot: {
    id: 'trot',
    label: 'Trot',
    strideHz: 1.48,
    speedMps: 5.4,
    speedMph: 12,
    phaseOffsets: [0.0, 0.5, 0.5, 0.0],
    stanceDuty: [0.46, 0.46, 0.46, 0.46],
    bounceAmp: 0.07,
    pitchAmp: 0.025,
    spineFlexAmp: 0.04,
    neckNodAmp: 0.045,
    strideScale: 0.6,
    dustIntensity: 0.28
  },
  walk: {
    id: 'walk',
    label: 'Walk',
    strideHz: 0.92,
    speedMps: 2.2,
    speedMph: 5,
    phaseOffsets: [0.0, 0.5, 0.75, 0.25],
    stanceDuty: [0.64, 0.64, 0.64, 0.64],
    bounceAmp: 0.025,
    pitchAmp: 0.015,
    spineFlexAmp: 0.02,
    neckNodAmp: 0.055,
    strideScale: 0.42,
    dustIntensity: 0.05
  }
};

export const LIGHTING_MOODS = {
  morning: {
    id: 'morning',
    label: 'Alpine Morning',
    sunElevation: 36,
    sunAzimuth: 142,
    sunColor: '#fff6e3',
    sunIntensity: 3.4,
    hemiSky: '#8cc9fa',
    hemiGround: '#5f7d40',
    hemiIntensity: 1.35,
    skyZenith: '#1a68cc',
    skyHorizon: '#cfe9ff',
    skyGlow: '#fff1cc',
    fogColor: '#c7e3fa',
    fogDensity: 0.00065,
    exposure: 1.07
  },
  noon: {
    id: 'noon',
    label: 'Colorado Noon',
    sunElevation: 60,
    sunAzimuth: 158,
    sunColor: '#fffdf7',
    sunIntensity: 3.65,
    hemiSky: '#7ec4fa',
    hemiGround: '#658246',
    hemiIntensity: 1.4,
    skyZenith: '#145ec4',
    skyHorizon: '#c5e6ff',
    skyGlow: '#fff9e6',
    fogColor: '#c2e2fa',
    fogDensity: 0.00055,
    exposure: 1.04
  },
  golden: {
    id: 'golden',
    label: 'Golden Hour',
    sunElevation: 18,
    sunAzimuth: 118,
    sunColor: '#ffc47d',
    sunIntensity: 3.55,
    hemiSky: '#a2bce8',
    hemiGround: '#6e5932',
    hemiIntensity: 1.12,
    skyZenith: '#22529c',
    skyHorizon: '#f9cf9e',
    skyGlow: '#ffad60',
    fogColor: '#ecc498',
    fogDensity: 0.0009,
    exposure: 1.08
  }
};

export const CAMERAS = {
  hero: {
    id: 'hero',
    label: 'Front 3/4',
    offset: [4.1, 1.55, 4.15],
    lookAt: [0.0, 1.05, 0.15],
    fov: 38
  },
  side: {
    id: 'side',
    label: 'Track Side',
    offset: [5.3, 1.65, 0.65],
    lookAt: [0.0, 1.08, 0.15],
    fov: 37
  },
  low: {
    id: 'low',
    label: 'Low Hero',
    offset: [-3.85, 0.92, 3.65],
    lookAt: [0.0, 1.18, 0.0],
    fov: 39
  },
  chase: {
    id: 'chase',
    label: 'Valley Chase',
    offset: [-1.55, 2.25, -5.6],
    lookAt: [0.0, 1.65, 14.0],
    fov: 45
  },
  orbit: {
    id: 'orbit',
    label: 'Free Orbit',
    offset: [4.6, 1.9, 3.6],
    lookAt: [0.0, 1.1, 0.0],
    fov: 40
  }
};

export const PRESET_SUMMARY = {
  location: 'Maroon Creek Valley, Colorado',
  elevationFeet: 9420,
  season: 'July Summer',
  coats: Object.values(COATS).map(({ id, label }) => ({ id, label })),
  gaits: Object.values(GAITS).map(({ id, label, speedMph }) => ({ id, label, speedMph })),
  lighting: Object.values(LIGHTING_MOODS).map(({ id, label }) => ({ id, label })),
  cameras: Object.values(CAMERAS).map(({ id, label }) => ({ id, label }))
};
