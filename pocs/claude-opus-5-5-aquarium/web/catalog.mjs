export const TANK = { width: 1.2, height: 0.62, depth: 0.5, water: 0.58 };

export const MAX_FISH = 40;

export const MAX_GRASS = 8;

export const MAX_FOOD = 24;

export const RACK_COLORS = [
  { id: 'natural', name: 'Natural', hex: null },
  { id: 'midnight', name: 'Midnight', hex: '#1c1d21' },
  { id: 'snow', name: 'Snow', hex: '#ece9e1' },
  { id: 'ocean', name: 'Ocean', hex: '#1f4e79' },
  { id: 'forest', name: 'Forest', hex: '#2f5d3a' },
  { id: 'cherry', name: 'Cherry', hex: '#8e2230' },
  { id: 'walnut', name: 'Walnut', hex: '#5a3a22' },
  { id: 'sand', name: 'Sand', hex: '#cdb891' },
  { id: 'slate', name: 'Slate', hex: '#5b6470' }
];

export const MATERIALS = [
  { id: 'wood', name: 'Wood', natural: '#9a6a3f', pattern: 'wood', roughness: 0.65, metalness: 0, bump: 0.6 },
  { id: 'steel', name: 'Steel', natural: '#b9bec4', pattern: 'brushed', roughness: 0.35, metalness: 0.9, bump: 0.15 },
  { id: 'bricks', name: 'Bricks', natural: '#a4553a', pattern: 'bricks', roughness: 0.9, metalness: 0, bump: 2.5 },
  { id: 'marble', name: 'Marble', natural: '#e8e4dc', pattern: 'marble', roughness: 0.18, metalness: 0, bump: 0.1 },
  { id: 'concrete', name: 'Concrete', natural: '#9d9a94', pattern: 'concrete', roughness: 0.95, metalness: 0, bump: 1.2 },
  { id: 'bamboo', name: 'Bamboo', natural: '#c9a55c', pattern: 'bamboo', roughness: 0.55, metalness: 0, bump: 1.5 },
  { id: 'stone', name: 'Stone', natural: '#8a8478', pattern: 'stone', roughness: 0.85, metalness: 0, bump: 2.2 },
  { id: 'carbon', name: 'Carbon', natural: '#3a3d42', pattern: 'carbon', roughness: 0.3, metalness: 0.4, bump: 0.8 },
  { id: 'brass', name: 'Brass', natural: '#c8a049', pattern: 'brushed', roughness: 0.3, metalness: 1, bump: 0.15 },
  { id: 'leather', name: 'Leather', natural: '#6b3b24', pattern: 'leather', roughness: 0.75, metalness: 0, bump: 1 }
];

export const FISH = [
  { id: 'neon', name: 'Neon Tetra', len: 0.05, height: 0.32, width: 0.22, pattern: 'neon', colors: ['#c7d2dc', '#18c6ff', '#ec2440'], fin: '#dde8f0', tail: 'fork', dorsal: 'small', speed: 0.13, band: [0.35, 0.85] },
  { id: 'clown', name: 'Clownfish', len: 0.075, height: 0.45, width: 0.26, pattern: 'clown', colors: ['#ff7417', '#ffffff', '#141414'], fin: '#ff7417', tail: 'round', dorsal: 'round', speed: 0.08, band: [0.15, 0.6] },
  { id: 'bluetang', name: 'Blue Tang', len: 0.1, height: 0.62, width: 0.2, pattern: 'tang', colors: ['#1f55e0', '#0a1030', '#ffd400'], fin: '#1f55e0', tail: 'fork', dorsal: 'long', speed: 0.1, band: [0.3, 0.8] },
  { id: 'goldfish', name: 'Goldfish', len: 0.09, height: 0.52, width: 0.32, pattern: 'belly', colors: ['#ff8a12', '#ffc070'], fin: '#ffa040', tail: 'fan', dorsal: 'round', speed: 0.07, band: [0.2, 0.7] },
  { id: 'angel', name: 'Angelfish', len: 0.085, height: 0.95, width: 0.16, pattern: 'bands', colors: ['#e9e5d6', '#262626'], fin: '#e9e5d6', tail: 'fork', dorsal: 'angel', speed: 0.06, band: [0.4, 0.85] },
  { id: 'betta', name: 'Betta', len: 0.065, height: 0.4, width: 0.24, pattern: 'fade', colors: ['#b3103c', '#4b12d4'], fin: '#7a1ad0', tail: 'veil', dorsal: 'long', speed: 0.05, band: [0.55, 0.9] },
  { id: 'guppy', name: 'Guppy', len: 0.045, height: 0.34, width: 0.22, pattern: 'guppy', colors: ['#a6b1bb', '#ff5fa2', '#35d0ff'], fin: '#ff7fb6', tail: 'fan', dorsal: 'small', speed: 0.11, band: [0.5, 0.9] },
  { id: 'discus', name: 'Discus', len: 0.085, height: 0.92, width: 0.18, pattern: 'wavy', colors: ['#e2552c', '#2fa9ca'], fin: '#e2552c', tail: 'round', dorsal: 'angel', speed: 0.05, band: [0.3, 0.75] },
  { id: 'mandarin', name: 'Mandarin', len: 0.055, height: 0.42, width: 0.3, pattern: 'psy', colors: ['#ff7b1c', '#1a7fd6'], fin: '#2f8fe0', tail: 'round', dorsal: 'round', speed: 0.05, band: [0.08, 0.3] },
  { id: 'yellowtang', name: 'Yellow Tang', len: 0.09, height: 0.78, width: 0.18, pattern: 'solid', colors: ['#ffe21a'], fin: '#ffe21a', tail: 'fork', dorsal: 'long', speed: 0.09, band: [0.3, 0.8] }
];

export const DECOR = [
  { id: 'ship', name: 'Sunken Ship', spots: [[-0.33, -0.1, 0.13]] },
  { id: 'car', name: 'Sunken Car', spots: [[0.37, 0.14, 0.09]] },
  { id: 'plane', name: 'Sunken Plane', spots: [[0.3, -0.12, 0.1]] },
  { id: 'chest', name: 'Treasure Chest', spots: [[-0.44, 0.15, 0.05]] },
  { id: 'castle', name: 'Castle', spots: [[-0.01, -0.13, 0.08]] },
  { id: 'coral', name: 'Coral Reef', spots: [[-0.2, 0.14, 0.1]] },
  { id: 'rocks', name: 'Rocks', spots: [[-0.53, -0.17, 0.04], [0.53, -0.01, 0.035], [0.17, 0.17, 0.03]] },
  { id: 'anchor', name: 'Anchor', spots: [[0.03, 0.15, 0.05]] },
  { id: 'helmet', name: 'Diver Helmet', spots: [[-0.09, 0.02, 0.04]] }
];

export const SUBSTRATES = [
  { id: 'sand', name: 'Sand' },
  { id: 'soil', name: 'Aqua soil' },
  { id: 'path', name: 'Soil + sand path' }
];

export const SCAPE = [
  { id: 'carpet', name: 'Carpet' },
  { id: 'bushes', name: 'Bushes' },
  { id: 'stones', name: 'Dragon stones' }
];

export const STONES = [
  [0.13, 0.0, 0.055, 0.2],
  [0.1, -0.2, 0.04, 0.15],
  [0.46, -0.13, 0.05, 0.17],
  [-0.13, -0.2, 0.045, 0.13]
];

export const BUSHES = [
  { x: -0.47, z: -0.19, rx: 0.1, rz: 0.045, h: 0.32, kind: 'red' },
  { x: -0.29, z: -0.2, rx: 0.09, rz: 0.04, h: 0.26, kind: 'green' },
  { x: -0.11, z: -0.2, rx: 0.08, rz: 0.04, h: 0.2, kind: 'lime' },
  { x: 0.08, z: -0.2, rx: 0.09, rz: 0.04, h: 0.29, kind: 'red' },
  { x: 0.27, z: -0.2, rx: 0.1, rz: 0.04, h: 0.31, kind: 'green' },
  { x: 0.43, z: -0.2, rx: 0.06, rz: 0.04, h: 0.23, kind: 'lime' }
];

export function byId(list, id) {
  return list.find(item => item.id === id) || null;
}
