export const spots = [
  { id: 'embarcadero', name: 'Embarcadero', title: 'The Embarcadero', caption: 'Wide open. All yours.', detail: 'BAY BREEZE. SMOOTH CONCRETE.', type: 'Waterfront cruise', color: '#dfe9d5', speed: 285, icon: 'M3 27h32M6 24V11m24 13V11M6 12q12 15 24 0M6 16h24M3 10h6m18 0h6' },
  { id: 'wharf', name: 'Fisherman’s Wharf', title: 'Fisherman’s Wharf', caption: 'Salt air. Fresh lines.', detail: 'OLD PIERS. NEW POSSIBILITIES.', type: 'Harborside hustle', color: '#eee3cd', speed: 305, icon: 'M3 27h31M7 26V14h23v12M5 14l13-9 14 9M12 26v-7h5v7m6-9v5M18 5V1' },
  { id: 'ferry', name: 'Ferry Building', title: 'The Ferry Building', caption: 'Meet you by the clock.', detail: 'TIME FLIES. SO CAN YOU.', type: 'Clocktower classics', color: '#eadfd1', speed: 295, icon: 'M2 28h32M6 28V18h24v10M14 18V6h9v12M12 6h13l-6-5zM16 11h5m-3-2v4' },
  { id: 'pier39', name: 'Pier 39', title: 'Pier 39', caption: 'Good times on board.', detail: 'SEA LIONS. SKY HIGH LINES.', type: 'Boardwalk flow', color: '#dce9df', speed: 320, icon: 'M2 28h33M7 26V8h23v18M4 8h29M10 8V4h17v4M11 15h15M11 20h15M15 26v-3h8v3' }
];
export const trickPoints = { kickflip: 150, heelflip: 200, spin: 250 };
export const trickNames = { kickflip: 'KICKFLIP', heelflip: 'HEELFLIP', spin: '180 SPIN' };
export function createGame(spot = spots[0], rider = 'boy', random = Math.random) {
  return { spot, rider, random, status: 'ready', time: 60, score: 0, distance: 0, y: 0, vy: 0, tricks: [], trick: '', trickAge: 0, obstacles: [], nextObstacle: 900, invincible: 0, message: '', messageTime: 0, landed: 0, bails: 0 };
}
export function startGame(game) {
  if (game.status === 'ready') game.status = 'playing';
}
export function jump(game) {
  if (game.status !== 'playing' || game.y > 0) return false;
  game.vy = 660;
  game.y = 0.1;
  game.tricks = [];
  return true;
}
export function performTrick(game, trick) {
  if (game.status !== 'playing' || game.y < 30 || !(trick in trickPoints) || game.tricks.includes(trick)) return false;
  game.tricks.push(trick);
  game.trick = trick;
  game.trickAge = 0;
  game.message = game.tricks.map(value => trickNames[value]).join(' + ');
  game.messageTime = 1.4;
  return true;
}
export function togglePause(game) {
  if (game.status === 'playing') game.status = 'paused';
  else if (game.status === 'paused') game.status = 'playing';
}
export function updateGame(game, delta) {
  if (game.status !== 'playing') return;
  const dt = Math.max(0, Math.min(delta, 0.05));
  game.time = Math.max(0, game.time - dt);
  game.distance += game.spot.speed * dt;
  game.invincible = Math.max(0, game.invincible - dt);
  game.messageTime = Math.max(0, game.messageTime - dt);
  game.trickAge += dt;
  if (game.y > 0) {
    game.vy -= 1450 * dt;
    game.y = Math.max(0, game.y + game.vy * dt);
    if (game.y === 0) {
      game.vy = 0;
      const points = (50 + game.tricks.reduce((sum, trick) => sum + trickPoints[trick], 0)) * Math.max(1, game.tricks.length);
      game.score += points;
      game.landed++;
      game.message = `${game.tricks.length > 1 ? `${game.tricks.length}× COMBO` : game.tricks.length ? trickNames[game.tricks[0]] : 'CLEAN OLLIE'}  +${points}`;
      game.messageTime = 1.2;
      game.tricks = [];
      game.trick = '';
    }
  }
  if (game.distance >= game.nextObstacle) {
    const type = ['cone', 'bench', 'planter'][Math.floor(game.random() * 3)];
    game.obstacles.push({ x: 1280, width: type === 'bench' ? 100 : type === 'planter' ? 65 : 35, height: type === 'cone' ? 43 : 48, type, passed: false });
    game.nextObstacle = game.distance + 410 + game.random() * 340;
  }
  for (const obstacle of game.obstacles) {
    obstacle.x -= game.spot.speed * dt;
    if (!obstacle.passed && obstacle.x < 305 && obstacle.x + obstacle.width > 265 && game.y < obstacle.height && game.invincible === 0) {
      game.score = Math.max(0, game.score - 100);
      game.tricks = [];
      game.trick = '';
      game.message = 'SHAKE IT OFF. KEEP ROLLING.';
      game.messageTime = 1.5;
      game.invincible = 1.3;
      game.bails++;
      obstacle.passed = true;
    } else if (!obstacle.passed && obstacle.x + obstacle.width < 265) {
      game.score += 75;
      obstacle.passed = true;
    }
  }
  game.obstacles = game.obstacles.filter(obstacle => obstacle.x > -120);
  if (game.time === 0) game.status = 'finished';
}
