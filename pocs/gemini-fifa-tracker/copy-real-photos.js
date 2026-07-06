import fs from 'fs';
import path from 'path';

const teams = [
  { id: 'canada' }, { id: 'mexico' }, { id: 'usa' }, { id: 'austria' }, { id: 'belgium' },
  { id: 'bosnia' }, { id: 'croatia' }, { id: 'czechia' }, { id: 'england' }, { id: 'france' },
  { id: 'germany' }, { id: 'netherlands' }, { id: 'norway' }, { id: 'portugal' }, { id: 'scotland' },
  { id: 'spain' }, { id: 'sweden' }, { id: 'switzerland' }, { id: 'turkiye' }, { id: 'argentina' },
  { id: 'brazil' }, { id: 'colombia' }, { id: 'ecuador' }, { id: 'paraguay' }, { id: 'uruguay' },
  { id: 'australia' }, { id: 'iran' }, { id: 'iraq' }, { id: 'japan' }, { id: 'jordan' },
  { id: 'qatar' }, { id: 'saudi-arabia' }, { id: 'south-korea' }, { id: 'uzbekistan' }, { id: 'algeria' },
  { id: 'caboverde' }, { id: 'cote-divoire' }, { id: 'dr-congo' }, { id: 'egypt' }, { id: 'ghana' },
  { id: 'morocco' }, { id: 'senegal' }, { id: 'south-africa' }, { id: 'tunisia' }, { id: 'curacao' },
  { id: 'haiti' }, { id: 'panama' }, { id: 'new-zealand' }
];

const sourceDir = '/Users/diegopacheco/.gemini/antigravity-cli/brain/5a83dcc9-8251-4464-a0d9-ebc5bb89f955';
const publicDir = path.join(process.cwd(), 'public');
const playersDir = path.join(publicDir, 'assets', 'players');
const dishesDir = path.join(publicDir, 'assets', 'dishes');

if (!fs.existsSync(playersDir)) fs.mkdirSync(playersDir, { recursive: true });
if (!fs.existsSync(dishesDir)) fs.mkdirSync(dishesDir, { recursive: true });

const starSource = path.join(sourceDir, 'soccer_player_star_1783379056516.jpg');
const legendSource = path.join(sourceDir, 'soccer_legend_1783379099772.jpg');
const tacosSource = path.join(sourceDir, 'gourmet_tacos_1783379067889.jpg');
const poutineSource = path.join(sourceDir, 'gourmet_poutine_1783379113749.jpg');
const paellaSource = path.join(sourceDir, 'gourmet_paella_1783379127932.jpg');

for (const team of teams) {
  fs.copyFileSync(starSource, path.join(playersDir, `${team.id}-star.jpg`));
  
  for (let i = 0; i < 3; i++) {
    fs.copyFileSync(legendSource, path.join(playersDir, `${team.id}-legend-${i}.jpg`));
  }
  
  fs.copyFileSync(tacosSource, path.join(dishesDir, `${team.id}-dish-0.jpg`));
  fs.copyFileSync(poutineSource, path.join(dishesDir, `${team.id}-dish-1.jpg`));
  fs.copyFileSync(paellaSource, path.join(dishesDir, `${team.id}-dish-2.jpg`));
}
