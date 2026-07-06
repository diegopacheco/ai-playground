import fs from 'fs';

const code = fs.readFileSync('public/app.js', 'utf8');
const dataMatch = code.match(/const teamsData = (\[[\s\S]*?\]);/);
const teamsData = eval(dataMatch[1]);

let missing = [];

for (const team of teamsData) {
  if (!fs.existsSync(`public/assets/players/${team.id}-star.svg`)) {
    missing.push(`players/${team.id}-star.svg`);
  }
  for (let i = 0; i < team.players.length; i++) {
    if (!fs.existsSync(`public/assets/players/${team.id}-legend-${i}.svg`)) {
      missing.push(`players/${team.id}-legend-${i}.svg`);
    }
  }
  for (let i = 0; i < 3; i++) {
    if (!fs.existsSync(`public/assets/dishes/${team.id}-dish-${i}.svg`)) {
      missing.push(`dishes/${team.id}-dish-${i}.svg`);
    }
  }
}

if (missing.length > 0) {
  console.log("Missing SVGs:", missing.length);
  console.log(missing.join('\n'));
} else {
  console.log("No missing SVGs!");
}
