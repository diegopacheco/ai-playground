import fs from 'fs';
import path from 'path';
import https from 'https';

const PLAYER_PLACEHOLDER = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Silueta.jpg/500px-Silueta.jpg';
const DISH_PLACEHOLDER = 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/500px-Good_Food_Display_-_NCI_Visuals_Online.jpg';

function download(url, dest) {
  return new Promise((resolve) => {
    const file = fs.createWriteStream(dest);
    https.get(url, { headers: { 'User-Agent': 'Bot' } }, res => {
      res.pipe(file);
      file.on('finish', () => resolve());
    });
  });
}

async function run() {
  await download(PLAYER_PLACEHOLDER, 'valid-player.jpg');
  await download(DISH_PLACEHOLDER, 'valid-dish.jpg');

  const pDir = 'public/assets/players';
  for (const f of fs.readdirSync(pDir)) {
    if (!f.endsWith('.jpg')) continue;
    const stat = fs.statSync(path.join(pDir, f));
    // If it's the 1989/2011 byte PNG disguised as JPG
    if (stat.size < 3000) {
      fs.copyFileSync('valid-player.jpg', path.join(pDir, f));
    }
  }

  const dDir = 'public/assets/dishes';
  if (fs.existsSync(dDir)) {
    for (const f of fs.readdirSync(dDir)) {
      if (!f.endsWith('.jpg')) continue;
      const stat = fs.statSync(path.join(dDir, f));
      // If it's the raw SVG XML disguised as JPG
      if (stat.size < 3000) {
        fs.copyFileSync('valid-dish.jpg', path.join(dDir, f));
      }
    }
  }
  console.log('Fixed corrupted images!');
}
run();
