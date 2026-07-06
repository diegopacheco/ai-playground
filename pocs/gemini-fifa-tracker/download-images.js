import fs from 'fs';
import path from 'path';
import https from 'https';

const images = JSON.parse(fs.readFileSync('real-images.json', 'utf8'));
const playersDir = path.join('public', 'assets', 'players');
const dishesDir = path.join('public', 'assets', 'dishes');

if (!fs.existsSync(playersDir)) fs.mkdirSync(playersDir, { recursive: true });
if (!fs.existsSync(dishesDir)) fs.mkdirSync(dishesDir, { recursive: true });

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    if (!url) {
      console.log(`No URL for ${dest}, skipping.`);
      return resolve();
    }
    const file = fs.createWriteStream(dest);
    https.get(url, { headers: { 'User-Agent': 'FifaTrackerBot/1.0' } }, (res) => {
      if (res.statusCode !== 200) {
        return reject(new Error(`Status ${res.statusCode}`));
      }
      res.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(dest, () => {});
      reject(err);
    });
  });
}

async function run() {
  for (const [team, data] of Object.entries(images)) {
    console.log(`Downloading for ${team}...`);
    try {
      if (data.playerImg) await downloadFile(data.playerImg, path.join(playersDir, `${team}-star.jpg`));
      if (data.dishImg) await downloadFile(data.dishImg, path.join(dishesDir, `${team}-dish.jpg`));
    } catch (e) {
      console.log(`Error downloading for ${team}: ${e.message}`);
    }
  }
  console.log('Done downloading.');
}
run();
