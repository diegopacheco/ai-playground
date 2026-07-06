import fs from 'fs';
import https from 'https';

const code = fs.readFileSync('public/app.js', 'utf8');
const dataMatch = code.match(/const teamsData = (\[[\s\S]*?\]);/);
const teamsData = eval(dataMatch[1]);

const GENERIC_PLAYER = 'https://upload.wikimedia.org/wikipedia/commons/7/7c/Profile_avatar_placeholder_large.png';
const GENERIC_DISH = 'https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg';

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const getSearchImg = async (query) => {
  await delay(150); // rate limiting
  return new Promise(resolve => {
    const url = `https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch=${encodeURIComponent(query)}&gsrlimit=1&prop=pageimages&pithumbsize=500&format=json`;
    https.get(url, { headers: { 'User-Agent': 'FifaTracker/2.0 (admin@example.com)' } }, res => {
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(d);
          if (parsed.query && parsed.query.pages) {
            const pageId = Object.keys(parsed.query.pages)[0];
            const thumb = parsed.query.pages[pageId].thumbnail;
            resolve(thumb ? thumb.source : null);
          } else {
            resolve(null);
          }
        } catch (e) { resolve(null); }
      });
    }).on('error', () => resolve(null));
  });
};

function download(url, dest, fallbackUrl) {
  return new Promise((resolve) => {
    if (!url) {
      download(fallbackUrl, dest, fallbackUrl).then(resolve);
      return;
    }
    const file = fs.createWriteStream(dest);
    https.get(url, { headers: { 'User-Agent': 'FifaTracker/2.0' } }, res => {
      if (res.statusCode !== 200 && url !== fallbackUrl) {
        fs.unlinkSync(dest);
        download(fallbackUrl, dest, fallbackUrl).then(resolve);
        return;
      }
      res.pipe(file);
      file.on('finish', () => resolve());
    }).on('error', () => {
      if (url !== fallbackUrl) download(fallbackUrl, dest, fallbackUrl).then(resolve);
      else resolve();
    });
  });
}

async function run() {
  fs.mkdirSync('public/assets/players', { recursive: true });
  fs.mkdirSync('public/assets/dishes', { recursive: true });
  
  for (const team of teamsData) {
    console.log('Fixing images for', team.name);
    
    // Star player
    const starImg = await getSearchImg(`${team.star} football`);
    await download(starImg, `public/assets/players/${team.id}-star.jpg`, GENERIC_PLAYER);
    
    // Legends
    for (let i = 0; i < team.players.length; i++) {
      const pImg = await getSearchImg(`${team.players[i]} football`);
      await download(pImg, `public/assets/players/${team.id}-legend-${i}.jpg`, GENERIC_PLAYER);
    }

    // Dishes
    const dishes = [`${team.name} national dish food`, `Traditional ${team.name} dessert`, `Popular ${team.name} street food`];
    for (let i = 0; i < 3; i++) {
      const dImg = await getSearchImg(dishes[i]);
      await download(dImg, `public/assets/dishes/${team.id}-dish-${i}.jpg`, GENERIC_DISH);
    }
  }
  console.log('All images fixed!');
}
run();
