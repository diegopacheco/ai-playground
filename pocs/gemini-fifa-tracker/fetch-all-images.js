import fs from 'fs';
import https from 'https';

const code = fs.readFileSync('public/app.js', 'utf8');
const dataMatch = code.match(/const teamsData = (\[[\s\S]*?\]);/);
const teamsData = eval(dataMatch[1]);

const getImg = (title) => new Promise(resolve => {
  if (!title) return resolve(null);
  const url = `https://en.wikipedia.org/w/api.php?action=query&titles=${encodeURIComponent(title)}&prop=pageimages&pithumbsize=500&format=json`;
  https.get(url, { headers: { 'User-Agent': 'Bot/1.0' } }, res => {
    let d = '';
    res.on('data', c => d += c);
    res.on('end', () => {
      try {
        const pages = JSON.parse(d).query.pages;
        const pageId = Object.keys(pages)[0];
        resolve(pages[pageId].thumbnail ? pages[pageId].thumbnail.source : null);
      } catch (e) { resolve(null); }
    });
  }).on('error', () => resolve(null));
});

const getSearchImg = (query) => new Promise(resolve => {
  const url = `https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch=${encodeURIComponent(query)}&gsrlimit=1&prop=pageimages&pithumbsize=500&format=json`;
  https.get(url, { headers: { 'User-Agent': 'Bot/1.0' } }, res => {
    let d = '';
    res.on('data', c => d += c);
    res.on('end', () => {
      try {
        const pages = JSON.parse(d).query.pages;
        const pageId = Object.keys(pages)[0];
        resolve(pages[pageId].thumbnail ? pages[pageId].thumbnail.source : null);
      } catch (e) { resolve(null); }
    });
  }).on('error', () => resolve(null));
});

function download(url, dest) {
  return new Promise((resolve) => {
    if (!url) {
      fs.copyFileSync('public/assets/players/canada-star.jpg', dest); // Fallback
      return resolve();
    }
    const file = fs.createWriteStream(dest);
    https.get(url, { headers: { 'User-Agent': 'Bot/1.0' } }, res => {
      if (res.statusCode !== 200) {
        fs.copyFileSync('public/assets/players/canada-star.jpg', dest);
        return resolve();
      }
      res.pipe(file);
      file.on('finish', () => resolve());
    }).on('error', () => {
      fs.copyFileSync('public/assets/players/canada-star.jpg', dest);
      resolve();
    });
  });
}

async function run() {
  fs.mkdirSync('public/assets/players', { recursive: true });
  fs.mkdirSync('public/assets/dishes', { recursive: true });
  for (const team of teamsData) {
    console.log('Fetching', team.id);
    let starImg = await getImg(team.star);
    if (!starImg) starImg = await getSearchImg(team.star + ' football');
    await download(starImg, `public/assets/players/${team.id}-star.jpg`);
    
    for (let i = 0; i < team.players.length; i++) {
      let pImg = await getImg(team.players[i]);
      if (!pImg) pImg = await getSearchImg(team.players[i] + ' football');
      await download(pImg, `public/assets/players/${team.id}-legend-${i}.jpg`);
    }

    // Dishes
    const dishes = [`${team.name} national dish`, `Traditional ${team.name} food`, `Popular ${team.name} cuisine`];
    for (let i = 0; i < 3; i++) {
      const dImg = await getSearchImg(dishes[i]);
      await download(dImg, `public/assets/dishes/${team.id}-dish-${i}.jpg`);
    }
  }
  console.log('Done all');
}
run();
