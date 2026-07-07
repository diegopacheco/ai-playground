import fs from 'fs';

const UA = 'FIFA2026TrackerPOC/1.0 (https://github.com/diegopacheco/ai-playground; diego.pacheco.it@gmail.com) node-fetch';
const API = 'https://en.wikipedia.org/w/api.php';

const suspects = {
  'players/canada-legend-1': ['Craig Forrest'],
  'dishes/germany-dish-1': ['Sauerkraut'],
  'players/jordan-legend-2': ['Odai Al-Saify'],
  'players/qatar-legend-0': ['Mansour Muftah'],
  'players/saudi-arabia-legend-2': ['Saeed Al-Owairan'],
  'players/caboverde-legend-2': ['Babanco'],
  'dishes/caboverde-dish-1': ['Pastel (food)'],
  'players/ghana-legend-2': ['Tony Yeboah'],
  'players/senegal-legend-2': ['Henri Camara'],
  'dishes/curacao-dish-2': ['Sopito'],
  'players/haiti-legend-0': ['Emmanuel Sanon']
};

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function apiGet(params) {
  const url = API + '?' + new URLSearchParams({ format: 'json', ...params });
  const res = await fetch(url, { headers: { 'User-Agent': UA } });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

const norm = s => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');

async function thumbForTitle(title) {
  const pages = await apiGet({
    action: 'query', titles: title, prop: 'pageimages',
    piprop: 'thumbnail', pithumbsize: 480, redirects: 1
  });
  const page = Object.values(pages?.query?.pages || {})[0];
  if (page?.thumbnail?.source) return page.thumbnail.source;

  const imgList = await apiGet({
    action: 'query', titles: title, prop: 'images', imlimit: 50, redirects: 1
  });
  const files = Object.values(imgList?.query?.pages || {})[0]?.images || [];
  const tokens = norm(title).replace(/\(.*\)/, '').split(/[\s-]+/).filter(t => t.length > 3);
  const match = files.find(f =>
    /\.(jpe?g|png)$/i.test(f.title) && tokens.some(t => norm(f.title).includes(t)));
  if (!match) return null;

  const info = await apiGet({
    action: 'query', titles: match.title, prop: 'imageinfo',
    iiprop: 'url', iiurlwidth: 480
  });
  const infoPage = Object.values(info?.query?.pages || {})[0];
  return infoPage?.imageinfo?.[0]?.thumburl || null;
}

function validImage(buf) {
  if (buf.length < 2048) return null;
  if (buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return 'jpg';
  if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) return 'png';
  return null;
}

const manifest = JSON.parse(fs.readFileSync('public/assets/real-images.json', 'utf8'));

for (const [key, titles] of Object.entries(suspects)) {
  if (manifest[key]) {
    const old = 'public' + manifest[key];
    if (fs.existsSync(old)) fs.unlinkSync(old);
    delete manifest[key];
  }
  let fixed = false;
  for (const title of titles) {
    const thumbUrl = await thumbForTitle(title);
    if (!thumbUrl || /logo|flag_of|emblem|crest|coat_of_arms/i.test(thumbUrl)) continue;
    const res = await fetch(thumbUrl, { headers: { 'User-Agent': UA } });
    if (!res.ok) continue;
    const buf = Buffer.from(await res.arrayBuffer());
    const ext = validImage(buf);
    if (!ext) continue;
    const file = `${key}.real.${ext}`;
    fs.writeFileSync(`public/assets/${file}`, buf);
    manifest[key] = `/assets/${file}`;
    console.log(`FIXED    ${key} <- ${thumbUrl}`);
    fixed = true;
    break;
  }
  if (!fixed) console.log(`SVG-FALLBACK ${key}`);
  await sleep(200);
}

fs.writeFileSync('public/assets/real-images.json', JSON.stringify(manifest, null, 2));
fs.writeFileSync('public/real-images.js',
  'const REAL_IMAGES = ' + JSON.stringify(manifest, null, 2) + ';\n');
console.log(`\nManifest entries: ${Object.keys(manifest).length}`);
