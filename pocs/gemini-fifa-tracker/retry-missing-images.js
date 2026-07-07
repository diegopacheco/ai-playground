import fs from 'fs';

const UA = 'FIFA2026TrackerPOC/1.0 (https://github.com/diegopacheco/ai-playground; diego.pacheco.it@gmail.com) node-fetch';
const API = 'https://en.wikipedia.org/w/api.php';

const code = fs.readFileSync('public/app.js', 'utf8');
const teamsData = eval(code.match(/const teamsData = (\[[\s\S]*?\]);/)[1]);
const teamDishes = eval('(' + code.match(/const teamDishes = (\{[\s\S]*?\});/)[1] + ')');
const nonQualifiers = eval(code.match(/const nonQualifiersData = (\[[\s\S]*?\]);/)[1]);
teamsData.push(...nonQualifiers);

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function apiGet(params, attempt = 1) {
  const url = API + '?' + new URLSearchParams({ format: 'json', ...params });
  const res = await fetch(url, { headers: { 'User-Agent': UA } });
  if ((res.status === 429 || res.status === 403 || res.status === 503) && attempt <= 4) {
    await sleep(1000 * attempt);
    return apiGet(params, attempt + 1);
  }
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

async function thumbForTitle(title) {
  const pages = await apiGet({
    action: 'query', titles: title, prop: 'pageimages',
    piprop: 'thumbnail', pithumbsize: 480, redirects: 1
  });
  const page = Object.values(pages?.query?.pages || {})[0];
  return page?.thumbnail?.source || null;
}

async function findThumb(query) {
  const search = await apiGet({
    action: 'query', list: 'search', srsearch: query, srlimit: 3, srnamespace: 0
  });
  for (const hit of search?.query?.search || []) {
    const thumb = await thumbForTitle(hit.title);
    if (thumb) return thumb;
  }
  return null;
}

function validImage(buf) {
  if (buf.length < 2048) return null;
  if (buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return 'jpg';
  if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) return 'png';
  return null;
}

async function download(url, attempt = 1) {
  const res = await fetch(url, { headers: { 'User-Agent': UA } });
  if ((res.status === 429 || res.status === 403 || res.status === 503) && attempt <= 4) {
    await sleep(1000 * attempt);
    return download(url, attempt + 1);
  }
  if (!res.ok) return null;
  return Buffer.from(await res.arrayBuffer());
}

const targets = [];
for (const team of teamsData) {
  targets.push({ key: `players/${team.id}-star`, queries: [team.star, `${team.star} ${team.name} footballer`] });
  team.players.forEach((p, i) =>
    targets.push({ key: `players/${team.id}-legend-${i}`, queries: [p, `${p} ${team.name} footballer`] }));
  (teamDishes[team.id] || []).forEach((d, i) =>
    targets.push({ key: `dishes/${team.id}-dish-${i}`, queries: [d, `${d} ${team.name} cuisine`] }));
}

const manifest = JSON.parse(fs.readFileSync('public/assets/real-images.json', 'utf8'));

let ok = 0, failed = 0;
for (const t of targets) {
  if (manifest[t.key]) continue;
  let done = false;
  for (const q of t.queries) {
    try {
      const thumbUrl = await findThumb(q);
      if (!thumbUrl) continue;
      const buf = await download(thumbUrl);
      const ext = buf && validImage(buf);
      if (!ext) continue;
      const file = `${t.key}.real.${ext}`;
      fs.writeFileSync(`public/assets/${file}`, buf);
      manifest[t.key] = `/assets/${file}`;
      ok++;
      done = true;
      console.log(`OK       ${t.key} via "${q}" <- ${thumbUrl}`);
      break;
    } catch (e) {
      console.log(`ERROR    ${t.key} "${q}": ${e.message}`);
    }
    await sleep(200);
  }
  if (!done) { failed++; console.log(`STILL-MISSING ${t.key}`); }
  fs.writeFileSync('public/assets/real-images.json', JSON.stringify(manifest, null, 2));
  await sleep(150);
}

fs.writeFileSync('public/real-images.js',
  'const REAL_IMAGES = ' + JSON.stringify(manifest, null, 2) + ';\n');
console.log(`\nRetry done. recovered=${ok} stillMissing=${failed}`);
