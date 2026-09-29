import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { readCompanies, LOGO_DIR, USER_AGENT } from "./dataset.mjs";

const GOOD_SCORE = 96;
const BROWSER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36";
const SOURCES = [
  domain => `https://${domain}/apple-touch-icon.png`,
  domain => `https://www.${domain}/apple-touch-icon.png`,
  domain => `https://www.google.com/s2/favicons?domain=${domain}&sz=256`,
  domain => `https://icons.duckduckgo.com/ip3/${domain}.ico`
];

function imageSize(bytes) {
  if (bytes.length < 24) return null;
  const word = at => (bytes[at] << 24) | (bytes[at + 1] << 16) | (bytes[at + 2] << 8) | bytes[at + 3];
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e) return { width: word(16), height: word(20) };
  if (bytes[0] === 0x00 && bytes[1] === 0x00 && bytes[2] === 0x01) return { width: bytes[6] || 256, height: bytes[7] || 256 };
  return null;
}

function score(size) {
  if (!size) return 0;
  const square = Math.max(size.width, size.height) / Math.min(size.width, size.height) < 1.4;
  return square ? Math.min(size.width, 512) : Math.min(size.width, size.height) / 4;
}

async function fetchImage(url) {
  try {
    const res = await fetch(url, { headers: { "User-Agent": USER_AGENT }, signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    const bytes = new Uint8Array(await res.arrayBuffer());
    const value = score(imageSize(bytes));
    return value > 0 ? { bytes, score: value } : null;
  } catch {
    return null;
  }
}

async function declaredIcons(domain) {
  try {
    const page = `https://www.${domain}/`;
    const res = await fetch(page, { headers: { "User-Agent": BROWSER_AGENT }, signal: AbortSignal.timeout(8000) });
    const html = await res.text();
    return [...html.matchAll(/<link\b[^>]*>/gi)]
      .map(([tag]) => ({ tag, href: tag.match(/href=["']([^"']+)["']/i)?.[1] }))
      .filter(({ tag, href }) => href && /rel=["'][^"']*icon/i.test(tag) && !/\.svg(\?|$)/i.test(href))
      .map(({ tag, href }) => ({ url: new URL(href, res.url || page).href, size: Number(tag.match(/sizes=["'](\d+)/i)?.[1] || (/apple/i.test(tag) ? 180 : 32)) }))
      .sort((a, b) => b.size - a.size)
      .map(icon => icon.url);
  } catch {
    return [];
  }
}

async function wikidata(params) {
  const res = await fetch(`https://www.wikidata.org/w/api.php?${new URLSearchParams({ format: "json", ...params })}`, { headers: { "User-Agent": USER_AGENT }, signal: AbortSignal.timeout(8000) });
  return res.json();
}

async function wikidataLogo(company) {
  try {
    const found = await wikidata({ action: "wbsearchentities", search: company.name.replace(/\s*\(.*\)/, ""), language: "en", limit: "5" });
    const ids = found.search.map(hit => hit.id).join("|");
    if (!ids) return [];
    const { entities } = await wikidata({ action: "wbgetentities", ids, props: "claims" });
    const root = rootDomain(company.domain);
    for (const entity of Object.values(entities)) {
      const sites = (entity.claims?.P856 || []).map(c => c.mainsnak.datavalue?.value || "");
      const logo = entity.claims?.P154?.[0]?.mainsnak.datavalue?.value;
      if (logo && sites.some(site => site.includes(root))) {
        return [`https://commons.wikimedia.org/wiki/Special:FilePath/${encodeURIComponent(logo)}?width=160`];
      }
    }
  } catch {
    return [];
  }
  return [];
}

const SIMPLE_ICONS = "https://cdn.jsdelivr.net/npm/simple-icons@latest";
let simpleIndex = null;

function rootDomain(domain) {
  return domain.split(".").slice(-2).join(".");
}

function renderSvg(svg) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "sv-map-logo-"));
  try {
    fs.writeFileSync(path.join(dir, "icon.svg"), svg);
    execFileSync("qlmanage", ["-t", "-s", "256", "-o", dir, path.join(dir, "icon.svg")], { stdio: "ignore" });
    return new Uint8Array(fs.readFileSync(path.join(dir, "icon.svg.png")));
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

function brandColor(hex) {
  const [r, g, b] = [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16));
  return 0.299 * r + 0.587 * g + 0.114 * b > 200 ? "111827" : hex;
}

async function simpleIconLogo(company) {
  try {
    simpleIndex ??= await (await fetch(`${SIMPLE_ICONS}/data/simple-icons.json`, { signal: AbortSignal.timeout(20000) })).json();
    const root = rootDomain(company.domain);
    const names = [company.name, company.name.replace(/\s*\(.*\)/, ""), company.name.split(" ")[0]].map(n => n.toLowerCase());
    const icon = simpleIndex.find(i => names.includes(i.title.toLowerCase()) && `${i.source} ${i.guidelines || ""}`.includes(root));
    if (!icon) return null;
    const res = await fetch(`${SIMPLE_ICONS}/icons/${icon.slug}.svg`, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    const svg = (await res.text()).replace("<svg ", `<svg fill="#${brandColor(icon.hex)}" width="256" height="256" `);
    return { bytes: renderSvg(svg), score: 256, source: `simple-icons ${icon.slug}` };
  } catch {
    return null;
  }
}

function normalizePng(file) {
  const size = imageSize(new Uint8Array(fs.readFileSync(file)));
  const isPng = fs.readFileSync(file).subarray(0, 4).toString("hex") === "89504e47";
  if (isPng && size && size.width <= 256) return;
  execFileSync("sips", ["-s", "format", "png", "-Z", "256", file, "--out", file], { stdio: "ignore" });
}

async function bestLogo(company) {
  const domain = company.domain;
  let best = null;
  const urls = [...(await declaredIcons(domain)).slice(0, 3), ...(await wikidataLogo(company)), ...SOURCES.map(source => source(domain))];
  for (const url of urls) {
    const image = await fetchImage(url);
    if (image && (!best || image.score > best.score)) best = image;
    if (best && best.score >= GOOD_SCORE) break;
  }
  if (!best || best.score < GOOD_SCORE) {
    const icon = await simpleIconLogo(company);
    if (icon && (!best || icon.score > best.score)) best = icon;
  }
  return best;
}

function currentScore(file) {
  return fs.existsSync(file) ? score(imageSize(new Uint8Array(fs.readFileSync(file)))) : 0;
}

fs.mkdirSync(LOGO_DIR, { recursive: true });
const missing = [];
for (const company of readCompanies()) {
  const file = path.join(LOGO_DIR, `${company.id}.png`);
  const current = currentScore(file);
  if (current >= GOOD_SCORE || (current > 0 && !process.argv.includes("--upgrade"))) continue;
  const logo = await bestLogo(company);
  if (logo && logo.score > current) {
    fs.writeFileSync(file, logo.bytes);
    console.log(`logo ${company.id} score ${logo.score}${logo.source ? ` from ${logo.source}` : ""}`);
  } else if (!current) {
    missing.push(company.id);
  }
}
for (const file of fs.readdirSync(LOGO_DIR)) normalizePng(path.join(LOGO_DIR, file));
if (missing.length) console.log(`no logo for: ${missing.join(", ")} (initials badge is used)`);
console.log("logos ready");
