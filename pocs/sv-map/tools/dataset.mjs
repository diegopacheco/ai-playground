import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const DATA_FILE = path.join(ROOT, "data/companies.json");
export const LOGO_DIR = path.join(ROOT, "data/logos");
export const USER_AGENT = "sv-map/1.0 (+https://github.com/diegopacheco/ai-playground)";

export function readCompanies() {
  return JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
}

export function writeCompanies(companies) {
  const groups = [];
  for (const company of companies) {
    const last = groups[groups.length - 1];
    if (last && last[0].category === company.category) last.push(company);
    else groups.push([company]);
  }
  const body = groups.map(group => group.map(c => "  " + JSON.stringify(c)).join(",\n")).join(",\n\n");
  fs.writeFileSync(DATA_FILE, `[\n${body}\n]\n`);
}

export function distanceKm(a, b) {
  const rad = d => (d * Math.PI) / 180;
  const dLat = rad(b.lat - a.lat);
  const dLon = rad(b.lon - a.lon);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLon / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(h));
}

export const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
