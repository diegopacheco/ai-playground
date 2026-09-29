import { readCompanies, writeCompanies, distanceKm, sleep, USER_AGENT } from "./dataset.mjs";

const MAX_SHIFT_KM = 2.5;

function hasStreet(address) {
  return /^(\d|One\b)/i.test(address);
}

async function geocode(company) {
  const params = new URLSearchParams({ street: company.address, city: company.city, state: "California", country: "USA", format: "json", limit: "1" });
  const res = await fetch(`https://nominatim.openstreetmap.org/search?${params}`, { headers: { "User-Agent": USER_AGENT } });
  if (!res.ok) throw new Error(`nominatim ${res.status}`);
  const [hit] = await res.json();
  return hit ? { lat: Number(hit.lat), lon: Number(hit.lon) } : null;
}

const companies = readCompanies();
let moved = 0;
for (const company of companies) {
  if (!hasStreet(company.address)) continue;
  const hit = await geocode(company);
  await sleep(1100);
  if (!hit) { console.log(`miss   ${company.name}`); continue; }
  const shift = distanceKm(company, hit);
  if (shift > MAX_SHIFT_KM) { console.log(`reject ${company.name} ${shift.toFixed(1)} km away`); continue; }
  company.lat = Number(hit.lat.toFixed(5));
  company.lon = Number(hit.lon.toFixed(5));
  moved++;
  console.log(`ok     ${company.name} ${(shift * 1000).toFixed(0)} m`);
}
writeCompanies(companies);
console.log(`geocoded ${moved} of ${companies.length}`);
