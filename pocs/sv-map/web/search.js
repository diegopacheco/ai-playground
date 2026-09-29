export const CATEGORIES = {
  bigtech: { label: "Big Tech", color: "#2563eb" },
  tech: { label: "Tech", color: "#059669" },
  ailab: { label: "AI Labs", color: "#9333ea" },
  aistartup: { label: "AI Startups", color: "#ea580c" }
};

export function normalize(text) {
  return String(text || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function scoreOf(company, query) {
  const name = normalize(company.name);
  if (name === query) return 100;
  if (name.startsWith(query)) return 80;
  if (name.split(" ").some(word => word.startsWith(query))) return 60;
  if (name.includes(query)) return 40;
  const place = normalize(`${company.address} ${company.city} ${company.domain}`);
  if (place.includes(query)) return 20;
  const label = normalize(CATEGORIES[company.category]?.label);
  if (label.includes(query)) return 10;
  return 0;
}

export function searchCompanies(companies, text, category = "all") {
  const pool = category === "all" ? companies : companies.filter(c => c.category === category);
  const query = normalize(text);
  if (!query) return [...pool].sort((a, b) => a.name.localeCompare(b.name));
  return pool
    .map(company => ({ company, score: scoreOf(company, query) }))
    .filter(hit => hit.score > 0)
    .sort((a, b) => b.score - a.score || a.company.name.localeCompare(b.company.name))
    .map(hit => hit.company);
}
