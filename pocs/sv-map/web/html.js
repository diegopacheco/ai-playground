import { CATEGORIES } from "./search.js";

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
}

export function initials(name) {
  return name.split(/[\s.&()-]+/).filter(Boolean).slice(0, 2).map(w => w[0].toUpperCase()).join("");
}

export function logoHtml(company) {
  const ring = CATEGORIES[company.category].color;
  const inner = company.logo ? `<img src="/logos/${company.id}.png" alt="">` : initials(company.name);
  return `<span class="logo" style="--ring:${ring}">${inner}</span>`;
}
