import { searchCompanies, CATEGORIES } from "./search.js";
import { escapeHtml } from "./html.js";

export function createPalette({ overlay, input, list, companies, logoHtml, onPick }) {
  let hits = [];
  let active = 0;

  const render = () => {
    hits = searchCompanies(companies, input.value).slice(0, 40);
    active = Math.min(active, Math.max(hits.length - 1, 0));
    list.innerHTML = hits.length
      ? hits.map((c, i) => `<li data-i="${i}" class="${i === active ? "active" : ""}">${logoHtml(c)}<span>${escapeHtml(c.name)}</span><span class="meta">${CATEGORIES[c.category].label} · ${escapeHtml(c.city)}</span></li>`).join("")
      : '<li class="meta">No company matches.</li>';
    list.querySelector(".active")?.scrollIntoView({ block: "nearest" });
  };
  const close = () => { overlay.hidden = true; };
  const open = () => { overlay.hidden = false; input.value = ""; active = 0; render(); input.focus(); };
  const pick = index => { const company = hits[index]; if (company) { close(); onPick(company); } };

  input.addEventListener("input", () => { active = 0; render(); });
  input.addEventListener("keydown", event => {
    if (event.key === "ArrowDown") { active = Math.min(active + 1, hits.length - 1); render(); event.preventDefault(); }
    if (event.key === "ArrowUp") { active = Math.max(active - 1, 0); render(); event.preventDefault(); }
    if (event.key === "Enter") { pick(active); event.preventDefault(); }
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      if (input.value) { input.value = ""; render(); } else close();
    }
  });
  list.addEventListener("click", event => {
    const item = event.target.closest("li[data-i]");
    if (item) pick(Number(item.dataset.i));
  });
  overlay.addEventListener("mousedown", event => { if (event.target === overlay) close(); });
  return { open, close, toggle: () => (overlay.hidden ? open() : close()), isOpen: () => !overlay.hidden };
}
