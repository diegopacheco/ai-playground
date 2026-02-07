import type { FragmentId } from "../types";

const keywordMap: { keywords: string[]; fragment: FragmentId }[] = [
  { keywords: ["nav", "navigation", "menu", "header"], fragment: "navbar" },
  { keywords: ["hero", "banner", "headline", "welcome"], fragment: "hero" },
  { keywords: ["card", "grid", "tiles", "items"], fragment: "cardGrid" },
  { keywords: ["footer", "bottom", "copyright"], fragment: "footer" },
  { keywords: ["feature", "benefit", "advantage"], fragment: "features" },
  { keywords: ["contact", "form", "email", "message"], fragment: "contactForm" },
  { keywords: ["sidebar", "aside"], fragment: "sidebar" },
  { keywords: ["table", "data", "list", "rows"], fragment: "table" },
  { keywords: ["gallery", "images", "photos", "portfolio"], fragment: "gallery" },
  { keywords: ["stats", "metrics", "numbers", "counter"], fragment: "stats" },
  { keywords: ["pricing", "plan", "tier", "subscription"], fragment: "pricing" },
  { keywords: ["cta", "call to action", "sign up", "get started"], fragment: "cta" },
];

const defaultFragments: FragmentId[] = ["navbar", "hero", "cardGrid", "footer"];

export function parsePrompt(prompt: string): FragmentId[] {
  const lower = prompt.toLowerCase();
  const matched = new Set<FragmentId>();

  for (const entry of keywordMap) {
    for (const kw of entry.keywords) {
      if (lower.includes(kw)) {
        matched.add(entry.fragment);
        break;
      }
    }
  }

  if (matched.size === 0) return defaultFragments;
  return Array.from(matched);
}
