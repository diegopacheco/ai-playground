import puppeteer from "puppeteer";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const url = process.argv[2];
const outDir = path.resolve(process.argv[3] || "a11y-report");

if (!url) {
  console.error("usage: audit.mjs <url> [outDir]");
  process.exit(1);
}

function analyze() {
  const issues = [];
  let counter = 0;
  const add = (o) => issues.push({ id: ++counter, ...o });

  function isVisible(el) {
    const s = getComputedStyle(el);
    if (s.display === "none" || s.visibility === "hidden" || parseFloat(s.opacity) === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 || r.height > 0;
  }
  function rectOf(el) {
    const r = el.getBoundingClientRect();
    return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
  }
  function cssPath(el) {
    const parts = [];
    let e = el;
    while (e && e.nodeType === 1 && parts.length < 5) {
      if (e.id) { parts.unshift("#" + e.id); break; }
      let sel = e.tagName.toLowerCase();
      if (e.classList.length) sel += "." + [...e.classList].slice(0, 2).join(".");
      const parent = e.parentElement;
      if (parent) {
        const sib = [...parent.children].filter((c) => c.tagName === e.tagName);
        if (sib.length > 1) sel += ":nth-of-type(" + (sib.indexOf(e) + 1) + ")";
      }
      parts.unshift(sel);
      e = e.parentElement;
    }
    return parts.join(" > ");
  }
  function snippet(el) {
    return el.outerHTML.replace(/\s+/g, " ").slice(0, 160);
  }
  function parseColor(str) {
    if (!str) return null;
    const m = str.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(",").map((s) => parseFloat(s.trim()));
    return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1];
  }
  function lum(c) {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
  }
  function ratio(a, b) {
    const L1 = lum(a), L2 = lum(b);
    const hi = Math.max(L1, L2), lo = Math.min(L1, L2);
    return (hi + 0.05) / (lo + 0.05);
  }
  function effBg(el) {
    let e = el;
    while (e) {
      const c = parseColor(getComputedStyle(e).backgroundColor);
      if (c && c[3] !== 0) return c;
      e = e.parentElement;
    }
    return [255, 255, 255, 1];
  }

  const html = document.documentElement;
  if (!html.getAttribute("lang"))
    add({ wcag: "3.1.1", criterion: "Language of Page", impact: "serious", selector: "html", snippet: "<html>", message: "The <html> element has no lang attribute, so assistive technology cannot determine the page language.", fix: "Add lang to <html>, for instance <html lang=\"en\">.", rect: null });
  if (!document.title || !document.title.trim())
    add({ wcag: "2.4.2", criterion: "Page Titled", impact: "serious", selector: "title", snippet: "<title>", message: "The document has no title.", fix: "Add a short descriptive <title> in the document head.", rect: null });
  if (!document.querySelector("main, [role=main]"))
    add({ wcag: "1.3.1", criterion: "Info and Relationships", impact: "moderate", selector: "body", snippet: "<body>", message: "There is no <main> landmark, so screen-reader users cannot skip straight to the primary content.", fix: "Wrap the primary content in a <main> element.", rect: null });
  if (!document.querySelector("h1"))
    add({ wcag: "2.4.6", criterion: "Headings and Labels", impact: "moderate", selector: "body", snippet: "<body>", message: "The page has no <h1> heading.", fix: "Add a single descriptive <h1> that names the page.", rect: null });

  document.querySelectorAll("img").forEach((img) => {
    if (!isVisible(img)) return;
    if (!img.hasAttribute("alt"))
      add({ wcag: "1.1.1", criterion: "Non-text Content", impact: "critical", selector: cssPath(img), snippet: snippet(img), message: "Image has no alt attribute, so its content is invisible to screen-reader users.", fix: "Add alt text describing the image, or alt=\"\" if it is purely decorative.", rect: rectOf(img) });
  });

  document.querySelectorAll("input, select, textarea").forEach((el) => {
    const type = (el.getAttribute("type") || "").toLowerCase();
    if (["hidden", "submit", "button", "reset", "image"].includes(type)) return;
    if (!isVisible(el)) return;
    const id = el.id;
    const labelled = (id && document.querySelector('label[for="' + CSS.escape(id) + '"]')) || el.closest("label") || el.getAttribute("aria-label") || el.getAttribute("aria-labelledby") || el.getAttribute("title");
    if (!labelled)
      add({ wcag: "4.1.2", criterion: "Name, Role, Value", impact: "critical", selector: cssPath(el), snippet: snippet(el), message: "Form control has no associated label or accessible name. A placeholder is not a label.", fix: "Add a <label for> tied to the control, or an aria-label.", rect: rectOf(el) });
  });

  document.querySelectorAll("button, a").forEach((el) => {
    if (!isVisible(el)) return;
    const img = el.querySelector("img");
    const name = (el.textContent || "").trim() || el.getAttribute("aria-label") || el.getAttribute("title") || (img && img.getAttribute("alt"));
    if (!name)
      add({ wcag: "4.1.2", criterion: "Name, Role, Value", impact: "serious", selector: cssPath(el), snippet: snippet(el), message: "<" + el.tagName.toLowerCase() + "> has no accessible name (no text, aria-label, or title).", fix: "Add visible text or an aria-label so the control announces its purpose.", rect: rectOf(el) });
  });

  const vague = ["click here", "read more", "here", "more", "link", "learn more", "this"];
  document.querySelectorAll("a").forEach((a) => {
    if (!isVisible(a)) return;
    const t = (a.textContent || "").trim().toLowerCase();
    if (vague.includes(t))
      add({ wcag: "2.4.4", criterion: "Link Purpose (In Context)", impact: "minor", selector: cssPath(a), snippet: snippet(a), message: 'Link text "' + t + '" does not describe where the link goes.', fix: "Use link text that names the destination.", rect: rectOf(a) });
  });

  document.querySelectorAll("[tabindex]").forEach((el) => {
    const ti = parseInt(el.getAttribute("tabindex"), 10);
    if (ti > 0 && isVisible(el))
      add({ wcag: "2.4.3", criterion: "Focus Order", impact: "moderate", selector: cssPath(el), snippet: snippet(el), message: "Positive tabindex (" + ti + ") overrides the natural keyboard focus order and is fragile.", fix: "Use tabindex=\"0\" and order the DOM to match the visual order.", rect: rectOf(el) });
  });

  document.querySelectorAll('[role="button"]').forEach((el) => {
    if (!isVisible(el)) return;
    const native = ["a", "button", "input", "select", "textarea"].includes(el.tagName.toLowerCase());
    if (!native && !el.hasAttribute("tabindex"))
      add({ wcag: "2.1.1", criterion: "Keyboard", impact: "serious", selector: cssPath(el), snippet: snippet(el), message: 'Element has role="button" but is not keyboard focusable, so keyboard users cannot reach it.', fix: "Add tabindex=\"0\" and a key handler, or use a native <button>.", rect: rectOf(el) });
  });

  const byId = {};
  document.querySelectorAll("[id]").forEach((el) => { (byId[el.id] = byId[el.id] || []).push(el); });
  Object.entries(byId).forEach(([id, els]) => {
    if (els.length > 1)
      add({ wcag: "4.1.1", criterion: "Parsing", impact: "minor", selector: cssPath(els[1]), snippet: snippet(els[1]), message: 'Duplicate id "' + id + '" appears ' + els.length + " times; ids must be unique.", fix: "Give each element a unique id.", rect: isVisible(els[1]) ? rectOf(els[1]) : null });
  });

  let prev = 0;
  document.querySelectorAll("h1,h2,h3,h4,h5,h6").forEach((h) => {
    if (!isVisible(h)) return;
    const lvl = parseInt(h.tagName[1], 10);
    if (prev && lvl > prev + 1)
      add({ wcag: "1.3.1", criterion: "Info and Relationships", impact: "moderate", selector: cssPath(h), snippet: snippet(h), message: "Heading level jumps from h" + prev + " to h" + lvl + ", skipping a level and breaking the outline.", fix: "Use h" + (prev + 1) + " here, or restructure the headings in order.", rect: rectOf(h) });
    prev = lvl;
  });

  const seenContrast = new Set();
  document.querySelectorAll("body *").forEach((el) => {
    if (!isVisible(el)) return;
    const direct = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim().length > 0);
    if (!direct) return;
    const cs = getComputedStyle(el);
    const fg = parseColor(cs.color);
    if (!fg) return;
    const bg = effBg(el);
    const r = ratio(fg, bg);
    const fs = parseFloat(cs.fontSize);
    const bold = parseInt(cs.fontWeight, 10) >= 700 || cs.fontWeight === "bold";
    const large = fs >= 24 || (fs >= 18.66 && bold);
    const req = large ? 3 : 4.5;
    if (r < req) {
      const key = cssPath(el);
      if (seenContrast.has(key)) return;
      seenContrast.add(key);
      add({ wcag: "1.4.3", criterion: "Contrast (Minimum)", impact: "serious", selector: key, snippet: snippet(el), message: "Text contrast ratio " + r.toFixed(2) + ":1 is below the " + req + ":1 minimum for " + (large ? "large" : "normal") + " text.", fix: "Darken the text or lighten the background until the ratio meets WCAG AA.", rect: rectOf(el), meta: { ratio: +r.toFixed(2), required: req, fg: "rgb(" + fg[0] + "," + fg[1] + "," + fg[2] + ")", bg: "rgb(" + bg[0] + "," + bg[1] + "," + bg[2] + ")" } });
    }
  });

  return {
    issues,
    doc: { w: document.documentElement.scrollWidth, h: document.documentElement.scrollHeight },
    url: location.href,
    title: document.title,
  };
}

const browser = await puppeteer.launch({ headless: true, args: ["--no-sandbox", "--disable-setuid-sandbox"] });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });

try {
  await page.goto(url, { waitUntil: "networkidle2", timeout: 30000 });
} catch (e) {
  console.error("could not load " + url + ": " + e.message);
  await browser.close();
  process.exit(1);
}
await new Promise((r) => setTimeout(r, 600));

const result = await page.evaluate(analyze);
const shot = await page.screenshot({ fullPage: true, type: "png" });
await browser.close();

const imgW = shot.readUInt32BE(16);
const imgH = shot.readUInt32BE(20);

const weights = { critical: 15, serious: 10, moderate: 5, minor: 2 };
const principleOf = (w) => ({ 1: "Perceivable", 2: "Operable", 3: "Understandable", 4: "Robust" })[w[0]] || "Robust";

for (const i of result.issues) {
  i.principle = principleOf(i.wcag);
  i.box = i.rect ? { left: (i.rect.x / imgW) * 100, top: (i.rect.y / imgH) * 100, width: (i.rect.w / imgW) * 100, height: (i.rect.h / imgH) * 100 } : null;
}

const order = { critical: 0, serious: 1, moderate: 2, minor: 3 };
result.issues.sort((a, b) => order[a.impact] - order[b.impact] || a.id - b.id);

const deduction = result.issues.reduce((s, i) => s + (weights[i.impact] || 3), 0);
const score = Math.max(0, 100 - deduction);
const grade = score >= 90 ? "A" : score >= 75 ? "B" : score >= 60 ? "C" : score >= 40 ? "D" : "F";

const count = (k) => result.issues.filter((i) => i.impact === k).length;
const principleCount = {};
for (const i of result.issues) principleCount[i.principle] = (principleCount[i.principle] || 0) + 1;

const data = {
  url: result.url,
  title: result.title,
  generatedAt: new Date().toISOString(),
  score,
  grade,
  image: { w: imgW, h: imgH },
  totals: { all: result.issues.length, critical: count("critical"), serious: count("serious"), moderate: count("moderate"), minor: count("minor") },
  principles: ["Perceivable", "Operable", "Understandable", "Robust"].map((p) => ({ name: p, count: principleCount[p] || 0 })),
  issues: result.issues,
};

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, "screenshot.png"), shot);
fs.writeFileSync(path.join(outDir, "data.json"), JSON.stringify(data, null, 2));

const template = fs.readFileSync(path.join(here, "..", "assets", "template.html"), "utf8");
const dataUri = "data:image/png;base64," + shot.toString("base64");
const out = template
  .replace("/*__DATA__*/", "window.__A11Y__ = " + JSON.stringify(data) + ";")
  .replace("__SHOT__", dataUri);
fs.writeFileSync(path.join(outDir, "index.html"), out);

console.log("accessibility audit  " + result.url);
console.log("score " + score + "/100  grade " + grade + "  issues " + data.totals.all);
console.log("  critical " + data.totals.critical + "  serious " + data.totals.serious + "  moderate " + data.totals.moderate + "  minor " + data.totals.minor);
for (const i of result.issues.slice(0, 12)) {
  console.log("  [" + i.impact.padEnd(8) + "] " + i.wcag + " " + i.criterion + " — " + i.selector);
}
if (result.issues.length > 12) console.log("  ... and " + (result.issues.length - 12) + " more");
console.log("report " + path.join(outDir, "index.html"));
