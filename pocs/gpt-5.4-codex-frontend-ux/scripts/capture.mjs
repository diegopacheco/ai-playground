import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pages = [
  ["index.html", "landing-01.png"],
  ["landing-02.html", "landing-02.png"],
  ["landing-03.html", "landing-03.png"],
  ["landing-04.html", "landing-04.png"],
  ["landing-05.html", "landing-05.png"],
  ["landing-06.html", "landing-06.png"],
  ["landing-07.html", "landing-07.png"],
  ["landing-08.html", "landing-08.png"],
  ["landing-09.html", "landing-09.png"],
  ["landing-10.html", "landing-10.png"]
];

await mkdir("screenshots", { recursive: true });

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1800 } });

for (const [pagePath, file] of pages) {
  await page.goto(`file://${root}/${pagePath}`, { waitUntil: "load" });
  await page.screenshot({ path: `screenshots/${file}`, fullPage: true });
}

await browser.close();
