import fs from 'fs';
import { chromium } from 'playwright';

const code = fs.readFileSync('public/app.js', 'utf8');
const dataMatch = code.match(/const teamsData = (\[[\s\S]*?\]);/);
const teamsData = eval(dataMatch[1]);

async function downloadImage(page, query, destPath) {
  try {
    await page.goto(`https://www.google.com/search?tbm=isch&q=${encodeURIComponent(query)}`, { waitUntil: 'domcontentloaded' });
    
    await page.waitForSelector('img', { timeout: 5000 });
    
    const src = await page.evaluate(() => {
      const imgs = document.querySelectorAll('img');
      for (let img of imgs) {
        const url = img.src || img.getAttribute('data-src');
        // Look for data URLs or normal http images that aren't Google's logo/icons
        if (url && (url.startsWith('data:image/') || (url.startsWith('http') && !url.includes('googlelogo') && !url.includes('gstatic.com/images/branding')))) {
          // ensure it has some size
          if (img.width && img.width > 50) return url;
        }
      }
      return null;
    });

    if (src) {
      if (src.startsWith('data:image/')) {
        const base64Data = src.split(',')[1];
        fs.writeFileSync(destPath, Buffer.from(base64Data, 'base64'));
      } else {
        const response = await page.context().request.get(src);
        const buffer = await response.body();
        fs.writeFileSync(destPath, buffer);
      }
      return true;
    } else {
      console.log(`No valid image found for: ${query}`);
    }
  } catch (err) {
    console.log(`Failed for ${query}: ${err.message}`);
  }
  return false;
}

async function run() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
  });
  const page = await context.newPage();

  fs.mkdirSync('public/assets/players', { recursive: true });
  fs.mkdirSync('public/assets/dishes', { recursive: true });

  for (const team of teamsData) {
    console.log('Scraping real images for', team.name);
    await downloadImage(page, `${team.star} football player 2026`, `public/assets/players/${team.id}-star.jpg`);
    
    for (let i = 0; i < team.players.length; i++) {
      await downloadImage(page, `${team.players[i]} football legend`, `public/assets/players/${team.id}-legend-${i}.jpg`);
    }

    const dishes = [`${team.name} national dish real food`, `Traditional ${team.name} cuisine`, `Popular ${team.name} street food`];
    for (let i = 0; i < 3; i++) {
      await downloadImage(page, dishes[i], `public/assets/dishes/${team.id}-dish-${i}.jpg`);
    }
  }

  await browser.close();
  console.log('All real images downloaded via Playwright!');
}
run();
