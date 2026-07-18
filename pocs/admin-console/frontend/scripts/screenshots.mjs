import { chromium } from 'playwright';
const OUT = process.argv[2];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
const wait = (ms) => new Promise(r => setTimeout(r, ms));
const shot = async (name) => { await page.screenshot({ path: `${OUT}/${name}` }); console.log('  ' + name); };

await page.goto('http://localhost:4321/');
await wait(1200);
await page.fill('input[aria-label="username"], input:below(:text("username"))', 'admin').catch(()=>{});
const inputs = await page.$$('.login input');
if (inputs.length >= 2) { await inputs[0].fill('admin'); await inputs[1].fill('admin'); await page.click('.login button'); }
await wait(2500);

// pick postgres and run
await page.click('.picker-trigger'); await wait(500);
await shot('02-engine-picker.png');
const cards = await page.$$('.picker-card');
for (const c of cards) { if ((await c.textContent()).includes('demo-postgres')) { await c.click(); break; } }
await wait(2000);
await page.evaluate(() => { const cm = document.querySelector('.cm-content'); cm.focus(); document.execCommand('selectAll'); document.execCommand('insertText', false, 'SELECT c.email, c.country, o.status, o.total_cents\nFROM customers c\nJOIN orders o ON o.customer_id = c.id\nORDER BY o.total_cents DESC'); });
await wait(400);
await page.click('button:has-text("Run")'); await wait(2500);
await shot('01-console-postgres.png');

// row detail
await page.evaluate(() => document.querySelectorAll('tbody tr')[2].dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));
await wait(800); await shot('04-row-detail.png');
await page.keyboard.press('Escape'); await wait(500);

// saved queries
await page.click('.saved-trigger'); await wait(900); await shot('05-saved-queries.png');
await page.keyboard.press('Escape'); await wait(400);

// command palette
await page.keyboard.press('Meta+k'); await wait(700); await shot('03-command-palette.png');
await page.keyboard.press('Escape'); await wait(400);

// kafka console
await page.click('.picker-trigger'); await wait(500);
const cards2 = await page.$$('.picker-card');
for (const c of cards2) { if ((await c.textContent()).includes('demo-kafka')) { await c.click(); break; } }
await wait(2000);
await page.click('button:has-text("Run")'); await wait(3000);
await shot('06-console-kafka.png');

// redis console
await page.click('.picker-trigger'); await wait(500);
const cards3 = await page.$$('.picker-card');
for (const c of cards3) { if ((await c.textContent()).includes('demo-redis')) { await c.click(); break; } }
await wait(2000);
await page.click('button:has-text("Run")'); await wait(2000);
await shot('07-console-redis.png');

// denied write
await page.click('.picker-trigger'); await wait(500);
const cards4 = await page.$$('.picker-card');
for (const c of cards4) { if ((await c.textContent()).includes('demo-postgres')) { await c.click(); break; } }
await wait(1800);
await page.evaluate(() => { const cm = document.querySelector('.cm-content'); cm.focus(); document.execCommand('selectAll'); document.execCommand('insertText', false, 'DELETE FROM customers WHERE country = \'BR\''); });
await wait(400);
await page.click('button:has-text("Run")'); await wait(1800);
await shot('08-read-only-denied.png');

await page.goto('http://localhost:4321/audit-trail'); await wait(2500); await shot('09-audit-trail.png');
await page.goto('http://localhost:4321/projects'); await wait(2500); await shot('10-projects.png');
await page.goto('http://localhost:4321/settings/ai'); await wait(2000); await shot('11-ai-settings.png');
await page.goto('http://localhost:4321/users'); await wait(2000); await shot('12-users.png');
await page.goto('http://localhost:6006/?path=/story/design-system-enginelogo--all-engines'); await wait(6000); await shot('13-storybook.png');
await browser.close();
