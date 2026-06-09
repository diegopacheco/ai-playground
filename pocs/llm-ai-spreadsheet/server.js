import { createServer } from 'http';
import { readFile, writeFile } from 'fs/promises';
import { extname, join, normalize, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const MODEL = process.env.OPENAI_MODEL || 'gpt-5.4-mini';
const BASE_URL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
const API_KEY = process.env.OPENAI_API_KEY;
const TEMPERATURE = Number(process.env.OPENAI_TEMPERATURE || '0.2');
const SHEET_FILE = join(root, 'sheet.json');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
};

function send(res, status, body, headers = {}) {
  res.writeHead(status, headers);
  res.end(body);
}

function json(res, status, obj) {
  send(res, status, JSON.stringify(obj), { 'Content-Type': 'application/json' });
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (c) => { data += c; });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

async function callOpenAI(prompt) {
  const res = await fetch(`${BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${API_KEY}` },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: 'system', content: 'Answer with a single word only. No punctuation, no explanation.' },
        { role: 'user', content: prompt },
      ],
      temperature: TEMPERATURE,
    }),
  });
  if (!res.ok) throw new Error(`openai ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? '';
}

async function serveStatic(req, res) {
  const url = req.url === '/' ? '/index.html' : req.url.split('?')[0];
  const safe = normalize(url).replace(/^(\.\.[/\\])+/, '');
  const file = join(root, 'public', safe);
  try {
    const content = await readFile(file);
    send(res, 200, content, { 'Content-Type': MIME[extname(file)] || 'application/octet-stream' });
  } catch {
    send(res, 404, 'Not Found');
  }
}

const server = createServer(async (req, res) => {
  if (req.method === 'POST' && req.url === '/api/ai') {
    try {
      const body = JSON.parse((await readBody(req)) || '{}');
      const prompt = String(body.prompt || '');
      if (!prompt) return json(res, 400, { error: 'empty prompt' });
      if (!API_KEY) return json(res, 500, { error: 'OPENAI_API_KEY not set' });
      return json(res, 200, { text: await callOpenAI(prompt) });
    } catch (e) {
      return json(res, 502, { error: String(e.message || e) });
    }
  }
  if (req.method === 'POST' && req.url === '/api/save') {
    try {
      await writeFile(SHEET_FILE, (await readBody(req)) || '{}');
      return json(res, 200, { ok: true });
    } catch (e) {
      return json(res, 500, { error: String(e.message || e) });
    }
  }
  if (req.method === 'GET' && req.url === '/api/load') {
    try {
      return send(res, 200, await readFile(SHEET_FILE, 'utf8'), { 'Content-Type': 'application/json' });
    } catch {
      return json(res, 200, { sheet: {} });
    }
  }
  return serveStatic(req, res);
});

server.listen(PORT, () => {
  console.log(`llm-ai-spreadsheet on http://localhost:${PORT}`);
});
