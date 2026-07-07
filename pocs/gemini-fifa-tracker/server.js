import http from 'http';
import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = 3000;
let syncInFlight = false;
let lastSyncAt = 0;
try {
  lastSyncAt = new Date(JSON.parse(fs.readFileSync(path.join(__dirname, 'bracket.json'), 'utf8')).syncedAt).getTime() || 0;
} catch (err) {
  lastSyncAt = 0;
}

const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
  const url = req.url === '/' ? '/index.html' : req.url;

  if (req.method === 'GET') {
    let filePath;
    if (url === '/bracket.json') {
      filePath = path.join(__dirname, 'bracket.json');
    } else {
      filePath = path.join(__dirname, 'public', url);
    }

    const ext = path.extname(filePath);
    const contentType = MIME_TYPES[ext] || 'text/plain';

    fs.readFile(filePath, (err, content) => {
      if (err) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Not Found');
      } else {
        res.writeHead(200, { 'Content-Type': contentType });
        res.end(content);
      }
    });
  } else if (req.method === 'POST' && url === '/api/bracket/update') {
    if (syncInFlight || Date.now() - lastSyncAt < 5 * 60 * 1000) {
      try {
        const content = fs.readFileSync(path.join(__dirname, 'bracket.json'), 'utf8');
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'success', log: 'Bracket already synced recently. Serving current real results.', data: JSON.parse(content) }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'error', message: 'No bracket data yet' }));
      }
      return;
    }
    syncInFlight = true;
    exec('node bracket-cli.js --sync', { timeout: 240000 }, (error, stdout, stderr) => {
      syncInFlight = false;
      lastSyncAt = Date.now();
      if (error) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'error', message: error.message }));
        return;
      }
      try {
        const content = fs.readFileSync(path.join(__dirname, 'bracket.json'), 'utf8');
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'success', log: stdout.trim(), data: JSON.parse(content) }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'error', message: 'Failed to read updated bracket' }));
      }
    });
  } else {
    res.writeHead(405, { 'Content-Type': 'text/plain' });
    res.end('Method Not Allowed');
  }
});

server.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
