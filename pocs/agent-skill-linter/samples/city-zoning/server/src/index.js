import http from 'node:http';
import { ZONES } from './zoning/zones.js';
import { evaluateProposal } from './zoning/rules.js';
import { permitFee } from './zoning/permits.js';

const PORT = process.env.PORT || 4000;
const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type'
};

function sendJson(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json', ...CORS });
  res.end(JSON.stringify(body));
}

const server = http.createServer((req, res) => {
  if (req.method === 'OPTIONS') {
    res.writeHead(204, CORS);
    res.end();
    return;
  }
  if (req.method === 'GET' && req.url === '/api/health') {
    sendJson(res, 200, { status: 'ok' });
    return;
  }
  if (req.method === 'GET' && req.url === '/api/zones') {
    sendJson(res, 200, ZONES);
    return;
  }
  if (req.method === 'POST' && req.url === '/api/evaluate') {
    let raw = '';
    req.on('data', chunk => { raw += chunk; });
    req.on('end', () => {
      try {
        const payload = JSON.parse(raw);
        const result = evaluateProposal(payload.zone, payload.proposal);
        result.permitFee = permitFee(payload.proposal, result.compliant);
        sendJson(res, 200, result);
      } catch (err) {
        sendJson(res, 400, { error: err.message });
      }
    });
    return;
  }
  sendJson(res, 404, { error: 'not found' });
});

server.listen(PORT, () => {
  console.log('city-zoning-server on http://localhost:' + PORT);
});
