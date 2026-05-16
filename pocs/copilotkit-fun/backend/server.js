import express from 'express';
import { CopilotRuntime, BuiltInAgent, createCopilotEndpointExpress } from '@copilotkit/runtime/v2';

process.on('unhandledRejection', (reason) => {
  console.error('Unhandled rejection (kept server alive):', reason?.message || reason);
});
process.on('uncaughtException', (err) => {
  console.error('Uncaught exception (kept server alive):', err?.message || err);
});

const app = express();

const stocks = {
  AAPL: { symbol: 'AAPL', name: 'Apple Inc.', price: 178.42, change: 1.23 },
  GOOGL: { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 142.65, change: -0.45 },
  MSFT: { symbol: 'MSFT', name: 'Microsoft Corp.', price: 412.88, change: 2.10 },
  AMZN: { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 185.07, change: 0.88 },
  TSLA: { symbol: 'TSLA', name: 'Tesla Inc.', price: 248.50, change: -3.21 },
  NVDA: { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 138.92, change: 4.55 }
};

const branches = {
  AAPL: [
    { name: 'Apple Park', lat: 37.3349, lng: -122.0090, address: 'Cupertino, CA' },
    { name: 'Apple 5th Ave', lat: 40.7637, lng: -73.9729, address: 'New York, NY' },
    { name: 'Apple Regent St', lat: 51.5126, lng: -0.1407, address: 'London, UK' },
    { name: 'Apple Marunouchi', lat: 35.6812, lng: 139.7670, address: 'Tokyo, Japan' }
  ],
  GOOGL: [
    { name: 'Googleplex', lat: 37.4220, lng: -122.0841, address: 'Mountain View, CA' },
    { name: 'Google NYC', lat: 40.7414, lng: -74.0033, address: 'New York, NY' },
    { name: 'Google London', lat: 51.5347, lng: -0.1242, address: 'London, UK' },
    { name: 'Google Zurich', lat: 47.3653, lng: 8.5249, address: 'Zurich, Switzerland' }
  ],
  MSFT: [
    { name: 'Microsoft HQ', lat: 47.6396, lng: -122.1283, address: 'Redmond, WA' },
    { name: 'Microsoft NYC', lat: 40.7549, lng: -73.9840, address: 'New York, NY' },
    { name: 'Microsoft Dublin', lat: 53.3431, lng: -6.2497, address: 'Dublin, Ireland' }
  ],
  AMZN: [
    { name: 'Amazon HQ', lat: 47.6228, lng: -122.3375, address: 'Seattle, WA' },
    { name: 'Amazon HQ2', lat: 38.8606, lng: -77.0508, address: 'Arlington, VA' },
    { name: 'Amazon Luxembourg', lat: 49.6116, lng: 6.1319, address: 'Luxembourg' }
  ],
  TSLA: [
    { name: 'Tesla HQ', lat: 30.2226, lng: -97.6175, address: 'Austin, TX' },
    { name: 'Tesla Fremont', lat: 37.4936, lng: -121.9447, address: 'Fremont, CA' },
    { name: 'Gigafactory Berlin', lat: 52.4045, lng: 13.8095, address: 'Grunheide, Germany' }
  ],
  NVDA: [
    { name: 'NVIDIA HQ', lat: 37.3700, lng: -121.9636, address: 'Santa Clara, CA' },
    { name: 'NVIDIA Tel Aviv', lat: 32.0853, lng: 34.7818, address: 'Tel Aviv, Israel' },
    { name: 'NVIDIA Bristol', lat: 51.4545, lng: -2.5879, address: 'Bristol, UK' }
  ]
};

app.get('/api/stocks', (req, res) => res.json(Object.values(stocks)));

app.get('/api/stocks/:symbol', (req, res) => {
  const stock = stocks[req.params.symbol.toUpperCase()];
  if (!stock) return res.status(404).json({ error: 'Stock not found' });
  res.json(stock);
});

app.get('/api/branches/:symbol', (req, res) => {
  const list = branches[req.params.symbol.toUpperCase()] || [];
  res.json(list);
});

if (!process.env.OPENAI_API_KEY) {
  console.warn('WARNING: OPENAI_API_KEY is not set. The Copilot chat will fail until you export it and restart.');
}

const defaultAgent = new BuiltInAgent({
  model: 'openai/gpt-4o-mini',
  ...(process.env.OPENAI_API_KEY ? { apiKey: process.env.OPENAI_API_KEY } : {}),
  maxSteps: 5,
  description: 'Stock assistant that helps users explore stock prices and find nearest company branches.'
});

const runtime = new CopilotRuntime({
  agents: { default: defaultAgent }
});

app.use(createCopilotEndpointExpress({
  runtime,
  basePath: '/api/copilotkit'
}));

const port = process.env.PORT || 4000;
app.listen(port, () => console.log(`Backend listening on http://localhost:${port}`));
