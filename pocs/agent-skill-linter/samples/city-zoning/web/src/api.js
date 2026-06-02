const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:4000';

export async function fetchZones() {
  const res = await fetch(BASE + '/api/zones');
  return res.json();
}

export async function evaluate(zone, proposal) {
  const res = await fetch(BASE + '/api/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ zone, proposal })
  });
  return res.json();
}
