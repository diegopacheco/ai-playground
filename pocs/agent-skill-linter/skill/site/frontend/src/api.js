const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8089';

async function getJson(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(path + ' returned ' + res.status);
  return res.json();
}

export const fetchReport = () => getJson('/api/report');
export const fetchHistory = () => getJson('/api/history');
export const fetchTree = () => getJson('/api/tree');

export async function fetchSource(path) {
  const res = await fetch(BASE + '/api/source?path=' + encodeURIComponent(path));
  if (!res.ok) throw new Error('source returned ' + res.status);
  return res.text();
}
