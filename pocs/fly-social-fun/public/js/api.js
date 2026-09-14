export function connect(onEvent, onStatus) {
  const source = new EventSource('/api/stream');
  source.onmessage = (message) => onEvent(JSON.parse(message.data));
  source.onopen = () => onStatus('live');
  source.onerror = () => onStatus('reconnecting');
  return source;
}

export async function postAction(action, spot) {
  const res = await fetch(`/api/${action}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ spot }),
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error);
  return body;
}

export async function fetchProfile(flyId) {
  const res = await fetch(`/api/flies/${encodeURIComponent(flyId)}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`profile request failed with ${res.status}`);
  return res.json();
}
