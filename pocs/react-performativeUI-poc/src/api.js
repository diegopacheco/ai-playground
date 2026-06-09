export async function ask(prompt, model) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ prompt, model }),
  })
  const data = await res.json().catch(() => ({ error: 'Malformed server response.' }))
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status}).`)
  return data.text
}
