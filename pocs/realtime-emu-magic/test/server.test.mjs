import { test } from 'node:test'
import assert from 'node:assert/strict'
import { request } from 'node:http'
import { createApp } from '../server.mjs'

const input = { prompt: 'Set blue', game: { name: 'color.sfc', internalTitle: 'COLOR LAB', sha256: 'a'.repeat(64), size: 32768 }, cheats: [] }
const reply = { message: 'Set color', unsupported: false, cheats: [{ label: 'Blue', code: '7E000000' }] }

async function server(t, agent = async () => reply) {
  const app = createApp({ agent, status: async () => ({ available: true }) })
  await new Promise(resolve => app.listen(0, '127.0.0.1', resolve))
  t.after(() => { app.closeAllConnections(); app.close() })
  const origin = `http://127.0.0.1:${app.address().port}`
  return { origin, post: (body = input, headers = {}) => fetch(`${origin}/api/change`, { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(body) }) }
}

test('change endpoint forwards validated intent and returns a validated plan', async t => {
  const { post } = await server(t, async request => { assert.deepEqual(request, input); return reply })
  const response = await post()
  assert.equal(response.status, 200)
  assert.deepEqual(await response.json(), reply)
})

test('change endpoint forwards the selected Codex model', async t => {
  const selected = 'gpt-5.6-terra'
  const { post } = await server(t, async (request, signal, model) => {
    assert.equal(request.model, selected)
    assert.equal(model, selected)
    assert.equal(signal.aborted, false)
    return reply
  })
  const response = await post({ ...input, model: selected })
  assert.equal(response.status, 200)
})

test('full memory batches fit the HTTP boundary while oversized bodies are rejected', async t => {
  const cheats = Array.from({ length: 256 }, (_, index) => ({ label: 'Progress '.repeat(11), code: (0x7e1000 + index).toString(16).toUpperCase() + '0F', once: true, mask: 'DF' }))
  const body = { ...input, cheats }
  assert.ok(Buffer.byteLength(JSON.stringify(body)) > 16384)
  const { post } = await server(t, async request => ({ message: 'Apply full batch', unsupported: false, cheats: request.cheats }))
  const response = await post(body)
  assert.equal(response.status, 200)
  assert.equal((await response.json()).cheats.length, 256)
  assert.equal((await post({ ...input, prompt: 'x'.repeat(65536) })).status, 413)
})

test('cross-origin requests and invalid game data cannot invoke Codex', async t => {
  let calls = 0
  const { post } = await server(t, async () => { calls++; return reply })
  assert.equal((await post(input, { Origin: 'https://foreign.invalid' })).status, 403)
  assert.equal((await post({ ...input, game: {} })).status, 400)
  assert.equal((await post(input, { 'Content-Type': 'text/plain' })).status, 403)
  assert.equal(calls, 0)
})

test('server rejects conflicting requests and stays responsive during agent work', async t => {
  let release
  let entered
  const waiting = new Promise(resolve => { entered = resolve })
  const { post, origin } = await server(t, () => new Promise(resolve => { release = resolve; entered() }))
  const first = post()
  await waiting
  assert.equal((await fetch(`${origin}/health`)).status, 200)
  assert.equal((await post()).status, 409)
  release(reply)
  assert.equal((await first).status, 200)
})

test('malformed agent output fails without becoming a player command', async t => {
  const { post } = await server(t, async () => ({ ...reply, cheats: [{ label: 'Bad', code: 'execute()' }] }))
  assert.equal((await post()).status, 502)
})

test('private source files and traversal paths are never served', async t => {
  const { origin } = await server(t)
  for (const path of ['/agent.mjs', '/reply.schema.json', '/.git/config', '/%2e%2e/server.mjs']) assert.equal((await fetch(`${origin}${path}`)).status, 404)
})

test('a malformed URL is rejected without terminating the local server', async t => {
  const { origin } = await server(t)
  const status = await new Promise((resolve, reject) => {
    const call = request({ hostname: '127.0.0.1', port: new URL(origin).port, path: '//[' }, response => {
      response.resume()
      response.on('end', () => resolve(response.statusCode))
    })
    call.on('error', reject)
    call.end()
  })
  assert.equal(status, 400)
  assert.equal((await fetch(`${origin}/health`)).status, 200)
})

test('disconnecting cancels agent work and releases the request slot', async t => {
  let entered
  let cancelled
  const waiting = new Promise(resolve => { entered = resolve })
  const aborted = new Promise(resolve => { cancelled = resolve })
  const { origin } = await server(t, (request, signal) => new Promise((resolve, reject) => {
    signal.addEventListener('abort', () => { cancelled(); reject(new Error('Cancelled')) }, { once: true })
    entered()
  }))
  const controller = new AbortController()
  const response = fetch(`${origin}/api/change`, { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' }, body: JSON.stringify(input), signal: controller.signal }).catch(error => error)
  await waiting
  controller.abort()
  await aborted
  assert.equal((await response).name, 'AbortError')
  assert.equal((await (await fetch(`${origin}/api/status`)).json()).busy, false)
})
