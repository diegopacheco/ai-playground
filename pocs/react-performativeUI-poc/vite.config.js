import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const SYSTEM_PROMPT =
  'You are Synthetica, a wry but genuinely helpful AI assistant. ' +
  'Answer in 2-4 concise sentences unless the user asks for more.'

function openaiProxy() {
  return {
    name: 'openai-proxy',
    configureServer(server) {
      server.middlewares.use('/api/chat', async (req, res) => {
        res.setHeader('content-type', 'application/json')
        if (req.method !== 'POST') {
          res.statusCode = 405
          return res.end(JSON.stringify({ error: 'Use POST' }))
        }
        const key = process.env.OPENAI_API_KEY
        if (!key) {
          res.statusCode = 400
          return res.end(
            JSON.stringify({ error: 'OPENAI_API_KEY is not set in the server environment.' })
          )
        }
        let raw = ''
        for await (const chunk of req) raw += chunk
        let prompt = ''
        let model = 'gpt-4o-mini'
        try {
          const parsed = JSON.parse(raw || '{}')
          prompt = (parsed.prompt || '').toString()
          if (parsed.model) model = parsed.model.toString()
        } catch {
          res.statusCode = 400
          return res.end(JSON.stringify({ error: 'Invalid JSON body.' }))
        }
        if (!prompt.trim()) {
          res.statusCode = 400
          return res.end(JSON.stringify({ error: 'Empty prompt.' }))
        }
        try {
          const upstream = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
              'content-type': 'application/json',
              authorization: `Bearer ${key}`,
            },
            body: JSON.stringify({
              model,
              temperature: 0.7,
              messages: [
                { role: 'system', content: SYSTEM_PROMPT },
                { role: 'user', content: prompt },
              ],
            }),
          })
          const data = await upstream.json()
          if (!upstream.ok) {
            res.statusCode = upstream.status
            return res.end(
              JSON.stringify({ error: data?.error?.message || 'OpenAI request failed.' })
            )
          }
          const text = data?.choices?.[0]?.message?.content ?? ''
          res.statusCode = 200
          res.end(JSON.stringify({ text, model }))
        } catch (err) {
          res.statusCode = 502
          res.end(JSON.stringify({ error: String(err) }))
        }
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), openaiProxy()],
  server: { port: 5173, strictPort: true },
})
