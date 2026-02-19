import { writeFile, unlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { randomUUID } from 'node:crypto'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
}

async function extractItemsWithClaude(imagePath: string): Promise<string[]> {
  const prompt = `@${imagePath}\nExtract all grocery items visible in this image. Return ONLY a JSON array of strings. Example: ["milk","eggs","bread"]. No explanation, no markdown, only the JSON array.`

  const proc = Bun.spawn(
    ['claude', '-p', prompt, '--dangerously-skip-permissions'],
    { stdout: 'pipe', stderr: 'pipe' }
  )

  const stdout = await new Response(proc.stdout).text()
  await proc.exited

  const trimmed = stdout.trim()
  const start = trimmed.indexOf('[')
  const end = trimmed.lastIndexOf(']')
  const json = start >= 0 && end >= 0 ? trimmed.slice(start, end + 1) : '[]'

  return JSON.parse(json) as string[]
}

Bun.serve({
  port: 3001,
  async fetch(req) {
    if (req.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders })
    }

    const url = new URL(req.url)

    if (req.method === 'POST' && url.pathname === '/extract-items') {
      let tmpPath: string | null = null
      try {
        const formData = await req.formData()
        const file = formData.get('image') as File | null
        if (!file) {
          return Response.json({ error: 'No image provided' }, { status: 400, headers: corsHeaders })
        }

        const ext = (file.name.split('.').pop() ?? 'jpg').toLowerCase()
        tmpPath = join(tmpdir(), `grocery-${randomUUID()}.${ext}`)
        const buffer = await file.arrayBuffer()
        await writeFile(tmpPath, Buffer.from(buffer))

        const items = await extractItemsWithClaude(tmpPath)
        return Response.json({ items }, { headers: corsHeaders })
      } catch (err) {
        console.error('Error extracting items:', err)
        return Response.json({ error: 'Failed to process image' }, { status: 500, headers: corsHeaders })
      } finally {
        if (tmpPath) await unlink(tmpPath).catch(() => {})
      }
    }

    return new Response('Not found', { status: 404, headers: corsHeaders })
  },
})

console.log('AI server running on http://localhost:3001')
