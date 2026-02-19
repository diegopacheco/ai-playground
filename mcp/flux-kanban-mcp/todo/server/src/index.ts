import Anthropic from '@anthropic-ai/sdk'

const client = new Anthropic()

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
}

Bun.serve({
  port: 3001,
  async fetch(req) {
    if (req.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders })
    }

    const url = new URL(req.url)

    if (req.method === 'POST' && url.pathname === '/extract-items') {
      try {
        const formData = await req.formData()
        const file = formData.get('image') as File | null
        if (!file) {
          return Response.json({ error: 'No image provided' }, { status: 400, headers: corsHeaders })
        }

        const buffer = await file.arrayBuffer()
        const base64 = Buffer.from(buffer).toString('base64')
        const mediaType = file.type as 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp'

        const response = await client.messages.create({
          model: 'claude-sonnet-4-6',
          max_tokens: 1024,
          messages: [
            {
              role: 'user',
              content: [
                {
                  type: 'image',
                  source: { type: 'base64', media_type: mediaType, data: base64 },
                },
                {
                  type: 'text',
                  text: 'Extract all grocery items visible in this image. Return ONLY a JSON array of strings, each string being one grocery item name. Example: ["milk","eggs","bread"]. No explanation, no markdown, only the JSON array.',
                },
              ],
            },
          ],
        })

        const textBlock = response.content.find(c => c.type === 'text')
        const text = textBlock && textBlock.type === 'text' ? textBlock.text.trim() : '[]'
        const items: string[] = JSON.parse(text)

        return Response.json({ items }, { headers: corsHeaders })
      } catch (err) {
        console.error('Error extracting items:', err)
        return Response.json({ error: 'Failed to process image' }, { status: 500, headers: corsHeaders })
      }
    }

    return new Response('Not found', { status: 404, headers: corsHeaders })
  },
})

console.log('AI server running on http://localhost:3001')
