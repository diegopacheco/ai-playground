const cache = new Map<string, string | null>()

const headers = { "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36", "Accept-Language": "en-US,en;q=0.9" }

export const findTrailer = async (query: string): Promise<string | null> => {
  const key = query.trim().toLowerCase()
  if (!key) return null
  if (cache.has(key)) return cache.get(key) ?? null
  const url = `https://www.youtube.com/results?search_query=${encodeURIComponent(query)}&hl=en&gl=US`
  const response = await fetch(url, { headers }).catch(() => null)
  const id = response?.ok ? (await response.text()).match(/"videoId":"([\w-]{11})"/)?.[1] || null : null
  cache.set(key, id)
  return id
}
