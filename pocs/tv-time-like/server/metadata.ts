import { db } from "./db.ts"

type Target = {
  id: string
  title: string
  year: number
  type: "movie" | "show"
  poster: string | null
  genres: string
}

type WikiPage = {
  title: string
  missing?: boolean
  thumbnail?: { source: string }
  pageprops?: { wikibase_item?: string }
}

type WikiResponse = {
  query?: {
    normalized?: { from: string; to: string }[]
    redirects?: { from: string; to: string }[]
    pages: WikiPage[]
  }
}

const chunks = <T,>(items: T[], size: number) => Array.from({ length: Math.ceil(items.length / size) }, (_, index) => items.slice(index * size, index * size + size))
const candidate = (target: Target) => target.title
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

const fetchRetry = async (url: string | URL, attempts = 4) => {
  for (let attempt = 0; attempt < attempts; attempt++) {
    const response = await fetch(url, { headers: { "User-Agent": "Reelmark/1.0 local-media-metadata" } }).catch(() => null)
    if (response?.ok) return response
    if (response && response.status !== 429 && response.status < 500) return response
    await delay(500 * (attempt + 1))
  }
  return null
}

const resolvePages = async (targets: Target[], fallback: boolean) => {
  const titles = targets.map(target => fallback ? target.title : candidate(target))
  const url = new URL("https://en.wikipedia.org/w/api.php")
  url.search = new URLSearchParams({ action: "query", format: "json", formatversion: "2", redirects: "1", prop: "pageimages|pageprops", piprop: "thumbnail", pithumbsize: "700", titles: titles.join("|") }).toString()
  const response = await fetchRetry(url)
  if (!response?.ok) return new Map<string, WikiPage>()
  const data = await response.json() as WikiResponse
  const aliases = new Map<string, string>()
  data.query?.normalized?.forEach(item => aliases.set(item.from.toLowerCase(), item.to))
  data.query?.redirects?.forEach(item => aliases.set(item.from.toLowerCase(), item.to))
  const pages = new Map((data.query?.pages || []).map(page => [page.title.toLowerCase(), page]))
  const result = new Map<string, WikiPage>()
  targets.forEach((target, index) => {
    let title = titles[index]
    const normalized = aliases.get(title.toLowerCase())
    if (normalized) title = normalized
    const redirected = aliases.get(title.toLowerCase())
    if (redirected) title = redirected
    const page = pages.get(title.toLowerCase())
    if (page && !page.missing) result.set(target.id, page)
  })
  return result
}

const wikipediaMetadata = async (targets: Target[]) => {
  const resolved = new Map<string, WikiPage>()
  for (const batchGroup of chunks(targets, 200)) {
    const batches = chunks(batchGroup, 40)
    const first = await Promise.all(batches.map(batch => resolvePages(batch, false)))
    first.forEach(result => result.forEach((page, id) => resolved.set(id, page)))
    await delay(300)
  }
  return resolved
}

type ImdbResult = { l?: string; y?: number; q?: string; i?: { imageUrl?: string } }
const normalizedTitle = (value: string) => value.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, " ").trim()

const imdbArtwork = async (targets: Target[]) => {
  const images = new Map<string, string>()
  for (const batch of chunks(targets, 6)) {
    const results = await Promise.all(batch.map(async target => {
      const slug = encodeURIComponent(target.title.toLowerCase())
      const response = await fetchRetry(`https://v2.sg.media-imdb.com/suggestion/x/${slug}.json`)
      if (!response?.ok) return null
      const data = await response.json() as { d?: ImdbResult[] }
      const compatible = data.d?.filter(item => {
        const format = item.q?.toLowerCase() || ""
        const typeMatches = target.type === "movie" ? !format.includes("tv series") : format.includes("tv") || format.includes("series")
        const yearMatches = !target.year || !item.y || Math.abs(item.y - target.year) <= 2
        return item.i?.imageUrl && typeMatches && yearMatches
      }) || []
      const exact = compatible.find(item => normalizedTitle(item.l || "") === normalizedTitle(target.title))
      return (exact || compatible[0])?.i?.imageUrl || null
    }))
    results.forEach((image, index) => {
      if (image) images.set(batch[index].id, image)
    })
    await delay(200)
  }
  return images
}

type Entity = { claims?: { P136?: { mainsnak?: { datavalue?: { value?: { id?: string } } } }[] }; labels?: { en?: { value: string } } }

const wikidataGenres = async (pages: Map<string, WikiPage>) => {
  const byQid = new Map<string, string[]>()
  const qids = [...new Set([...pages.values()].map(page => page.pageprops?.wikibase_item).filter((id): id is string => Boolean(id)))]
  const entities = new Map<string, Entity>()
  for (const batchGroup of chunks(qids, 250)) {
    const results = await Promise.all(chunks(batchGroup, 50).map(async batch => {
      const url = `https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=claims&ids=${batch.join("|")}`
      const response = await fetchRetry(url)
      return response?.ok ? (await response.json() as { entities: Record<string, Entity> }).entities : {}
    }))
    results.forEach(result => Object.entries(result).forEach(([id, entity]) => entities.set(id, entity)))
  }
  const genreIds = new Set<string>()
  entities.forEach((entity, id) => {
    const genres = entity.claims?.P136?.map(claim => claim.mainsnak?.datavalue?.value?.id).filter((genre): genre is string => Boolean(genre)) || []
    byQid.set(id, genres)
    genres.forEach(genre => genreIds.add(genre))
  })
  const labels = new Map<string, string>()
  for (const batchGroup of chunks([...genreIds], 250)) {
    const results = await Promise.all(chunks(batchGroup, 50).map(async batch => {
      const url = `https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=labels&languages=en&ids=${batch.join("|")}`
      const response = await fetchRetry(url)
      return response?.ok ? (await response.json() as { entities: Record<string, Entity> }).entities : {}
    }))
    results.forEach(result => Object.entries(result).forEach(([id, entity]) => {
      if (entity.labels?.en?.value) labels.set(id, entity.labels.en.value)
    }))
  }
  const byMedia = new Map<string, string[]>()
  pages.forEach((page, mediaId) => {
    const qid = page.pageprops?.wikibase_item
    const genres = qid ? (byQid.get(qid) || []).map(id => labels.get(id)).filter((label): label is string => Boolean(label)) : []
    byMedia.set(mediaId, genres)
  })
  return byMedia
}

const needsArtwork = (poster: string | null) => !poster || !poster.includes("media-amazon.com")

export const syncMetadata = async () => {
  const targets = db.query("SELECT id, title, year, type, poster, genres FROM media WHERE poster IS NULL OR poster NOT LIKE '%media-amazon.com%' OR genres = '[]'").all() as Target[]
  const pages = await wikipediaMetadata(targets)
  const genres = await wikidataGenres(pages)
  const imdbImages = await imdbArtwork(targets.filter(target => needsArtwork(target.poster)))
  const update = db.prepare("UPDATE media SET poster = COALESCE(?, poster), genres = CASE WHEN genres = '[]' AND ? != '[]' THEN ? ELSE genres END WHERE id = ?")
  let imagesUpdated = 0
  let genresUpdated = 0
  const write = db.transaction(() => {
    for (const target of targets) {
      const image = imdbImages.get(target.id) || (needsArtwork(target.poster) ? pages.get(target.id)?.thumbnail?.source : null) || null
      const values = genres.get(target.id) || []
      if (image) imagesUpdated++
      if (values.length) genresUpdated++
      const encoded = JSON.stringify(values)
      update.run(image, encoded, encoded, target.id)
    }
  })
  write()
  return { scanned: targets.length, imagesUpdated, genresUpdated, missingImages: targets.length - imagesUpdated, pagesResolved: pages.size }
}
