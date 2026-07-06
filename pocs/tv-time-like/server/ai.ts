import type { AiProvider, CatalogItem, CatalogResponse, Media } from "../shared/types.ts"
import { readFileSync } from "node:fs"
import { getCatalogCache, libraryTitleKey, libraryTitleKeys, saveCatalogCache } from "./db.ts"
import { searchMedia } from "./media.ts"

type AiItem = {
  title: string
  type: "movie" | "show"
  year: number
  summary: string
  genres: string[]
  runtime: number
  sourceUrl: string
  sourceName: string
  reason: string
}

const promptTemplate = readFileSync("prompts/catalog.md", "utf8")
const prompt = (topic: string) => promptTemplate.replace("{{topic}}", topic.trim() || "noteworthy films and television across eras")
const schema = readFileSync("server/ai-schema.json", "utf8")

const commands: Record<AiProvider, (value: string) => string[]> = {
  codex: value => ["codex", "--search", "exec", "--ephemeral", "--sandbox", "read-only", "--output-schema", "server/ai-schema.json", value],
  claude: value => ["claude", "-p", "--output-format", "json", "--no-session-persistence", "--permission-mode", "bypassPermissions", "--allowed-tools", "WebSearch", "WebFetch", "--disallowed-tools", "Bash", "Write", "Edit", "--json-schema", schema, value],
  gemini: value => ["gemini", "-p", value, "--output-format", "json", "--approval-mode", "plan"]
}

const run = async (provider: AiProvider, value: string) => {
  const args = commands[provider](value)
  const child = Bun.spawn(args, { cwd: process.cwd(), stdout: "pipe", stderr: "pipe", env: process.env })
  const timer = setTimeout(() => child.kill(), 180_000)
  const [stdout, stderr, exitCode] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
  clearTimeout(timer)
  if (exitCode !== 0) throw new Error(stderr.trim() || `${provider} exited with code ${exitCode}`)
  if (provider === "gemini") {
    const outer = JSON.parse(stdout)
    return JSON.parse(outer.response || outer.result || stdout)
  }
  if (provider === "claude") {
    const outer = JSON.parse(stdout)
    return outer.structured_output || JSON.parse(outer.result)
  }
  return JSON.parse(stdout)
}

const colorFor = (title: string) => {
  const colors = ["#e86d52", "#2f6d64", "#d9947c", "#3f7890", "#a55f35", "#496c4f"]
  return colors[[...title].reduce((sum, char) => sum + char.charCodeAt(0), 0) % colors.length]
}

const enrich = async (item: AiItem): Promise<CatalogItem> => {
  const matches = await searchMedia(item.title)
  const match = matches.find(media => media.type === item.type && media.title.toLowerCase() === item.title.toLowerCase())
  const media: Media = match || {
    id: `ai-${item.type}-${item.title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
    providerId: item.title,
    provider: "ai-catalog",
    type: item.type,
    title: item.title,
    year: item.year,
    overview: item.summary,
    genres: item.genres,
    runtime: item.runtime,
    poster: null,
    backdrop: null,
    color: colorFor(item.title),
    status: item.type === "movie" ? "Released" : "Unknown",
    rating: 0,
    inLibrary: false,
    watched: false,
    watchedAt: null,
    episodes: []
  }
  return { media, sourceUrl: item.sourceUrl, sourceName: item.sourceName, reason: item.reason }
}

export const buildAiCatalog = async (provider: AiProvider, topic: string, refresh: boolean): Promise<CatalogResponse> => {
  const cacheKey = `${provider}:${topic.trim().toLowerCase() || "all"}`
  const owned = libraryTitleKeys()
  const available = (response: CatalogResponse): CatalogResponse => ({ ...response, items: response.items.filter(item => !owned.has(libraryTitleKey(item.media.type, item.media.title))) })
  if (!refresh) {
    const cached = getCatalogCache(cacheKey)
    if (cached) return available(cached)
  }
  const data = await run(provider, prompt(topic)) as { items: AiItem[] }
  const items = await Promise.all(data.items.map(enrich))
  const response: CatalogResponse = { provider, cached: false, generatedAt: new Date().toISOString(), items }
  saveCatalogCache(cacheKey, response)
  return available(response)
}
