import { getLibrary, getMedia, saveMedia, db } from "./db.ts"
import { importTvTime } from "./importer.ts"
import { searchMedia } from "./media.ts"
import { buildAiCatalog } from "./ai.ts"
import { findTrailer } from "./trailer.ts"
import { getSettings, saveSettings } from "./db.ts"
import type { AiProvider } from "../shared/types.ts"
import type { Media } from "../shared/types.ts"
import { syncMetadata } from "./metadata.ts"

const json = (data: unknown, status = 200) => Response.json(data, { status, headers: { "Cache-Control": "no-store" } })

const server = Bun.serve({
  hostname: "127.0.0.1",
  port: Number(process.env.PORT || 3001),
  idleTimeout: 255,
  routes: {
    "/api/health": () => json({ status: "ok" }),
    "/api/library": () => json(getLibrary()),
    "/api/settings": {
      GET: () => json(getSettings()),
      PUT: async request => {
        const body = await request.json() as { aiProvider: AiProvider }
        return ["codex", "claude", "gemini"].includes(body.aiProvider) ? json(saveSettings(body)) : json({ error: "Invalid AI provider" }, 400)
      }
    },
    "/api/catalog/ai": {
      POST: async request => {
        try {
          const body = await request.json() as { topic?: string; refresh?: boolean }
          const settings = getSettings()
          return json(await buildAiCatalog(settings.aiProvider, body.topic || "", Boolean(body.refresh)))
        } catch (error) {
          return json({ error: error instanceof Error ? error.message : "AI catalog failed" }, 502)
        }
      }
    },
    "/api/metadata/sync": {
      POST: async () => {
        try {
          return json(await syncMetadata())
        } catch (error) {
          return json({ error: error instanceof Error ? error.message : "Metadata sync failed" }, 502)
        }
      }
    },
    "/api/search": async request => {
      const query = new URL(request.url).searchParams.get("q") || ""
      return json(await searchMedia(query))
    },
    "/api/trailer": async request => {
      const query = new URL(request.url).searchParams.get("q") || ""
      return json({ videoId: await findTrailer(query) })
    },
    "/api/media/:id": request => {
      const media = getMedia().find(item => item.id === request.params.id)
      return media ? json(media) : json({ error: "Media not found" }, 404)
    },
    "/api/library/:id": {
      POST: request => {
        const media = getMedia().find(item => item.id === request.params.id)
        return media ? (saveMedia(media), json(media, 201)) : json({ error: "Media not found" }, 404)
      },
      DELETE: request => {
        db.prepare("UPDATE media SET in_library = 0, watched = 0, watched_at = NULL WHERE id = ?").run(request.params.id)
        db.prepare("UPDATE episodes SET watched = 0, watched_at = NULL WHERE media_id = ?").run(request.params.id)
        return json({ removed: true })
      }
    },
    "/api/media": {
      POST: async request => {
        const media = await request.json() as Media
        saveMedia(media)
        return json(media, 201)
      }
    },
    "/api/media/:id/watched": {
      PATCH: request => {
        const media = getMedia().find(item => item.id === request.params.id)
        if (!media) return json({ error: "Media not found" }, 404)
        const watched = !media.watched
        db.prepare("UPDATE media SET in_library = 1, watched = ?, watched_at = ? WHERE id = ?").run(watched ? 1 : 0, watched ? new Date().toISOString() : null, media.id)
        return json({ watched })
      }
    },
    "/api/episodes/:id/watched": {
      PATCH: request => {
        const row = db.query("SELECT watched, media_id FROM episodes WHERE id = ?").get(request.params.id) as { watched: number; media_id: string } | null
        if (!row) return json({ error: "Episode not found" }, 404)
        const watched = !row.watched
        db.prepare("UPDATE episodes SET watched = ?, watched_at = ? WHERE id = ?").run(watched ? 1 : 0, watched ? new Date().toISOString() : null, request.params.id)
        db.prepare("UPDATE media SET in_library = 1 WHERE id = ?").run(row.media_id)
        return json({ watched })
      }
    },
    "/api/import/tvtime": {
      POST: async request => json(await importTvTime(await request.text()))
    }
  },
  fetch: () => json({ error: "Not found" }, 404)
})

console.log(`Reelmark API listening on ${server.url}`)
