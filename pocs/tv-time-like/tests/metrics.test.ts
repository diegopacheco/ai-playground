import { describe, expect, test } from "bun:test"
import type { Media } from "../shared/types"
import { getMetrics } from "../server/db"

const base: Media = {
  id: "movie-1",
  providerId: "1",
  provider: "local",
  type: "movie",
  title: "Film",
  year: 2026,
  overview: "",
  genres: ["Drama"],
  runtime: 120,
  poster: null,
  backdrop: null,
  color: "#000",
  status: "Released",
  rating: 8,
  inLibrary: true,
  watched: true,
  watchedAt: "2026-07-05T12:00:00.000Z",
  episodes: []
}

describe("library metrics", () => {
  test("counts watched movies and time", () => {
    const metrics = getMetrics([base])
    expect(metrics.movieCount).toBe(1)
    expect(metrics.movieMinutes).toBe(120)
    expect(metrics.genreBreakdown).toEqual([{ genre: "Drama", count: 1 }])
  })
})
