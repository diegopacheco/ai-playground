import { describe, expect, test } from "bun:test"
import { parseCsv } from "../server/importer"

describe("TV Time CSV parser", () => {
  test("reads quoted fields and episode coordinates", () => {
    const rows = parseCsv('show_name,season_number,episode_number,title\n"The Bear",2,7,"Forks, Part One"\n')
    expect(rows).toEqual([{ show_name: "The Bear", season_number: "2", episode_number: "7", title: "Forks, Part One" }])
  })

  test("reads Windows line endings", () => {
    const rows = parseCsv("show_name,season,episode\r\nSeverance,1,1\r\n")
    expect(rows).toHaveLength(1)
    expect(rows[0].show_name).toBe("Severance")
  })
})
