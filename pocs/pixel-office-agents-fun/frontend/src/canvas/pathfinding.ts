import { Point } from './types'

export function findPath(
  start: Point,
  end: Point,
  walkable: boolean[][],
  cols: number,
  rows: number
): Point[] {
  if (start.x === end.x && start.y === end.y) return []
  if (!isValid(end, cols, rows, walkable)) return []

  const open: { point: Point; g: number; f: number; parent: string | null }[] = []
  const closed = new Set<string>()
  const key = (p: Point) => `${p.x},${p.y}`
  const parentMap = new Map<string, Point | null>()

  const h = (p: Point) => Math.abs(p.x - end.x) + Math.abs(p.y - end.y)

  open.push({ point: start, g: 0, f: h(start), parent: null })
  parentMap.set(key(start), null)

  const dirs = [
    { x: 0, y: -1 }, { x: 0, y: 1 },
    { x: -1, y: 0 }, { x: 1, y: 0 },
  ]

  while (open.length > 0) {
    open.sort((a, b) => a.f - b.f)
    const current = open.shift()!

    if (current.point.x === end.x && current.point.y === end.y) {
      const path: Point[] = []
      let k: string | null = key(current.point)
      while (k) {
        const [x, y] = k.split(',').map(Number)
        path.unshift({ x, y })
        const parent = parentMap.get(k)
        k = parent ? key(parent) : null
      }
      path.shift()
      return path
    }

    closed.add(key(current.point))

    for (const d of dirs) {
      const next: Point = { x: current.point.x + d.x, y: current.point.y + d.y }
      const nk = key(next)
      if (closed.has(nk)) continue
      if (!isValid(next, cols, rows, walkable)) continue

      const g = current.g + 1
      const existing = open.find(o => key(o.point) === nk)
      if (!existing) {
        open.push({ point: next, g, f: g + h(next), parent: key(current.point) })
        parentMap.set(nk, current.point)
      } else if (g < existing.g) {
        existing.g = g
        existing.f = g + h(next)
        parentMap.set(nk, current.point)
      }
    }
  }

  return []
}

function isValid(p: Point, cols: number, rows: number, walkable: boolean[][]): boolean {
  return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows && walkable[p.y][p.x]
}
