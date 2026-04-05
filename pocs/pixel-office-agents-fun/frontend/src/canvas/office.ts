import { OfficeAgent, DeskSlot, Point } from './types'
import { loadCharacterSprites, loadFloorTiles, loadWallTile, loadFurnitureSprite } from './sprites'
import { findPath } from './pathfinding'
import { TILE_SIZE, SCALE, CHAR_FRAME_W, CHAR_FRAME_H, CHAR_COLS } from '../types'

const COLS = 20
const ROWS = 14
const CANVAS_W = COLS * TILE_SIZE * SCALE
const CANVAS_H = ROWS * TILE_SIZE * SCALE

const FLOOR_LAYOUT = [
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,7,7,7,7,7,7,7,7,7,0,8,8,8,8,8,8,8,8,0],
  [0,7,7,7,7,7,7,7,7,7,0,8,8,8,8,8,8,8,8,0],
  [0,7,7,7,7,7,7,7,7,7,0,8,8,8,8,8,8,8,8,0],
  [0,7,7,7,7,7,7,7,7,7,0,8,8,6,6,6,6,8,8,0],
  [0,7,7,7,6,6,6,7,7,7,0,8,8,6,6,6,6,8,8,0],
  [0,7,7,7,6,6,6,7,7,7,8,8,8,6,6,6,6,8,8,0],
  [0,7,7,7,6,6,6,7,7,7,8,8,8,6,6,6,6,8,8,0],
  [0,7,7,7,7,7,7,7,7,7,0,8,8,8,8,8,8,8,8,0],
  [0,7,7,7,7,7,7,7,7,7,0,8,8,8,8,8,8,8,8,0],
  [0,7,7,7,7,7,7,7,7,7,0,8,8,8,8,8,8,8,8,0],
  [0,7,7,7,7,7,7,7,7,7,0,8,8,8,8,8,8,8,8,0],
  [0,7,7,7,7,7,7,7,7,7,0,8,8,8,8,8,8,8,8,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]

const DESK_SLOTS: DeskSlot[] = [
  { x: 2, y: 2, chairX: 2, chairY: 3, pcX: 2, pcY: 1, facing: 0 },
  { x: 5, y: 2, chairX: 5, chairY: 3, pcX: 5, pcY: 1, facing: 0 },
  { x: 8, y: 2, chairX: 8, chairY: 3, pcX: 8, pcY: 1, facing: 0 },
  { x: 2, y: 10, chairX: 2, chairY: 9, pcX: 2, pcY: 11, facing: 0 },
  { x: 5, y: 10, chairX: 5, chairY: 9, pcX: 5, pcY: 11, facing: 0 },
  { x: 8, y: 10, chairX: 8, chairY: 9, pcX: 8, pcY: 11, facing: 0 },
]

const WALKABLE: boolean[][] = FLOOR_LAYOUT.map(row =>
  row.map(tile => tile !== 0)
)

for (const desk of DESK_SLOTS) {
  WALKABLE[desk.y][desk.x] = false
  WALKABLE[desk.pcY][desk.pcX] = false
}

export class PixelOfficeEngine {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private charSprites: HTMLImageElement[] = []
  private floorTiles: HTMLImageElement[] = []
  private wallTile: HTMLImageElement | null = null
  private furnitureSprites: Map<string, HTMLImageElement> = new Map()
  private agents: Map<string, OfficeAgent> = new Map()
  private animTimer = 0
  private loaded = false
  private onClickCb: ((agentId: string, clicks: number) => void) | null = null
  private clickTimers: Map<string, number> = new Map()

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')!
    this.ctx.imageSmoothingEnabled = false
    canvas.width = CANVAS_W
    canvas.height = CANVAS_H
    canvas.style.imageRendering = 'pixelated'
    canvas.addEventListener('click', this.handleClick.bind(this))
  }

  async init() {
    this.charSprites = await loadCharacterSprites()
    this.floorTiles = await loadFloorTiles()
    this.wallTile = await loadWallTile()

    const furniturePaths = [
      'DESK/DESK_FRONT.png', 'DESK/DESK_SIDE.png',
      'PC/PC_FRONT_ON_1.png', 'PC/PC_FRONT_ON_2.png', 'PC/PC_FRONT_ON_3.png',
      'PC/PC_FRONT_OFF.png', 'PC/PC_BACK.png',
      'WOODEN_CHAIR/WOODEN_CHAIR_FRONT.png', 'WOODEN_CHAIR/WOODEN_CHAIR_BACK.png',
      'PLANT/PLANT.png', 'LARGE_PLANT/LARGE_PLANT.png',
      'BOOKSHELF/BOOKSHELF.png', 'WHITEBOARD/WHITEBOARD.png',
      'SOFA/SOFA_FRONT.png', 'COFFEE_TABLE/COFFEE_TABLE.png',
      'CACTUS/CACTUS.png', 'SMALL_PAINTING/SMALL_PAINTING.png',
      'CLOCK/CLOCK.png', 'BIN/BIN.png',
    ]

    for (const path of furniturePaths) {
      try {
        const img = await loadFurnitureSprite(path)
        this.furnitureSprites.set(path, img)
      } catch {
      }
    }

    this.loaded = true
  }

  onClick(cb: (agentId: string, clicks: number) => void) {
    this.onClickCb = cb
  }

  private handleClick(e: MouseEvent) {
    const rect = this.canvas.getBoundingClientRect()
    const scaleX = this.canvas.width / rect.width
    const scaleY = this.canvas.height / rect.height
    const mx = (e.clientX - rect.left) * scaleX
    const my = (e.clientY - rect.top) * scaleY
    const tileX = Math.floor(mx / (TILE_SIZE * SCALE))
    const tileY = Math.floor(my / (TILE_SIZE * SCALE))

    for (const [id, agent] of this.agents) {
      const ax = Math.floor(agent.position.x)
      const ay = Math.floor(agent.position.y)
      if (Math.abs(ax - tileX) <= 1 && Math.abs(ay - tileY) <= 1) {
        const existing = this.clickTimers.get(id)
        if (existing) {
          clearTimeout(existing)
          this.clickTimers.delete(id)
          this.onClickCb?.(id, 2)
        } else {
          const timer = window.setTimeout(() => {
            this.clickTimers.delete(id)
            this.onClickCb?.(id, 1)
          }, 300)
          this.clickTimers.set(id, timer)
        }
        return
      }
    }
  }

  addAgent(id: string, name: string, agentType: string, color: string, deskIndex: number) {
    const charIndex = this.agents.size % 6
    const entrance: Point = { x: 10, y: 6 }
    const desk = DESK_SLOTS[deskIndex % DESK_SLOTS.length]

    const agent: OfficeAgent = {
      id,
      name,
      agentType,
      color,
      charIndex,
      deskIndex,
      position: { ...entrance },
      targetPosition: { x: desk.chairX, y: desk.chairY },
      path: [],
      state: 'walking',
      status: 'spawning',
      animFrame: 0,
      direction: 0,
      speechBubble: null,
      speechTimer: 0,
    }

    agent.path = findPath(entrance, agent.targetPosition!, WALKABLE, COLS, ROWS)
    this.agents.set(id, agent)
  }

  updateAgentStatus(id: string, status: OfficeAgent['status']) {
    const agent = this.agents.get(id)
    if (!agent) return
    agent.status = status

    const isWalking = agent.path.length > 0

    if (status === 'thinking') {
      if (!isWalking) {
        agent.speechBubble = '...'
        agent.speechTimer = 180
      }
    } else if (status === 'working') {
      if (!isWalking) agent.state = 'typing'
      agent.speechBubble = null
    } else if (status === 'done') {
      if (isWalking) {
        const desk = DESK_SLOTS[agent.deskIndex % DESK_SLOTS.length]
        agent.position = { x: desk.chairX, y: desk.chairY }
        agent.path = []
      }
      agent.state = 'sitting'
      agent.speechBubble = 'Done!'
      agent.speechTimer = 120
    } else if (status === 'error') {
      if (isWalking) {
        const desk = DESK_SLOTS[agent.deskIndex % DESK_SLOTS.length]
        agent.position = { x: desk.chairX, y: desk.chairY }
        agent.path = []
      }
      agent.state = 'sitting'
      agent.speechBubble = 'Error!'
      agent.speechTimer = 120
    } else if (status === 'stopped') {
      if (isWalking) {
        const desk = DESK_SLOTS[agent.deskIndex % DESK_SLOTS.length]
        agent.position = { x: desk.chairX, y: desk.chairY }
        agent.path = []
      }
      agent.state = 'sitting'
    }
  }

  setAgentTyping(id: string, typing: boolean) {
    const agent = this.agents.get(id)
    if (!agent) return
    if (typing) {
      agent.state = 'typing'
      agent.speechBubble = '...'
      agent.speechTimer = 9999
    } else {
      agent.state = 'sitting'
      agent.speechBubble = null
      agent.speechTimer = 0
    }
  }

  removeAgent(id: string) {
    this.agents.delete(id)
  }

  getAgentIds(): string[] {
    return Array.from(this.agents.keys())
  }

  update() {
    this.animTimer++
    for (const agent of this.agents.values()) {
      if (agent.path.length > 0) {
        const target = agent.path[0]
        const dx = target.x - agent.position.x
        const dy = target.y - agent.position.y
        const speed = 0.08

        if (Math.abs(dx) > speed) agent.position.x += Math.sign(dx) * speed
        else agent.position.x = target.x

        if (Math.abs(dy) > speed) agent.position.y += Math.sign(dy) * speed
        else agent.position.y = target.y

        if (dy < 0) agent.direction = 1
        else if (dy > 0) agent.direction = 0
        else if (dx !== 0) agent.direction = 2

        agent.state = 'walking'

        if (Math.abs(agent.position.x - target.x) < 0.01 && Math.abs(agent.position.y - target.y) < 0.01) {
          agent.position = { ...target }
          agent.path.shift()
          if (agent.path.length === 0) {
            agent.state = 'sitting'
          }
        }
      }

      if (this.animTimer % 10 === 0) {
        agent.animFrame = (agent.animFrame + 1) % 4
      }

      if (agent.speechTimer > 0) {
        agent.speechTimer--
        if (agent.speechTimer === 0) agent.speechBubble = null
      }
    }
  }

  render() {
    if (!this.loaded) return
    const ctx = this.ctx
    const S = TILE_SIZE * SCALE

    ctx.fillStyle = '#0f0f23'
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H)

    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const tile = FLOOR_LAYOUT[y][x]
        if (tile === 0) {
          this.drawWall(ctx, x * S, y * S, S)
        } else if (tile === 7) {
          this.drawWoodFloor(ctx, x * S, y * S, S, x, y)
        } else if (tile === 8) {
          this.drawCarpetFloor(ctx, x * S, y * S, S, x, y)
        } else if (tile === 6) {
          this.drawRugFloor(ctx, x * S, y * S, S, x, y)
        } else {
          ctx.fillStyle = '#555555'
          ctx.fillRect(x * S, y * S, S, S)
        }
      }
    }

    this.renderDecorations(ctx, S)
    this.renderDesks(ctx, S)
    this.renderAgents(ctx, S)
  }

  private drawWall(ctx: CanvasRenderingContext2D, px: number, py: number, S: number) {
    ctx.fillStyle = '#2c2440'
    ctx.fillRect(px, py, S, S)
    ctx.fillStyle = '#3a3055'
    ctx.fillRect(px + 2, py + 2, S - 4, S - 4)
    ctx.fillStyle = '#33294a'
    ctx.fillRect(px + 4, py + S - 8, S - 8, 4)
    ctx.fillStyle = '#443a60'
    ctx.fillRect(px + 6, py + 6, S - 12, 3)
  }

  private drawWoodFloor(ctx: CanvasRenderingContext2D, px: number, py: number, S: number, tx: number, ty: number) {
    const shade = ((tx + ty) % 2 === 0) ? 0 : 1
    ctx.fillStyle = shade ? '#7a5c2e' : '#6d5228'
    ctx.fillRect(px, py, S, S)
    ctx.fillStyle = shade ? '#85653a' : '#7a5c30'
    ctx.fillRect(px, py, S, 2)
    const plankH = Math.floor(S / 3)
    for (let i = 1; i < 3; i++) {
      ctx.fillStyle = '#5a4220'
      ctx.fillRect(px, py + i * plankH, S, 1)
    }
    const notchX = ((tx * 7 + ty * 13) % 3) * Math.floor(S / 3) + Math.floor(S / 6)
    ctx.fillStyle = '#5a4220'
    ctx.fillRect(px + notchX, py, 1, S)
    ctx.fillStyle = shade ? '#8d6d40' : '#85653a'
    ctx.fillRect(px + 3, py + 3, 4, 2)
  }

  private drawCarpetFloor(ctx: CanvasRenderingContext2D, px: number, py: number, S: number, tx: number, ty: number) {
    const shade = ((tx + ty) % 2 === 0) ? 0 : 1
    ctx.fillStyle = shade ? '#3a5068' : '#354a62'
    ctx.fillRect(px, py, S, S)
    ctx.fillStyle = shade ? '#3e5570' : '#39506a'
    ctx.fillRect(px + 1, py + 1, S - 2, S - 2)
    if ((tx + ty) % 3 === 0) {
      ctx.fillStyle = '#405872'
      ctx.fillRect(px + 8, py + 8, 6, 6)
    }
  }

  private drawRugFloor(ctx: CanvasRenderingContext2D, px: number, py: number, S: number, tx: number, ty: number) {
    ctx.fillStyle = '#5c3030'
    ctx.fillRect(px, py, S, S)
    ctx.fillStyle = '#6b3838'
    ctx.fillRect(px + 1, py + 1, S - 2, S - 2)
    ctx.fillStyle = '#7a4040'
    ctx.fillRect(px + 3, py + 3, S - 6, S - 6)
    if ((tx + ty) % 2 === 0) {
      ctx.fillStyle = '#804545'
      ctx.fillRect(px + 6, py + 6, 8, 8)
    }
    ctx.fillStyle = '#502828'
    ctx.fillRect(px, py, S, 1)
    ctx.fillRect(px, py, 1, S)
  }

  private renderDecorations(ctx: CanvasRenderingContext2D, S: number) {
    const decorations: { sprite: string; x: number; y: number; w: number; h: number }[] = [
      { sprite: 'BOOKSHELF/BOOKSHELF.png', x: 1, y: 0, w: 1, h: 2 },
      { sprite: 'BOOKSHELF/BOOKSHELF.png', x: 3, y: 0, w: 1, h: 2 },
      { sprite: 'WHITEBOARD/WHITEBOARD.png', x: 6, y: 0, w: 2, h: 2 },
      { sprite: 'PLANT/PLANT.png', x: 1, y: 5, w: 1, h: 1 },
      { sprite: 'PLANT/PLANT.png', x: 9, y: 5, w: 1, h: 1 },
      { sprite: 'LARGE_PLANT/LARGE_PLANT.png', x: 1, y: 7, w: 1, h: 2 },
      { sprite: 'BIN/BIN.png', x: 9, y: 7, w: 1, h: 1 },
      { sprite: 'SOFA/SOFA_FRONT.png', x: 13, y: 4, w: 3, h: 2 },
      { sprite: 'COFFEE_TABLE/COFFEE_TABLE.png', x: 14, y: 6, w: 1, h: 1 },
      { sprite: 'CACTUS/CACTUS.png', x: 18, y: 1, w: 1, h: 1 },
      { sprite: 'SMALL_PAINTING/SMALL_PAINTING.png', x: 15, y: 0, w: 1, h: 2 },
      { sprite: 'CLOCK/CLOCK.png', x: 17, y: 0, w: 1, h: 1 },
      { sprite: 'PLANT/PLANT.png', x: 18, y: 12, w: 1, h: 1 },
      { sprite: 'LARGE_PLANT/LARGE_PLANT.png', x: 11, y: 10, w: 1, h: 2 },
      { sprite: 'BOOKSHELF/BOOKSHELF.png', x: 12, y: 0, w: 1, h: 2 },
    ]

    for (const d of decorations) {
      const img = this.furnitureSprites.get(d.sprite)
      if (img) {
        ctx.drawImage(img, d.x * S, d.y * S, d.w * S, d.h * S)
      } else {
        ctx.fillStyle = '#3a5a3a'
        ctx.fillRect(d.x * S + 4, d.y * S + 4, d.w * S - 8, d.h * S - 8)
      }
    }
  }

  private renderDesks(ctx: CanvasRenderingContext2D, S: number) {
    for (const desk of DESK_SLOTS) {
      const deskImg = this.furnitureSprites.get('DESK/DESK_FRONT.png')
      if (deskImg) {
        ctx.drawImage(deskImg, desk.x * S, desk.y * S, S, S)
      } else {
        ctx.fillStyle = '#8B6914'
        ctx.fillRect(desk.x * S + 2, desk.y * S + 2, S - 4, S - 4)
      }

      const hasAgentWorking = Array.from(this.agents.values()).some(
        a => a.deskIndex === DESK_SLOTS.indexOf(desk) && (a.status === 'working' || a.status === 'thinking')
      )

      const pcSprite = hasAgentWorking
        ? `PC/PC_FRONT_ON_${(Math.floor(this.animTimer / 15) % 3) + 1}.png`
        : 'PC/PC_FRONT_OFF.png'
      const pcImg = this.furnitureSprites.get(pcSprite)
      if (pcImg) {
        ctx.drawImage(pcImg, desk.pcX * S, desk.pcY * S, S, S)
      } else {
        ctx.fillStyle = hasAgentWorking ? '#4488ff' : '#333333'
        ctx.fillRect(desk.pcX * S + 6, desk.pcY * S + 2, S - 12, S - 8)
        ctx.fillStyle = '#222222'
        ctx.fillRect(desk.pcX * S + S / 2 - 3, desk.pcY * S + S - 8, 6, 6)
      }

      const chairImg = this.furnitureSprites.get('WOODEN_CHAIR/WOODEN_CHAIR_FRONT.png')
      if (chairImg) {
        ctx.drawImage(chairImg, desk.chairX * S, desk.chairY * S, S, S)
      } else {
        ctx.fillStyle = '#654321'
        ctx.fillRect(desk.chairX * S + 8, desk.chairY * S + 4, S - 16, S - 8)
      }
    }
  }

  private renderAgents(ctx: CanvasRenderingContext2D, S: number) {
    const sorted = Array.from(this.agents.values()).sort((a, b) => a.position.y - b.position.y)

    for (const agent of sorted) {
      const px = agent.position.x * S
      const py = agent.position.y * S - S * 0.5

      const sprite = this.charSprites[agent.charIndex]
      if (sprite) {
        const frameX = (agent.state === 'walking' ? agent.animFrame : (agent.state === 'typing' ? (agent.animFrame % 2) + 4 : 0))
        const row = agent.direction
        ctx.drawImage(
          sprite,
          frameX * CHAR_FRAME_W, row * CHAR_FRAME_H,
          CHAR_FRAME_W, CHAR_FRAME_H,
          px, py,
          CHAR_FRAME_W * SCALE, CHAR_FRAME_H * SCALE
        )
      } else {
        ctx.fillStyle = agent.color
        ctx.fillRect(px + 8, py + 4, S - 16, S * 1.5)
        ctx.fillStyle = '#ffccaa'
        ctx.fillRect(px + 12, py + 4, S - 24, S * 0.4)
      }

      this.renderNameTag(ctx, agent, px, py)

      if (agent.speechBubble) {
        this.renderSpeechBubble(ctx, agent.speechBubble, px + S / 2, py - 10)
      }

      this.renderStatusIndicator(ctx, agent, px + S - 8, py + 4)
    }
  }

  private renderNameTag(ctx: CanvasRenderingContext2D, agent: OfficeAgent, px: number, py: number) {
    const S = TILE_SIZE * SCALE
    ctx.font = 'bold 10px Courier New'
    ctx.textAlign = 'center'
    const textW = ctx.measureText(agent.name).width
    ctx.fillStyle = 'rgba(0,0,0,0.7)'
    ctx.fillRect(px + S / 2 - textW / 2 - 3, py - 14, textW + 6, 14)
    ctx.fillStyle = agent.color
    ctx.fillText(agent.name, px + S / 2, py - 3)
  }

  private renderSpeechBubble(ctx: CanvasRenderingContext2D, text: string, x: number, y: number) {
    ctx.font = '9px Courier New'
    ctx.textAlign = 'center'
    const w = ctx.measureText(text).width + 12
    const h = 16

    ctx.fillStyle = '#ffffff'
    ctx.beginPath()
    ctx.roundRect(x - w / 2, y - h - 4, w, h, 4)
    ctx.fill()

    ctx.fillStyle = '#ffffff'
    ctx.beginPath()
    ctx.moveTo(x - 4, y - 4)
    ctx.lineTo(x, y + 2)
    ctx.lineTo(x + 4, y - 4)
    ctx.fill()

    ctx.fillStyle = '#333333'
    ctx.fillText(text, x, y - h + 8)
  }

  private renderStatusIndicator(ctx: CanvasRenderingContext2D, agent: OfficeAgent, x: number, y: number) {
    const colors: Record<string, string> = {
      spawning: '#888888',
      thinking: '#ffaa00',
      working: '#00cc44',
      done: '#4488ff',
      error: '#ff4444',
      stopped: '#888888',
    }
    ctx.fillStyle = colors[agent.status] || '#888888'
    ctx.beginPath()
    ctx.arc(x, y, 4, 0, Math.PI * 2)
    ctx.fill()

    if (agent.status === 'working' || agent.status === 'thinking') {
      ctx.strokeStyle = colors[agent.status]
      ctx.lineWidth = 1
      ctx.globalAlpha = 0.5 + 0.5 * Math.sin(this.animTimer * 0.1)
      ctx.beginPath()
      ctx.arc(x, y, 7, 0, Math.PI * 2)
      ctx.stroke()
      ctx.globalAlpha = 1
    }
  }

  getCanvasSize() {
    return { width: CANVAS_W, height: CANVAS_H }
  }

  destroy() {
    this.canvas.removeEventListener('click', this.handleClick.bind(this))
  }
}
