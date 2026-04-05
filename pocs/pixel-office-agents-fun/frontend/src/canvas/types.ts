export interface Point {
  x: number
  y: number
}

export interface OfficeAgent {
  id: string
  name: string
  agentType: string
  color: string
  charIndex: number
  deskIndex: number
  position: Point
  targetPosition: Point | null
  path: Point[]
  state: 'walking' | 'sitting' | 'typing' | 'idle'
  status: 'spawning' | 'thinking' | 'working' | 'done' | 'error' | 'stopped'
  animFrame: number
  direction: number
  speechBubble: string | null
  speechTimer: number
}

export interface DeskSlot {
  x: number
  y: number
  chairX: number
  chairY: number
  pcX: number
  pcY: number
  facing: number
}

export interface Furniture {
  type: string
  x: number
  y: number
  img: HTMLImageElement | null
}
