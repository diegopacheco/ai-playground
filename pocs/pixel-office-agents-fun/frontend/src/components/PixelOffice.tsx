import { useRef, useEffect, useCallback } from 'react'
import { PixelOfficeEngine } from '../canvas/office'
import { Agent, AGENT_TYPES } from '../types'

interface Props {
  agents: Agent[]
  onAgentClick: (agentId: string, clicks: number) => void
}

export default function PixelOffice({ agents, onAgentClick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<PixelOfficeEngine | null>(null)
  const knownAgentsRef = useRef<Set<string>>(new Set())
  const rafRef = useRef<number>(0)

  const loop = useCallback(() => {
    const engine = engineRef.current
    if (!engine) return
    engine.update()
    engine.render()
    rafRef.current = requestAnimationFrame(loop)
  }, [])

  useEffect(() => {
    if (!canvasRef.current) return
    const engine = new PixelOfficeEngine(canvasRef.current)
    engineRef.current = engine

    engine.onClick((id, clicks) => onAgentClick(id, clicks))

    engine.init().then(() => {
      rafRef.current = requestAnimationFrame(loop)
    })

    return () => {
      cancelAnimationFrame(rafRef.current)
      engine.destroy()
    }
  }, [loop, onAgentClick])

  useEffect(() => {
    const engine = engineRef.current
    if (!engine) return

    const currentIds = new Set(agents.map(a => a.id))

    for (const oldId of knownAgentsRef.current) {
      if (!currentIds.has(oldId)) {
        engine.removeAgent(oldId)
        knownAgentsRef.current.delete(oldId)
      }
    }

    for (const agent of agents) {
      if (!knownAgentsRef.current.has(agent.id)) {
        const typeInfo = AGENT_TYPES.find(t => t.id === agent.agent_type)
        engine.addAgent(
          agent.id,
          agent.name,
          agent.agent_type,
          typeInfo?.color || '#888888',
          agent.desk_index
        )
        knownAgentsRef.current.add(agent.id)
      }
      engine.updateAgentStatus(agent.id, agent.status)
    }
  }, [agents])

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        maxWidth: 960,
        imageRendering: 'pixelated',
        border: '2px solid #333',
        borderRadius: 8,
        cursor: 'pointer',
      }}
    />
  )
}
