import { useState, useEffect, useRef } from 'react'
import { sendAction, getGame } from '../api/games'
import { useGameSSE } from '../hooks/useGameSSE'
import CharacterPanel from './CharacterPanel'

interface Message {
  role: string
  content: string
}

interface Character {
  hp: number
  max_hp: number
  level: number
  xp: number
  gold: number
  inventory: string[]
  location: string
}

interface Props {
  gameId: string
}

export default function GamePlay({ gameId }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(true)
  const [character, setCharacter] = useState<Character | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    getGame(gameId).then((data) => {
      if (data.messages?.length > 0) {
        setMessages(data.messages)
      }
      if (data.character) {
        setCharacter(data.character)
      }
    })
  }, [gameId])

  useGameSSE(
    gameId,
    (text) => {
      setMessages((prev) => [...prev, { role: 'dm', content: text }])
      setThinking(false)
      getGame(gameId).then((data) => {
        if (data.character) setCharacter(data.character)
      })
    },
    () => setThinking(true),
    (msg) => {
      setMessages((prev) => [...prev, { role: 'system', content: msg }])
      setThinking(false)
    },
  )

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking])

  const handleSend = async () => {
    if (!input.trim() || thinking) return
    const action = input.trim()
    setInput('')
    setMessages((prev) => [...prev, { role: 'player', content: action }])
    await sendAction(gameId, action)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex gap-6">
      <div className="flex-1 flex flex-col">
        <div className="bg-[#0d0d14] border border-amber-900/30 rounded-lg p-4 h-[60vh] overflow-y-auto mb-4">
          {messages.map((msg, i) => (
            <div key={i} className={`mb-4 ${msg.role === 'player' ? 'text-right' : ''}`}>
              {msg.role === 'dm' && (
                <div className="mb-1">
                  <span className="text-amber-500 font-bold text-sm">Dungeon Master</span>
                </div>
              )}
              {msg.role === 'player' && (
                <div className="mb-1">
                  <span className="text-emerald-400 font-bold text-sm">You</span>
                </div>
              )}
              {msg.role === 'system' && (
                <div className="mb-1">
                  <span className="text-red-400 font-bold text-sm">System</span>
                </div>
              )}
              <div
                className={`inline-block max-w-[85%] px-4 py-3 rounded-lg text-left whitespace-pre-wrap ${
                  msg.role === 'dm'
                    ? 'bg-amber-900/20 border border-amber-900/30 text-gray-200'
                    : msg.role === 'player'
                    ? 'bg-emerald-900/20 border border-emerald-900/30 text-gray-200'
                    : 'bg-red-900/20 border border-red-900/30 text-red-300'
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))}

          {thinking && (
            <div className="mb-4">
              <span className="text-amber-500 font-bold text-sm">Dungeon Master</span>
              <div className="mt-1 text-amber-400 animate-pulse">The Dungeon Master is weaving the story...</div>
            </div>
          )}

          <div ref={scrollRef} />
        </div>

        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="What do you do?"
            disabled={thinking}
            className="flex-1 px-4 py-3 rounded bg-[#1a1a2e] border border-amber-900/40 text-white placeholder-gray-500 focus:outline-none focus:border-amber-500 disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={thinking || !input.trim()}
            className="px-6 py-3 rounded bg-amber-700 hover:bg-amber-600 text-white font-bold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Act
          </button>
        </div>
      </div>

      {character && <CharacterPanel character={character} />}
    </div>
  )
}
