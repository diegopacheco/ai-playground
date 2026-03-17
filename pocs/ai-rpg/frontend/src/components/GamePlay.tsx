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
      if (data.messages?.length > 0) setMessages(data.messages)
      if (data.character) setCharacter(data.character)
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
    <div className="flex gap-5 fade-in">
      <div className="flex-1 flex flex-col">
        <div className="bg-[#1e1528]/80 backdrop-blur border border-white/8 rounded-2xl p-5 h-[65vh] overflow-y-auto mb-4 shadow-xl shadow-black/20">
          {messages.length === 0 && !thinking && (
            <div className="flex items-center justify-center h-full text-white/20 text-sm">
              Your adventure begins...
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`mb-5 fade-in ${msg.role === 'player' ? 'flex flex-col items-end' : ''}`}>
              {msg.role === 'dm' && (
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-6 h-6 rounded-lg bg-amber-500/20 flex items-center justify-center">
                    <span className="text-amber-400 text-xs">DM</span>
                  </div>
                  <span className="text-amber-400/70 font-medium text-xs uppercase tracking-wider">Dungeon Master</span>
                </div>
              )}
              {msg.role === 'player' && (
                <div className="flex items-center gap-2 mb-2 justify-end">
                  <span className="text-emerald-400/70 font-medium text-xs uppercase tracking-wider">You</span>
                  <div className="w-6 h-6 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                    <span className="text-emerald-400 text-xs">&#9823;</span>
                  </div>
                </div>
              )}
              {msg.role === 'system' && (
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-6 h-6 rounded-lg bg-red-500/20 flex items-center justify-center">
                    <span className="text-red-400 text-xs">!</span>
                  </div>
                  <span className="text-red-400/70 font-medium text-xs uppercase tracking-wider">System</span>
                </div>
              )}
              <div
                className={`max-w-[88%] px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
                  msg.role === 'dm'
                    ? 'bg-white/5 border border-white/8 text-amber-100/90'
                    : msg.role === 'player'
                    ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-100/90'
                    : 'bg-red-500/10 border border-red-500/20 text-red-300/90'
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))}

          {thinking && (
            <div className="mb-4 fade-in">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-6 rounded-lg bg-amber-500/20 flex items-center justify-center">
                  <span className="text-amber-400 text-xs">DM</span>
                </div>
                <span className="text-amber-400/70 font-medium text-xs uppercase tracking-wider">Dungeon Master</span>
              </div>
              <div className="max-w-[88%] px-4 py-3 rounded-2xl bg-white/5 border border-white/8 shimmer">
                <div className="flex items-center gap-2 text-amber-300/60 text-sm">
                  <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Weaving the story...
                </div>
              </div>
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
            className="flex-1 px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-amber-100 placeholder-white/20 focus:outline-none focus:border-amber-500/50 focus:bg-white/8 transition-all duration-200 text-sm disabled:opacity-40"
          />
          <button
            onClick={handleSend}
            disabled={thinking || !input.trim()}
            className="px-6 py-3 rounded-xl bg-gradient-to-r from-amber-600 to-amber-700 hover:from-amber-500 hover:to-amber-600 text-white font-semibold text-sm transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed shadow-lg shadow-amber-900/30"
          >
            Act
          </button>
        </div>
      </div>

      {character && <CharacterPanel character={character} />}
    </div>
  )
}
