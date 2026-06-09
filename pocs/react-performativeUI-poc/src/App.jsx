import { useRef, useState } from 'react'
import {
  Sparkle,
  StatusDot,
  PromptHero,
  AsciiHero,
  NodeGraphBackground,
  LogoMarquee,
  Prompt,
  ChatBubble,
  TokenStream,
  MockIDE,
} from 'performative-ui'
import { ask } from './api'

const MODELS = ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini']

const LOGOS = [
  { kind: 'node', node: 'NEXUS', key: 'nexus' },
  { kind: 'node', node: 'Quantal', key: 'quantal' },
  { kind: 'node', node: 'HYPERLOOP', key: 'hyperloop' },
  { kind: 'node', node: 'Vectyr', key: 'vectyr' },
  { kind: 'node', node: 'Onyx', key: 'onyx' },
  { kind: 'node', node: 'Parallax', key: 'parallax' },
  { kind: 'node', node: 'COGNITA', key: 'cognita' },
  { kind: 'node', node: 'Mistraal', key: 'mistraal' },
]

const IDE_TOKENS = [
  { c: 'const', cls: 'key' }, { c: ' agent ' }, { c: '=' }, { c: ' ' },
  { c: 'new', cls: 'key' }, { c: ' ' }, { c: 'Synthetica', cls: 'fn' }, { c: '({ ' },
  { c: 'model', cls: 'key' }, { c: ': ' }, { c: '"gpt-4o"', cls: 'str' }, { c: ' })\n' },
  { c: 'const', cls: 'key' }, { c: ' plan ' }, { c: '=' }, { c: ' ' },
  { c: 'await', cls: 'key' }, { c: ' agent.' }, { c: 'think', cls: 'fn' }, { c: '(goal)\n' },
  { c: 'await', cls: 'key' }, { c: ' agent.' }, { c: 'ship', cls: 'fn' }, { c: '(plan)' },
]

let nextId = 0

export default function App() {
  const [messages, setMessages] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const convoRef = useRef(null)

  async function submit(text, model) {
    const prompt = (text || '').trim()
    if (!prompt || busy) return
    setError('')
    setBusy(true)
    setMessages((m) => [...m, { id: ++nextId, role: 'user', text: prompt }])
    requestAnimationFrame(() => convoRef.current?.scrollIntoView({ behavior: 'smooth' }))
    try {
      const answer = await ask(prompt, model)
      setMessages((m) => [...m, { id: ++nextId, role: 'ai', text: answer }])
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page">
      <nav className="nav">
        <span className="brand">
          <Sparkle /> Synthetica
        </span>
        <span className="nav-status">
          <StatusDot /> all systems operational
        </span>
      </nav>

      <header className="hero">
        <AsciiHero
          variant="bare"
          colorful
          baseOpacity={0.16}
          spotlightOpacity={0.85}
          spotlightRadius={9}
          className="hero-ascii"
        />
        <div className="hero-inner">
          <span className="eyebrow">
            <StatusDot color="var(--pui-grad-mid)" /> now with frontier reasoning <Sparkle solid />
          </span>
          <h1 className="hero-title">
            The agent that ships <em>while you sleep</em> <Sparkle />
          </h1>
          <p className="hero-sub">
            Describe the outcome. Synthetica plans, writes, and deploys it. One prompt,
            zero standups, infinite leverage.
          </p>
          <PromptHero
            className="hero-prompt"
            placeholder="Build me a realtime analytics dashboard…"
            ctaLabel="Generate"
            onSubmit={(value) => submit(value, MODELS[0])}
          />
          <p className="hero-hint">Wired to the OpenAI API — set OPENAI_API_KEY and ask anything.</p>
        </div>
      </header>

      <section className="marquee-wrap">
        <p className="caption">Trusted by teams shipping at the frontier</p>
        <LogoMarquee logos={LOGOS} speed={28} gap={64} fade pauseOnHover />
      </section>

      <section className="platform">
        <NodeGraphBackground
          className="platform-graph"
          density={64}
          linkColor="rgba(124,58,237,0.5)"
          colors={['#7c3aed', '#ec4899', '#06b6d4']}
          baseOpacity={0.4}
        />
        <div className="platform-inner">
          <h2 className="section-title">
            <Sparkle solid /> One model, every layer
          </h2>
          <div className="cards">
            <article className="card">
              <h3>Plans</h3>
              <p>Turns a one-line goal into a dependency-ordered task graph in milliseconds.</p>
            </article>
            <article className="card">
              <h3>Writes</h3>
              <p>Generates production code with tests, types, and a tidy commit message.</p>
            </article>
            <article className="card">
              <h3>Ships</h3>
              <p>Opens the PR, watches CI, and rolls forward the moment it goes green.</p>
            </article>
          </div>
        </div>
      </section>

      <section className="ide-wrap">
        <div className="ide-copy">
          <h2 className="section-title">
            Watch it think in real time <Sparkle />
          </h2>
          <p className="muted">
            Every token is streamed. No hidden steps, no smoke — just the agent writing
            the same code you would have, faster.
          </p>
        </div>
        <MockIDE
          className="ide"
          filename="agent.ts"
          tokens={IDE_TOKENS}
          loop
          charMs={[24, 70]}
          thinkingLabel="Synthetica is writing…"
        />
      </section>

      <section className="playground" ref={convoRef}>
        <h2 className="section-title">
          <Sparkle solid /> Try the playground
        </h2>
        <p className="muted">
          Pick a model, send a prompt. The answer streams back token by token below.
        </p>

        <Prompt
          className="playground-prompt"
          placeholder="Ask Synthetica anything…"
          models={MODELS}
          defaultModel={MODELS[0]}
          rows={3}
          onSubmit={(value, ctx) => submit(value, ctx.model)}
        />

        {error && <p className="error">⚠ {error}</p>}

        <div className="convo">
          {messages.length === 0 && !busy && (
            <p className="convo-empty">Your conversation will appear here.</p>
          )}
          {messages.map((m) =>
            m.role === 'user' ? (
              <ChatBubble key={m.id} role="user">
                {m.text}
              </ChatBubble>
            ) : (
              <ChatBubble key={m.id} role="ai" agent="Synthetica" thinking={false}>
                <TokenStream text={m.text} speedMs={[14, 38]} />
              </ChatBubble>
            )
          )}
          {busy && (
            <ChatBubble role="ai" agent="Synthetica" thinking="reasoning…">
              <span className="dim">thinking…</span>
            </ChatBubble>
          )}
        </div>
      </section>

      <footer className="footer">
        <span className="brand">
          <Sparkle solid /> Synthetica
        </span>
        <span className="muted">
          Built with React 19 · Vite · performative-ui · OpenAI
        </span>
      </footer>
    </div>
  )
}
