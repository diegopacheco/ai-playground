import { useState } from 'react'
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter'
import yaml from 'react-syntax-highlighter/dist/esm/languages/hljs/yaml'
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs'

SyntaxHighlighter.registerLanguage('yaml', yaml)

interface Props {
  yamlContent: string
  title: string
  onClose: () => void
}

export default function YamlModal({ yamlContent, title, onClose }: Props) {
  const [fullscreen, setFullscreen] = useState(false)

  const modalStyle: React.CSSProperties = fullscreen
    ? { position: 'fixed', inset: 0, background: '#0d1117', zIndex: 1000, display: 'flex', flexDirection: 'column' }
    : { position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }

  const contentStyle: React.CSSProperties = fullscreen
    ? { flex: 1, overflow: 'auto', padding: 16 }
    : { background: '#161b22', borderRadius: 8, border: '1px solid #30363d', maxWidth: '80vw', maxHeight: '80vh', overflow: 'auto', width: '100%' }

  return (
    <div style={modalStyle} onClick={onClose}>
      <div style={contentStyle} onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', borderBottom: '1px solid #30363d', position: 'sticky', top: 0, background: '#161b22', zIndex: 1 }}>
          <span style={{ fontWeight: 600, color: '#58a6ff' }}>{title}</span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={() => setFullscreen(!fullscreen)} style={{ padding: '4px 12px', background: '#21262d', color: '#c9d1d9', border: '1px solid #30363d', borderRadius: 4, cursor: 'pointer', fontSize: 13 }}>
              {fullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
            </button>
            <button onClick={onClose} style={{ padding: '4px 12px', background: '#21262d', color: '#c9d1d9', border: '1px solid #30363d', borderRadius: 4, cursor: 'pointer', fontSize: 13 }}>
              Close
            </button>
          </div>
        </div>
        <SyntaxHighlighter language="yaml" style={atomOneDark} customStyle={{ margin: 0, padding: 16, background: 'transparent', fontSize: 13 }}>
          {yamlContent}
        </SyntaxHighlighter>
      </div>
    </div>
  )
}
