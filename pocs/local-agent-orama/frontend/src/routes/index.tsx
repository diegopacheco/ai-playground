import { useState, useEffect, useCallback } from 'react'
import PromptInput from '../components/PromptInput'
import AgentCard from '../components/AgentCard'
import AgentTabs from '../components/AgentTabs'
import FileBrowser from '../components/FileBrowser'
import CodeViewer from '../components/CodeViewer'
import { Agent, FileEntry, runAgents, getStatus, getFiles, getFileContent } from '../api/client'

const DEFAULT_AGENTS: Agent[] = [
  { name: 'Claude Code', model: 'claude-sonnet-4-20250514', status: 'pending', worktree: '' },
  { name: 'Codex', model: 'o4-mini', status: 'pending', worktree: '' },
  { name: 'Gemini', model: 'gemini-2.5-pro', status: 'pending', worktree: '' },
  { name: 'Copilot CLI', model: 'gpt-4o', status: 'pending', worktree: '' },
]

function IndexPage() {
  const [agents, setAgents] = useState<Agent[]>(DEFAULT_AGENTS)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [selectedAgent, setSelectedAgent] = useState('Claude Code')
  const [files, setFiles] = useState<FileEntry[]>([])
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [fileContent, setFileContent] = useState<string | null>(null)
  const [fileLanguage, setFileLanguage] = useState('text')

  const handleSubmit = async (prompt: string, projectName: string) => {
    setIsRunning(true)
    setAgents(DEFAULT_AGENTS.map((a) => ({ ...a, status: 'running' })))
    setSelectedFile(null)
    setFileContent(null)
    setFiles([])

    try {
      const response = await runAgents(prompt, projectName)
      setSessionId(response.session_id)
    } catch (error) {
      console.error('Failed to start agents:', error)
      setIsRunning(false)
      setAgents(DEFAULT_AGENTS.map((a) => ({ ...a, status: 'error' })))
    }
  }

  const pollStatus = useCallback(async () => {
    if (!sessionId) return
    try {
      const response = await getStatus(sessionId)
      setAgents(response.agents)
      const allDone = response.agents.every(
        (a) => a.status === 'done' || a.status === 'error' || a.status === 'timeout'
      )
      if (allDone) {
        setIsRunning(false)
      }
    } catch (error) {
      console.error('Failed to get status:', error)
    }
  }, [sessionId])

  useEffect(() => {
    if (!sessionId || !isRunning) return
    const interval = setInterval(pollStatus, 2000)
    return () => clearInterval(interval)
  }, [sessionId, isRunning, pollStatus])

  const loadFiles = useCallback(async () => {
    if (!sessionId) return
    const agentKey = selectedAgent.toLowerCase().replace(' ', '-')
    try {
      const response = await getFiles(sessionId, agentKey)
      setFiles(response.files)
    } catch (error) {
      console.error('Failed to load files:', error)
      setFiles([])
    }
  }, [sessionId, selectedAgent])

  useEffect(() => {
    loadFiles()
    setSelectedFile(null)
    setFileContent(null)
  }, [selectedAgent, loadFiles])

  const handleSelectFile = async (path: string) => {
    if (!sessionId) return
    setSelectedFile(path)
    const agentKey = selectedAgent.toLowerCase().replace(' ', '-')
    try {
      const response = await getFileContent(sessionId, agentKey, path)
      setFileContent(response.content)
      setFileLanguage(response.language)
    } catch (error) {
      console.error('Failed to load file content:', error)
      setFileContent('Error loading file')
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="py-8 text-center">
        <h1 className="text-4xl font-bold mb-2">Local Agent Orama</h1>
        <p className="text-slate-400">Run multiple AI coding assistants in parallel</p>
      </header>

      <main className="flex-1 px-4 pb-8">
        <div className="mb-8">
          <PromptInput onSubmit={handleSubmit} disabled={isRunning} />
        </div>

        <div className="grid grid-cols-4 gap-4 mb-8 max-w-5xl mx-auto">
          {agents.map((agent) => (
            <AgentCard
              key={agent.name}
              agent={agent}
              isSelected={selectedAgent === agent.name}
              onClick={() => setSelectedAgent(agent.name)}
            />
          ))}
        </div>

        {sessionId && (
          <div className="max-w-6xl mx-auto bg-slate-800 rounded-lg overflow-hidden">
            <AgentTabs
              agents={agents}
              selectedAgent={selectedAgent}
              onSelectAgent={setSelectedAgent}
            />
            <div className="flex h-96">
              <div className="w-64 flex-shrink-0">
                <FileBrowser
                  files={files}
                  selectedFile={selectedFile}
                  onSelectFile={handleSelectFile}
                />
              </div>
              <div className="flex-1">
                <CodeViewer
                  content={fileContent}
                  language={fileLanguage}
                  filePath={selectedFile}
                />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default IndexPage
