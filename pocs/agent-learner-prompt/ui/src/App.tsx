import { useState } from 'react'
import { Layout } from './components/Layout'
import { TaskForm } from './components/TaskForm'
import { CycleProgress } from './components/CycleProgress'
import { SessionSummary } from './components/SessionSummary'
import { ConfigPanel } from './components/ConfigPanel'
import { ProjectBrowser } from './components/ProjectBrowser'
import { ProjectDetail } from './components/ProjectDetail'
import { useTaskStatus } from './api/queries'
import { useSSE } from './hooks/useSSE'

export function App() {
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null)
  const [selectedProject, setSelectedProject] = useState<string | null>(null)
  const { data: taskStatus } = useTaskStatus(currentTaskId)
  const { events } = useSSE(currentTaskId)

  const handleTaskCreated = (taskId: string) => {
    setCurrentTaskId(taskId)
  }

  return (
    <Layout>
      {(activeTab) => {
        if (activeTab === 'tasks') {
          return (
            <>
              <TaskForm onTaskCreated={handleTaskCreated} />
              {currentTaskId && (
                <>
                  <CycleProgress status={taskStatus || null} events={events} />
                  <SessionSummary status={taskStatus || null} />
                </>
              )}
            </>
          )
        }
        if (activeTab === 'projects') {
          if (selectedProject) {
            return <ProjectDetail projectName={selectedProject} onBack={() => setSelectedProject(null)} />
          }
          return <ProjectBrowser onSelect={setSelectedProject} />
        }
        if (activeTab === 'config') {
          return <ConfigPanel />
        }
        return null
      }}
    </Layout>
  )
}
