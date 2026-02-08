import { useCallback } from 'react'
import { useParams, useNavigate } from '@tanstack/react-router'
import BuildProgress from '../components/BuildProgress'

export default function ProgressPage() {
  const { projectId } = useParams({ from: '/progress/$projectId' })
  const navigate = useNavigate()

  const onComplete = useCallback(() => {
    navigate({ to: '/preview/$projectId', params: { projectId } })
  }, [navigate, projectId])

  const onError = useCallback((msg: string) => {
    alert('Build failed: ' + msg)
    navigate({ to: '/' })
  }, [navigate])

  return <BuildProgress projectId={projectId} onComplete={onComplete} onError={onError} />
}
