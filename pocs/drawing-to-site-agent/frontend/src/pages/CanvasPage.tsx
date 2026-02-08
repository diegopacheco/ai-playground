import { useParams, useNavigate } from '@tanstack/react-router'
import { useMutation } from '@tanstack/react-query'
import { createProject } from '../api/client'
import DrawingCanvas from '../components/DrawingCanvas'

export default function CanvasPage() {
  const { projectId } = useParams({ from: '/canvas/$projectId' })
  const navigate = useNavigate()

  let config: { name: string; engine: string }
  try {
    config = JSON.parse(decodeURIComponent(projectId))
  } catch {
    config = { name: 'untitled', engine: 'claude/opus' }
  }

  const mutation = useMutation({
    mutationFn: createProject,
    onSuccess: (data) => {
      navigate({ to: '/progress/$projectId', params: { projectId: data.id } })
    },
  })

  const handleCapture = (dataUrl: string) => {
    mutation.mutate({
      name: config.name,
      engine: config.engine,
      image: dataUrl,
    })
  }

  return <DrawingCanvas onCapture={handleCapture} />
}
