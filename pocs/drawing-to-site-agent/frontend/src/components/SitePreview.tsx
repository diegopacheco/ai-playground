import { previewUrl } from '../api/client'

interface Props {
  projectId: string
}

export default function SitePreview({ projectId }: Props) {
  return (
    <iframe
      src={previewUrl(projectId)}
      style={{
        width: '100%',
        height: 'calc(100vh - 120px)',
        border: '1px solid #2a2a4a',
        borderRadius: '8px',
        background: '#fff',
      }}
      title="Site Preview"
    />
  )
}
